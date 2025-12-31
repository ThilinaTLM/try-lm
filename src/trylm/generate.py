"""Text generation utilities for TryLM."""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from trylm.model import GPT
from trylm.tokenizer import TryLMTokenizer


def top_k_filtering(logits: Tensor, top_k: int) -> Tensor:
    """Filter logits to keep only top-k tokens.

    Args:
        logits: Logits tensor of shape [..., vocab_size]
        top_k: Number of top tokens to keep

    Returns:
        Filtered logits with -inf for tokens outside top-k
    """
    if top_k <= 0:
        return logits

    # Get the top-k values and indices
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[..., -1, None]

    # Set all logits below the minimum top-k value to -inf
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def top_p_filtering(logits: Tensor, top_p: float) -> Tensor:
    """Filter logits using nucleus (top-p) sampling.

    Keep the smallest set of tokens whose cumulative probability exceeds top_p.

    Args:
        logits: Logits tensor of shape [..., vocab_size]
        top_p: Cumulative probability threshold

    Returns:
        Filtered logits with -inf for tokens outside nucleus
    """
    if top_p >= 1.0:
        return logits

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Find where cumulative probability exceeds top_p
    sorted_indices_to_remove = cumulative_probs > top_p

    # Shift the mask right to keep the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Create mask in original order
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )

    return torch.where(indices_to_remove, torch.full_like(logits, float("-inf")), logits)


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: TryLMTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: torch.device | None = None,
    stop_token: str | None = None,
) -> str:
    """Generate text from a prompt.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top-k tokens (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: Penalty for repeating tokens (1.0 = disabled)
        device: Device to run on
        stop_token: Stop generation when this token is generated

    Returns:
        Generated text including prompt
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Get stop token id
    stop_id = tokenizer.eos_id
    if stop_token is not None:
        stop_ids = tokenizer.encode(stop_token, add_special_tokens=False)
        if stop_ids:
            stop_id = stop_ids[0]

    # Track generated tokens for repetition penalty
    generated_tokens: list[int] = []

    for _ in range(max_new_tokens):
        # Truncate to context length
        if input_ids.size(1) > model.config.context_length:
            input_ids = input_ids[:, -model.config.context_length:]

        # Forward pass
        outputs = model(input_ids)
        logits = outputs["logits"][:, -1, :]  # [batch, vocab_size]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply repetition penalty
        if repetition_penalty != 1.0 and generated_tokens:
            for token_id in set(generated_tokens):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Apply filtering
        filtered_logits = top_k_filtering(logits, top_k)
        filtered_logits = top_p_filtering(filtered_logits, top_p)

        # Sample from distribution
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        generated_tokens.append(next_token.item())

        # Check for stop token
        if next_token.item() == stop_id:
            break

    # Decode
    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


@torch.no_grad()
def generate_batch(
    model: GPT,
    tokenizer: TryLMTokenizer,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: torch.device | None = None,
) -> list[str]:
    """Generate text from multiple prompts in parallel.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        prompts: List of input prompts
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Keep only top-k tokens
        top_p: Nucleus sampling threshold
        device: Device to run on

    Returns:
        List of generated texts
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Encode prompts
    encoded = tokenizer.encode_batch(prompts, add_special_tokens=True)

    # Pad to same length
    max_len = max(len(ids) for ids in encoded)
    padded = [ids + [tokenizer.pad_id] * (max_len - len(ids)) for ids in encoded]

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = (input_ids != tokenizer.pad_id).long()

    # Track which sequences are finished
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    eos_id = tokenizer.eos_id

    for _ in range(max_new_tokens):
        # Truncate to context length
        if input_ids.size(1) > model.config.context_length:
            input_ids = input_ids[:, -model.config.context_length:]
            attention_mask = attention_mask[:, -model.config.context_length:]

        # Forward pass
        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"][:, -1, :]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply filtering
        filtered_logits = top_k_filtering(logits, top_k)
        filtered_logits = top_p_filtering(filtered_logits, top_p)

        # Sample from distribution
        probs = F.softmax(filtered_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        # For finished sequences, use padding
        next_tokens = torch.where(
            finished.unsqueeze(1),
            torch.full_like(next_tokens, tokenizer.pad_id),
            next_tokens,
        )

        # Append to sequence
        input_ids = torch.cat([input_ids, next_tokens], dim=1)
        attention_mask = torch.cat(
            [attention_mask, (~finished).long().unsqueeze(1)],
            dim=1,
        )

        # Check for EOS
        finished = finished | (next_tokens.squeeze(-1) == eos_id)

        if finished.all():
            break

    # Decode
    results = []
    for i in range(len(prompts)):
        output_ids = input_ids[i].tolist()
        results.append(tokenizer.decode(output_ids, skip_special_tokens=True))

    return results


def interactive_generate(
    model: GPT,
    tokenizer: TryLMTokenizer,
    device: torch.device | None = None,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_new_tokens: int = 200,
) -> None:
    """Interactive text generation loop.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        device: Device to run on
        temperature: Default sampling temperature
        top_p: Default nucleus sampling threshold
        max_new_tokens: Default max tokens to generate
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    console.print(Panel.fit(
        "[bold green]Interactive Text Generation[/]\n\n"
        "Enter a prompt to generate text.\n"
        "Commands:\n"
        "  :temp <value>  - Set temperature\n"
        "  :top_p <value> - Set top_p\n"
        "  :max <value>   - Set max tokens\n"
        "  :quit          - Exit\n",
        title="TryLM",
    ))

    while True:
        try:
            prompt = console.input("\n[bold cyan]Prompt:[/] ")
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt.strip():
            continue

        # Handle commands
        if prompt.startswith(":"):
            parts = prompt.split()
            cmd = parts[0].lower()

            if cmd == ":quit":
                break
            elif cmd == ":temp" and len(parts) > 1:
                temperature = float(parts[1])
                console.print(f"[dim]Temperature set to {temperature}[/]")
                continue
            elif cmd == ":top_p" and len(parts) > 1:
                top_p = float(parts[1])
                console.print(f"[dim]Top-p set to {top_p}[/]")
                continue
            elif cmd == ":max" and len(parts) > 1:
                max_new_tokens = int(parts[1])
                console.print(f"[dim]Max tokens set to {max_new_tokens}[/]")
                continue
            else:
                console.print("[red]Unknown command[/]")
                continue

        # Generate
        with console.status("[bold green]Generating..."):
            text = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )

        console.print(Panel(text, title="Generated", border_style="green"))

    console.print("[bold green]Goodbye![/]")
