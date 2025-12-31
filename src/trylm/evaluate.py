"""Evaluation utilities for TryLM."""

import math
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from trylm.config import Config
from trylm.model import GPT
from trylm.tokenizer import TryLMTokenizer


console = Console()


@torch.no_grad()
def compute_perplexity(
    model: GPT,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    mixed_precision: bool = True,
) -> dict[str, float]:
    """Compute perplexity on a dataset.

    Args:
        model: Trained GPT model
        dataloader: DataLoader for evaluation data
        device: Device to run on
        max_batches: Maximum number of batches to evaluate (None for all)
        mixed_precision: Whether to use mixed precision

    Returns:
        Dictionary with loss and perplexity
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in dataloader:
        if max_batches is not None and num_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=mixed_precision):
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]

        # Count non-padding tokens
        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        num_batches += 1

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "num_batches": num_batches,
    }


def evaluate_model(
    model: GPT,
    tokenizer: TryLMTokenizer,
    config: Config,
    device: torch.device | None = None,
    split: str = "validation",
    max_batches: int | None = None,
) -> dict[str, float]:
    """Full model evaluation.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        config: Configuration
        device: Device to run on
        split: Dataset split to evaluate on
        max_batches: Maximum number of batches

    Returns:
        Evaluation metrics
    """
    from trylm.data import TinyStoriesDataset
    from torch.utils.data import DataLoader

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Create dataset
    dataset = TinyStoriesDataset(
        tokenizer=tokenizer,
        split=split,
        context_length=config.model.context_length,
        dataset_name=config.train.dataset_name,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    console.print(f"[bold blue]Evaluating on {split} set...[/]")

    metrics = compute_perplexity(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
        mixed_precision=config.train.mixed_precision,
    )

    # Print results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Loss", f"{metrics['loss']:.4f}")
    table.add_row("Perplexity", f"{metrics['perplexity']:.2f}")
    table.add_row("Total Tokens", f"{metrics['total_tokens']:,}")
    table.add_row("Batches Evaluated", f"{metrics['num_batches']}")

    console.print(table)

    return metrics


def load_model_for_eval(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[GPT, TryLMTokenizer, Config]:
    """Load model from checkpoint for evaluation.

    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, config)
    """
    from safetensors.torch import load_file

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config = Config.from_yaml(checkpoint_path / "config.yaml")

    # Load tokenizer
    tokenizer = TryLMTokenizer.load(config.tokenizer.save_path)

    # Create and load model
    model = GPT.from_config(config.model)
    state_dict = load_file(checkpoint_path / "model.safetensors")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    console.print(f"[bold green]Loaded model from {checkpoint_path}[/]")
    console.print(f"[bold blue]Parameters:[/] {model.count_parameters():,}")

    return model, tokenizer, config
