"""Typer CLI for TryLM."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="trylm",
    help="TryLM - A small language model from scratch for learning",
    add_completion=False,
)
tokenizer_app = typer.Typer(help="Tokenizer commands")
app.add_typer(tokenizer_app, name="tokenizer")

console = Console()


def get_device():
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@tokenizer_app.command("train")
def train_tokenizer(
    vocab_size: int = typer.Option(16384, help="Vocabulary size"),
    dataset: str = typer.Option("roneneldan/TinyStories", help="Dataset name"),
    output: Path = typer.Option(Path("data/tokenizer"), help="Output directory"),
    num_samples: Optional[int] = typer.Option(None, help="Number of samples to train on"),
):
    """Train a BPE tokenizer on a dataset."""
    from trylm.config import TokenizerConfig
    from trylm.tokenizer import train_tokenizer_on_dataset

    config = TokenizerConfig(vocab_size=vocab_size, save_path=output)

    tokenizer = train_tokenizer_on_dataset(
        dataset_name=dataset,
        config=config,
        num_samples=num_samples,
    )

    tokenizer.save(output)
    console.print(f"[bold green]Tokenizer saved to {output}[/]")


@tokenizer_app.command("encode")
def encode_text(
    text: str = typer.Argument(..., help="Text to encode"),
    tokenizer_path: Path = typer.Option(Path("data/tokenizer"), help="Tokenizer path"),
):
    """Encode text using the tokenizer."""
    from trylm.tokenizer import TryLMTokenizer

    tokenizer = TryLMTokenizer.load(tokenizer_path)
    ids = tokenizer.encode(text)

    table = Table(title="Tokenization Result")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Input text", text)
    table.add_row("Token IDs", str(ids))
    table.add_row("Num tokens", str(len(ids)))
    table.add_row("Decoded", tokenizer.decode(ids))

    console.print(table)


@app.command()
def train(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Config YAML file"),
    resume: Optional[Path] = typer.Option(None, help="Resume from checkpoint"),
    batch_size: int = typer.Option(64, help="Batch size"),
    max_steps: int = typer.Option(10000, help="Maximum training steps"),
    learning_rate: float = typer.Option(3e-4, help="Learning rate"),
    wandb_project: str = typer.Option("trylm", help="W&B project name"),
    wandb_run: Optional[str] = typer.Option(None, help="W&B run name"),
    no_compile: bool = typer.Option(False, help="Disable torch.compile"),
    streaming: bool = typer.Option(False, help="Use streaming dataset"),
):
    """Train the language model."""
    import torch
    from trylm.config import Config, ModelConfig, TrainConfig, TokenizerConfig
    from trylm.data import create_dataloaders
    from trylm.model import GPT
    from trylm.tokenizer import TryLMTokenizer
    from trylm.trainer import Trainer

    device = get_device()
    console.print(f"[bold blue]Using device:[/] {device}")

    # Load or create config
    if config_path and config_path.exists():
        config = Config.from_yaml(config_path)
        console.print(f"[bold blue]Loaded config from:[/] {config_path}")
    else:
        config = Config()

    # Override with CLI args
    config.train.batch_size = batch_size
    config.train.max_steps = max_steps
    config.train.learning_rate = learning_rate
    config.train.wandb_project = wandb_project
    config.train.wandb_run_name = wandb_run
    config.train.compile = not no_compile

    # Load tokenizer
    tokenizer_path = config.tokenizer.save_path
    if not (tokenizer_path / "tokenizer.json").exists():
        console.print("[bold yellow]Tokenizer not found. Training tokenizer first...[/]")
        from trylm.tokenizer import train_tokenizer_on_dataset
        tokenizer = train_tokenizer_on_dataset(config=config.tokenizer)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = TryLMTokenizer.load(tokenizer_path)
        console.print(f"[bold green]Loaded tokenizer from {tokenizer_path}[/]")

    # Update model vocab size
    config.model.vocab_size = tokenizer.vocab_size

    # Print config summary
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Vocab size", str(config.model.vocab_size))
    table.add_row("Model dim", str(config.model.d_model))
    table.add_row("Layers", str(config.model.n_layers))
    table.add_row("Heads", str(config.model.n_heads))
    table.add_row("Context length", str(config.model.context_length))
    table.add_row("Batch size", str(config.train.batch_size))
    table.add_row("Grad accumulation", str(config.train.gradient_accumulation_steps))
    table.add_row("Learning rate", str(config.train.learning_rate))
    table.add_row("Max steps", str(config.train.max_steps))

    console.print(table)

    # Create dataloaders
    console.print("[bold blue]Loading dataset...[/]")
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        model_config=config.model,
        train_config=config.train,
        streaming=streaming,
    )

    # Create model
    model = GPT.from_config(config.model)
    console.print(f"[bold blue]Model parameters:[/] {model.count_parameters():,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Train
    trainer.train(resume_from=resume)


@app.command()
def generate(
    prompt: str = typer.Argument("Once upon a time", help="Prompt to generate from"),
    checkpoint: Path = typer.Option(Path("checkpoints/best"), help="Checkpoint path"),
    max_tokens: int = typer.Option(200, help="Max tokens to generate"),
    temperature: float = typer.Option(0.8, help="Sampling temperature"),
    top_k: int = typer.Option(50, help="Top-k sampling (0 to disable)"),
    top_p: float = typer.Option(0.9, help="Nucleus sampling threshold"),
    interactive: bool = typer.Option(False, "-i", help="Interactive mode"),
):
    """Generate text from a trained model."""
    from trylm.evaluate import load_model_for_eval
    from trylm.generate import generate as gen_text, interactive_generate

    device = get_device()

    # Load model
    model, tokenizer, config = load_model_for_eval(checkpoint, device)

    if interactive:
        interactive_generate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
        )
    else:
        # Generate
        with console.status("[bold green]Generating..."):
            text = gen_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
            )

        console.print(Panel(text, title="Generated Text", border_style="green"))


@app.command("eval")
def evaluate(
    checkpoint: Path = typer.Option(Path("checkpoints/best"), help="Checkpoint path"),
    split: str = typer.Option("validation", help="Dataset split to evaluate"),
    max_batches: Optional[int] = typer.Option(None, help="Max batches to evaluate"),
):
    """Evaluate a trained model."""
    from trylm.evaluate import load_model_for_eval, evaluate_model

    device = get_device()

    # Load model
    model, tokenizer, config = load_model_for_eval(checkpoint, device)

    # Evaluate
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        split=split,
        max_batches=max_batches,
    )


@app.command()
def info(
    checkpoint: Optional[Path] = typer.Option(None, help="Checkpoint path"),
):
    """Show model and training information."""
    from trylm.config import Config, ModelConfig

    if checkpoint and checkpoint.exists():
        config = Config.from_yaml(checkpoint / "config.yaml")

        console.print(Panel.fit(
            f"[bold]Checkpoint:[/] {checkpoint}\n"
            f"[bold]Vocab size:[/] {config.model.vocab_size}\n"
            f"[bold]Layers:[/] {config.model.n_layers}\n"
            f"[bold]Heads:[/] {config.model.n_heads}\n"
            f"[bold]Model dim:[/] {config.model.d_model}\n"
            f"[bold]Context length:[/] {config.model.context_length}\n",
            title="Model Info",
        ))

        # Estimate parameters
        params = config.model.count_parameters()
        table = Table(title="Parameter Count")
        table.add_column("Component", style="cyan")
        table.add_column("Parameters", style="green")

        for name, count in params.items():
            table.add_row(name.replace("_", " ").title(), f"{count:,}")

        console.print(table)
    else:
        # Show default config
        config = ModelConfig()
        params = config.count_parameters()

        console.print(Panel.fit(
            f"[bold]Default Configuration[/]\n\n"
            f"Vocab size: {config.vocab_size}\n"
            f"Layers: {config.n_layers}\n"
            f"Heads: {config.n_heads}\n"
            f"Model dim: {config.d_model}\n"
            f"FFN dim: {config.d_ff}\n"
            f"Context length: {config.context_length}\n"
            f"[bold]Total parameters:[/] ~{params['total']:,}",
            title="TryLM Default Model",
        ))


@app.command()
def init(
    output: Path = typer.Option(Path("configs/tiny_stories.yaml"), help="Output config path"),
):
    """Initialize a default configuration file."""
    from trylm.config import Config

    output.parent.mkdir(parents=True, exist_ok=True)
    config = Config()
    config.to_yaml(output)
    console.print(f"[bold green]Created config at {output}[/]")


if __name__ == "__main__":
    app()
