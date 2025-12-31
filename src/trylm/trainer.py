"""Training loop with W&B integration for TryLM."""

import math
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from safetensors.torch import save_file, load_file
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import wandb

from trylm.config import Config, ModelConfig, TrainConfig
from trylm.model import GPT
from trylm.tokenizer import TryLMTokenizer


console = Console()


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a cosine learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / warmup_steps
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Trainer for GPT model with W&B logging."""

    def __init__(
        self,
        model: GPT,
        tokenizer: TryLMTokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: torch.device | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: GPT model to train
            tokenizer: Tokenizer for sample generation
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Full configuration
            device: Device to train on (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.train_config = config.train

        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Move model to device
        self.model = self.model.to(device)

        # Compile model if requested
        if self.train_config.compile and hasattr(torch, "compile"):
            console.print("[bold blue]Compiling model with torch.compile...[/]")
            self.model = torch.compile(self.model)

        # Set up optimizer
        self.optimizer = self._configure_optimizer()

        # Set up scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=self.train_config.warmup_steps,
            total_steps=self.train_config.max_steps,
        )

        # Mixed precision
        self.scaler = GradScaler() if self.train_config.mixed_precision else None

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Create checkpoint directory
        self.train_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _configure_optimizer(self) -> AdamW:
        """Configure optimizer with weight decay for non-layernorm/bias params."""
        # Separate parameters that should have weight decay
        decay_params = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Don't apply weight decay to bias, layernorm, or embedding
            if "bias" in name or "norm" in name or "embedding" in name:
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.train_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            optim_groups,
            lr=self.train_config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def _train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass with mixed precision
        with autocast(enabled=self.train_config.mixed_precision):
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs["loss"] / self.train_config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.train_config.gradient_accumulation_steps

    def _optimizer_step(self) -> float:
        """Perform optimizer step with gradient clipping."""
        # Unscale gradients for clipping
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.train_config.grad_clip,
        )

        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Scheduler step
        self.scheduler.step()

        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)

        return grad_norm.item()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        val_iter = iter(self.val_loader)
        for _ in range(self.train_config.eval_steps):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(self.val_loader)
                batch = next(val_iter)

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast(enabled=self.train_config.mixed_precision):
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs["loss"]

            # Count non-padding tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    @torch.no_grad()
    def generate_samples(self, prompts: list[str], max_new_tokens: int = 100) -> list[str]:
        """Generate samples for logging."""
        from trylm.generate import generate

        self.model.eval()

        samples = []
        for prompt in prompts:
            text = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.9,
                device=self.device,
            )
            samples.append(text)

        return samples

    def save_checkpoint(self, path: Path | None = None) -> Path:
        """Save model checkpoint."""
        if path is None:
            path = self.train_config.checkpoint_dir / f"step_{self.global_step}"

        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        state_dict = self.model.state_dict()
        # Handle compiled model
        if hasattr(self.model, "_orig_mod"):
            state_dict = self.model._orig_mod.state_dict()

        save_file(state_dict, path / "model.safetensors")

        # Save optimizer and scheduler state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "scaler": self.scaler.state_dict() if self.scaler else None,
            },
            path / "training_state.pt",
        )

        # Save config
        self.config.to_yaml(path / "config.yaml")

        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        # Load model weights
        state_dict = load_file(path / "model.safetensors")

        # Handle compiled model
        if hasattr(self.model, "_orig_mod"):
            self.model._orig_mod.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        # Load training state
        training_state = torch.load(path / "training_state.pt", map_location=self.device)
        self.optimizer.load_state_dict(training_state["optimizer"])
        self.scheduler.load_state_dict(training_state["scheduler"])
        self.global_step = training_state["global_step"]
        self.best_val_loss = training_state["best_val_loss"]

        if self.scaler and training_state["scaler"]:
            self.scaler.load_state_dict(training_state["scaler"])

        console.print(f"[bold green]Loaded checkpoint from step {self.global_step}[/]")

    def train(self, resume_from: Path | None = None) -> None:
        """Run training loop."""
        # Resume from checkpoint if provided
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        # Initialize W&B
        wandb.init(
            project=self.train_config.wandb_project,
            name=self.train_config.wandb_run_name,
            config={
                "model": self.config.model.__dict__,
                "train": self.config.train.__dict__,
                "tokenizer": {
                    "vocab_size": self.config.tokenizer.vocab_size,
                },
            },
        )

        # Log model architecture
        wandb.watch(self.model, log="gradients", log_freq=100)

        console.print(f"[bold green]Starting training from step {self.global_step}[/]")
        console.print(f"[bold blue]Device:[/] {self.device}")
        console.print(f"[bold blue]Model parameters:[/] {self.model.count_parameters():,}")
        console.print(f"[bold blue]Max steps:[/] {self.train_config.max_steps}")

        # Sample prompts for generation
        sample_prompts = [
            "Once upon a time",
            "The little girl",
            "One day, a boy named",
        ]

        # Training loop
        train_iter = iter(self.train_loader)
        accumulated_loss = 0.0
        step_start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Training...",
                total=self.train_config.max_steps - self.global_step,
            )

            while self.global_step < self.train_config.max_steps:
                # Get batch
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                # Training step
                loss = self._train_step(batch)
                accumulated_loss += loss

                # Gradient accumulation
                if (self.global_step + 1) % self.train_config.gradient_accumulation_steps == 0:
                    grad_norm = self._optimizer_step()
                    avg_loss = accumulated_loss / self.train_config.gradient_accumulation_steps

                    # Logging
                    if self.global_step % self.train_config.log_interval == 0:
                        step_time = time.time() - step_start_time
                        tokens_per_sec = (
                            self.train_config.batch_size
                            * self.config.model.context_length
                            * self.train_config.gradient_accumulation_steps
                            / step_time
                        )

                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/perplexity": math.exp(avg_loss),
                                "train/grad_norm": grad_norm,
                                "train/learning_rate": self.scheduler.get_last_lr()[0],
                                "train/tokens_per_sec": tokens_per_sec,
                            },
                            step=self.global_step,
                        )

                        progress.update(
                            task,
                            description=f"Loss: {avg_loss:.4f} | PPL: {math.exp(avg_loss):.2f}",
                        )

                        step_start_time = time.time()

                    accumulated_loss = 0.0

                # Evaluation
                if self.global_step > 0 and self.global_step % self.train_config.eval_interval == 0:
                    eval_metrics = self.evaluate()

                    wandb.log(
                        {
                            "eval/loss": eval_metrics["val_loss"],
                            "eval/perplexity": eval_metrics["val_perplexity"],
                        },
                        step=self.global_step,
                    )

                    console.print(
                        f"\n[bold]Step {self.global_step}[/] | "
                        f"Val Loss: {eval_metrics['val_loss']:.4f} | "
                        f"Val PPL: {eval_metrics['val_perplexity']:.2f}"
                    )

                    # Generate samples
                    samples = self.generate_samples(sample_prompts)
                    sample_table = wandb.Table(
                        columns=["prompt", "generated"],
                        data=list(zip(sample_prompts, samples)),
                    )
                    wandb.log({"samples": sample_table}, step=self.global_step)

                    # Save best model
                    if eval_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = eval_metrics["val_loss"]
                        self.save_checkpoint(self.train_config.checkpoint_dir / "best")
                        console.print("[bold green]Saved best model![/]")

                # Save checkpoint
                if self.global_step > 0 and self.global_step % self.train_config.save_interval == 0:
                    self.save_checkpoint()
                    console.print(f"[bold blue]Saved checkpoint at step {self.global_step}[/]")

                self.global_step += 1
                progress.advance(task)

        # Final save
        self.save_checkpoint(self.train_config.checkpoint_dir / "final")
        console.print("[bold green]Training complete![/]")

        wandb.finish()
