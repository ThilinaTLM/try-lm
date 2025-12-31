"""Configuration dataclasses for TryLM."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class TokenizerConfig:
    """Configuration for BPE tokenizer training."""

    vocab_size: int = 16384
    min_frequency: int = 2
    special_tokens: list[str] = field(
        default_factory=lambda: ["<|pad|>", "<|eos|>", "<|bos|>", "<|unk|>"]
    )
    save_path: Path = Path("data/tokenizer")

    @property
    def pad_token(self) -> str:
        return "<|pad|>"

    @property
    def eos_token(self) -> str:
        return "<|eos|>"

    @property
    def bos_token(self) -> str:
        return "<|bos|>"

    @property
    def unk_token(self) -> str:
        return "<|unk|>"


@dataclass
class ModelConfig:
    """Configuration for the GPT model architecture."""

    vocab_size: int = 16384
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: int | None = None  # Defaults to 4 * d_model
    context_length: int = 512
    dropout: float = 0.1
    bias: bool = False  # Following GPT-2/3 - no bias in attention/FFN

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        # Validate head dimension
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def count_parameters(self) -> dict[str, int]:
        """Estimate parameter count for each component."""
        # Token embeddings: vocab_size * d_model
        token_emb = self.vocab_size * self.d_model

        # Position embeddings: context_length * d_model
        pos_emb = self.context_length * self.d_model

        # Per transformer block:
        # - Layer norm 1: 2 * d_model (weight + bias)
        # - Attention: 4 * d_model * d_model (Q, K, V, O projections)
        # - Layer norm 2: 2 * d_model
        # - FFN: d_model * d_ff + d_ff * d_model = 2 * d_model * d_ff
        ln_params = 2 * self.d_model  # weight only if no bias
        attn_params = 4 * self.d_model * self.d_model
        ffn_params = 2 * self.d_model * self.d_ff
        block_params = 2 * ln_params + attn_params + ffn_params

        # Final layer norm
        final_ln = self.d_model

        # Output projection (weight tied with token embeddings usually)
        # We'll count it separately for clarity
        output_proj = 0  # Tied with token embeddings

        total = token_emb + pos_emb + self.n_layers * block_params + final_ln + output_proj

        return {
            "token_embeddings": token_emb,
            "position_embeddings": pos_emb,
            "transformer_blocks": self.n_layers * block_params,
            "final_layer_norm": final_ln,
            "total": total,
        }


@dataclass
class TrainConfig:
    """Configuration for training."""

    # Data
    dataset_name: str = "roneneldan/TinyStories"
    train_split: str = "train"
    val_split: str = "validation"

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 10000
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Evaluation
    eval_interval: int = 500
    eval_steps: int = 100
    log_interval: int = 10

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_interval: int = 1000
    keep_last_n: int = 3

    # Hardware
    mixed_precision: bool = True
    compile: bool = True
    seed: int = 42

    # W&B
    wandb_project: str = "trylm"
    wandb_run_name: str | None = None


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        tokenizer_cfg = TokenizerConfig(**data.get("tokenizer", {}))
        model_cfg = ModelConfig(**data.get("model", {}))
        train_cfg = TrainConfig(**data.get("train", {}))

        # Sync vocab size between tokenizer and model
        if "vocab_size" in data.get("tokenizer", {}):
            model_cfg.vocab_size = tokenizer_cfg.vocab_size

        return cls(tokenizer=tokenizer_cfg, model=model_cfg, train=train_cfg)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = {
            "tokenizer": {
                "vocab_size": self.tokenizer.vocab_size,
                "min_frequency": self.tokenizer.min_frequency,
                "special_tokens": self.tokenizer.special_tokens,
                "save_path": str(self.tokenizer.save_path),
            },
            "model": {
                "vocab_size": self.model.vocab_size,
                "n_layers": self.model.n_layers,
                "n_heads": self.model.n_heads,
                "d_model": self.model.d_model,
                "d_ff": self.model.d_ff,
                "context_length": self.model.context_length,
                "dropout": self.model.dropout,
                "bias": self.model.bias,
            },
            "train": {
                "dataset_name": self.train.dataset_name,
                "batch_size": self.train.batch_size,
                "gradient_accumulation_steps": self.train.gradient_accumulation_steps,
                "learning_rate": self.train.learning_rate,
                "weight_decay": self.train.weight_decay,
                "max_steps": self.train.max_steps,
                "warmup_steps": self.train.warmup_steps,
                "grad_clip": self.train.grad_clip,
                "eval_interval": self.train.eval_interval,
                "eval_steps": self.train.eval_steps,
                "log_interval": self.train.log_interval,
                "checkpoint_dir": str(self.train.checkpoint_dir),
                "save_interval": self.train.save_interval,
                "mixed_precision": self.train.mixed_precision,
                "compile": self.train.compile,
                "seed": self.train.seed,
                "wandb_project": self.train.wandb_project,
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
