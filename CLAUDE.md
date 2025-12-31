# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TryLM is a small (~10-17M parameter) GPT-style language model built from scratch in PyTorch for educational purposes. It trains on the TinyStories dataset and uses modern LLM techniques.

## Commands

All commands use `uv run` as the project manager:

```bash
# Train tokenizer (required before first training run)
uv run trylm tokenizer train --vocab-size 16384

# Train model
uv run trylm train --config configs/tiny_stories.yaml

# Train with custom parameters
uv run trylm train --batch-size 32 --max-steps 5000 --learning-rate 1e-4

# Resume training from checkpoint
uv run trylm train --resume checkpoints/step_5000

# Generate text
uv run trylm generate "Once upon a time" --temperature 0.8 --top-p 0.9

# Interactive generation mode
uv run trylm generate -i

# Evaluate model perplexity
uv run trylm eval --checkpoint checkpoints/best

# Show model info
uv run trylm info --checkpoint checkpoints/best

# Sync dependencies
uv sync
```

## Architecture

### Model (`src/trylm/model.py`)
GPT decoder-only transformer with modern techniques:
- **RMSNorm** instead of LayerNorm (more efficient)
- **Rotary Position Embeddings (RoPE)** instead of learned positional embeddings
- **SwiGLU activation** in feed-forward layers (better than GELU)
- **Pre-normalization** (norm before attention/FFN, not after)
- **Weight tying** between token embeddings and output projection

Layer hierarchy: `GPT` → `TransformerBlock` → `MultiHeadSelfAttention` + `FeedForward`

### Configuration (`src/trylm/config.py`)
Three dataclasses that can be loaded from YAML:
- `TokenizerConfig`: vocab size, special tokens
- `ModelConfig`: architecture (layers, heads, dims, context length)
- `TrainConfig`: hyperparameters, W&B settings, checkpointing

Use `Config.from_yaml()` and `Config.to_yaml()` for serialization.

### Data Flow
1. `tokenizer.py`: Train BPE tokenizer using HuggingFace `tokenizers` library
2. `data.py`: `TinyStoriesDataset` tokenizes and chunks text into context windows
3. `trainer.py`: Training loop with mixed precision, gradient accumulation, W&B logging
4. `generate.py`: Autoregressive generation with top-k, top-p, temperature sampling

### Checkpointing
Checkpoints saved to `checkpoints/` contain:
- `model.safetensors`: Model weights
- `training_state.pt`: Optimizer, scheduler, step count
- `config.yaml`: Full configuration for reproducibility

## Key Patterns

- Model forward returns `{"logits": ..., "loss": ...}` dict when labels provided
- Labels use `-100` for ignored tokens (PyTorch cross-entropy convention)
- Attention mask: `1` = attend, `0` = ignore
- All configs use dataclasses with sensible defaults
- CLI is Typer-based with `trylm` as the entry point
