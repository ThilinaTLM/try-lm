"""GPT model architecture implemented from scratch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trylm.config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm and used in modern LLMs like LLaMA.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Calculate RMS
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Used in modern LLMs for better position encoding.
    Encodes position information directly into attention scores.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Compute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for all positions
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int) -> None:
        """Precompute cos/sin for efficiency."""
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        """Return cos and sin for the given sequence length."""
        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # Add batch dimension to cos/sin
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with Rotary Position Embeddings.

    Implements scaled dot-product attention with causal masking.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Rotary embeddings
        self.rotary_emb = RotaryPositionalEmbedding(
            dim=config.head_dim,
            max_seq_len=config.context_length,
        )

        # Causal mask (will be created on first forward pass)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1).bool(),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            attention_mask: Optional mask of shape [batch, seq_len]

        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary position embeddings
        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """Feed-Forward Network with SwiGLU activation.

    SwiGLU is used in modern LLMs like LLaMA and PaLM for better performance.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # SwiGLU uses 2/3 of the hidden dimension for each gate
        hidden_dim = int(2 * config.d_ff / 3)
        # Round to nearest multiple of 256 for efficiency
        hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with SwiGLU activation."""
        # SwiGLU: swish(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        output = self.down_proj(hidden)
        return output


class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Pre-normalization
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

        # Attention and FFN
        self.attention = MultiHeadSelfAttention(config)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with residual connections."""
        # Attention with residual
        x = x + self.attention(self.attn_norm(x), attention_mask)

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x


class GPT(nn.Module):
    """GPT-style language model.

    Features:
    - Token and rotary position embeddings
    - Pre-normalization transformer blocks
    - Weight tying between token embeddings and output
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = RMSNorm(config.d_model)

        # Output projection (weight tied with token embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Special scaled initialization for output projections
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape [batch, seq_len]
            attention_mask: Attention mask of shape [batch, seq_len]
            labels: Target token IDs of shape [batch, seq_len]

        Returns:
            Dictionary with:
            - logits: Output logits of shape [batch, seq_len, vocab_size]
            - loss: Cross entropy loss (if labels provided)
        """
        # Token embeddings
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final normalization
        x = self.final_norm(x)

        # Output logits
        logits = self.lm_head(x)

        result = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "GPT":
        """Create model from configuration."""
        return cls(config)
