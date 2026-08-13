"""Misawantiga Fry — v1 architecture. Frozen: do not modify after shipping"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from misawantiga.config.defaults import Config
from training.loss import calculate_loss


ARCH_FIELDS = (
    "vocab_size",
    "block_size",
    "n_layer",
    "n_head",
    "d_model",
    "dropout",
    "bias"
)


class CasualSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.d_model % config.n_head == 0

        self.d_model = config.d_model
        self.n_head = config.n_head
        self.head_dim = self.d_model // config.n_head

        # Combined Q, V, K projection layer for efficiency
        # Produces a tensor of size (batch, seq_len, 3 * d_model)
        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model, bias=False)

        # Output projection back to the embedding space
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # batch_size, seq_len, channels (d_model)

        # 1. Project x to Q, K, V and split them
        # (B, T, C) -> (B, T, 3 * C)
        qkv = self.c_attn(x)

        # Split into separate q, k, v tensors: each of shape (B, T, C)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape for Multi-Head Attention:
        # (B, T, C) -> (B, T, n_head, head_dim) -> transpose to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Causal Attention
        # is_causal=True automatically applies the triangular masking so tokens can't look ahead
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.0
        )

        # 4. Re-assemble all head outputs back into a single vector
        # (B, n_head, T, head_dim) -> (B, T, n_head * head_dim) = (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Output projection
        out = self.c_proj(y)
        return out


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # In SwiGLU, hidden_dim is usually 8/3 of n_embd (or 4 * n_embd for standard GELU)
        hidden_dim = int(4 * config.d_model)

        # SwiGLU uses two parallel projections for gating
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(config.d_model, hidden_dim, bias=False)  # Up projection
        self.c_proj = nn.Linear(hidden_dim, config.d_model, bias=False)  # Down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: SiLU(w1(x)) * w2(x)
        x = F.silu(self.w1(x) * self.w2(x))
        # Project back down to original embedding size
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.d_model)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize -> Attention -> Add back residual
        x = x + self.attn(self.ln_1(x))
        # Normalize -> MLP -> Add back residual
        x = x + self.mlp(self.ln_2(x))

        return x


class FryNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Embedding layers
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.config.vocab_size, self.config.d_model),
                wpe=nn.Embedding(self.config.context, self.config.d_model),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.RMSNorm(config.d_model)
            )
        )

        # Final output projection head
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        # Weight tying (lm_head and wte share memory)
        self.transformer.wte.weight = self.lm_head.weight # type: ignore

        # Initialize weights properly across all layers
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Standard Gaussian initialization for Transformer weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None 
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx     : Tensor of shape (batch_size, seq_len) with token IDs
        targets : Optional Tensor of shape (batch_size, seq_len) with target token IDs
        """
        device = idx.device
        b, t = idx.size()

        assert (
            t <= self.config.context
        ), f"Cannot forward sequence of length {t}, max sequence length is {self.config.context}"

        # Create position IDs: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Token embeddings + positional embeddings
        tok_emb = self.transformer.wte(idx) # type: ignore ; (b, t, n_head)
        pos_emb = self.transformer.wpe(pos) # type: ignore ; (t, d_model)
        x = tok_emb + pos_emb # Broadcasting adds pos_emb across batch dimension

        # PAss through all stacked Transformer blocks
        for block in self.transformer.h: # type: ignore
            x = block(x)

        # Apply final normalization
        x = self.transformer.ln_f(x)  # type: ignore

        # Compute logits for the next token prediction
        if targets is not None:
            # During training: compute full logits and loss using loss.py
            logits = self.lm_head(x)
            loss = calculate_loss(logits, targets)
        else:
            # During inference: only compute logits for the very last token position for speed
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss