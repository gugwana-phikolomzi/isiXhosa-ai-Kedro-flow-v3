# src/main/pipelines/model_4/gpt.py
from __future__ import annotations

from dataclasses import dataclass
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 256
    dropout: float = 0.0
    bias: bool = True  # use biases like GPT-2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Register a boolean causal mask once (no gradients, non-persistent)
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)
            ).view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, time, channels
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # reshape for multi-head: (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention with causal masking
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, hs)

        # merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

        # weight tying (as in GPT-2)
        self.head.weight = self.tok_emb.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        assert T <= self.cfg.block_size, "Sequence length exceeds block_size"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B, T, C)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss


# --------- Utilities for training (optimizer hygiene, EMA, param count) ---------

def configure_adamw(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta2: float = 0.95,
    eps: float = 1e-8,
    no_decay_patterns: list[str] | None = None,
) -> torch.optim.Optimizer:
    """
    Create AdamW with two parameter groups:
      - 'decay': weight matrices get weight decay
      - 'no_decay': bias, LayerNorm, and embedding weights do not
    """
    if no_decay_patterns is None:
        no_decay_patterns = [
            "bias",
            "LayerNorm.weight",
            "norm.weight",
            "embedding",
            "emb.weight",
        ]

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if any(t in n for t in no_decay_patterns) else decay).append(p)

    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate, betas=(0.9, beta2), eps=eps
    )


class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.999, start_step: int = 1000):
        self.decay = decay
        self.start_step = start_step
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        if step < self.start_step:
            for q, p in zip(self.ema.parameters(), model.parameters()):
                q.data.copy_(p.data)
            return
        d = self.decay
        for q, p in zip(self.ema.parameters(), model.parameters()):
            q.data.mul_(d).add_(p.data, alpha=1 - d)

    def eval_model(self) -> nn.Module:
        return self.ema


def num_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters (optionally only trainable ones)."""
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )
