
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()

        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")

        self.attn = nn.MultiheadAttention(
            n_embd,
            n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        _, t, _ = x.shape

        causal_mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        y, _ = self.attn(
            x,
            x,
            x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        return self.dropout(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)

        self.ln2 = nn.LayerNorm(n_embd)

        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        x = x + self.attn(
            self.ln1(x),
            padding_mask=padding_mask,
        )

        return x + self.ff(self.ln2(x))


class GPT0(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        pad_token_id=0,
    ):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.token_emb = nn.Embedding(
            vocab_size,
            n_embd,
        )

        self.pos_emb = nn.Embedding(
            block_size,
            n_embd,
        )

        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)

        self.head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False,
        )

        # Weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, idx, targets=None):
        if idx.ndim != 2:
            raise ValueError("Input must have shape [batch, sequence]")

        b, t = idx.shape

        if t == 0 or t > self.block_size:
            raise ValueError("Invalid sequence length")

        if idx.dtype != torch.long:
            raise TypeError("Input token IDs must use torch.long")

        if torch.any(idx < 0) or torch.any(idx >= self.vocab_size):
            raise ValueError("Input contains token IDs outside vocabulary")

        padding_mask = idx.eq(self.pad_token_id)

        positions = torch.arange(
            t,
            device=idx.device,
        )

        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(
                x,
                padding_mask=padding_mask,
            )

        x = self.ln_f(x)

        logits = self.head(x)

        loss = None

        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("Targets must have the same shape as inputs")

            if targets.dtype != torch.long:
                raise TypeError("Targets must use torch.long")

            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss
