"""The Gram Critic — a set-transformer that predicts a *velocity field* on Gram geometry.

Given a snapshot of a network's activation geometry (a stack of K×K Gram channels:
the input Gram, hidden Gram, and label Gram), the critic predicts ``ΔK`` — how a single
label-informed validation step *would* reshape the activation Gram. Because it operates
entirely on Gram matrices it is invariant to sample order and feature rotation, so it
transfers across networks. At deploy time it injects that predicted push as a regularizer,
with no validation labels.

Architecture (permutation-equivariant in the K samples):
  per-row DeepSet embed  ->  ISAB (set attention over rows)  ->  symmetric pairwise head -> K×K
"""

from __future__ import annotations

import torch
import torch.nn as nn

D_MODEL = 32
N_HEADS = 2
M_INDUCE = 16


class _MAB(nn.Module):
    """Multihead attention block: ``LN(q + attn) -> LN(h + FF(h))``."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h = self.ln1(q + self.attn(q, kv, kv)[0])
        return self.ln2(h + self.ff(h))


class _ISAB(nn.Module):
    """Induced set-attention block — O(K·m) attention over the K rows via m inducing points."""

    def __init__(self, dim: int, n_heads: int, m_induce: int):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, m_induce, dim) * 0.1)
        self.mab1 = _MAB(dim, n_heads)
        self.mab2 = _MAB(dim, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        induced = self.mab1(self.inducing.expand(x.size(0), -1, -1), x)
        return self.mab2(x, induced)


class GramCritic(nn.Module):
    """Predict the one-step validation ``ΔK`` from a stack of Gram channels.

    Args:
        n_channels: number of Gram input channels (e.g. 4: input, hidden-1, hidden-2, label).
    Call:
        ``critic(grams, k_out)`` where ``grams`` is a list of ``n_channels`` tensors of shape
        ``(B, K, K)`` and ``k_out`` is the current hidden Gram ``(B, K, K)``. Returns the
        predicted symmetric ``(B, K, K)`` velocity ``ΔK``.
    """

    def __init__(self, n_channels: int, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 m_induce: int = M_INDUCE):
        super().__init__()
        self.n_channels = n_channels
        # DeepSet over each row: embed the (channels + self-flag) at every (i, j) entry...
        self.embed = nn.Sequential(nn.Linear(n_channels + 1, d_model), nn.GELU(),
                                   nn.Linear(d_model, d_model))
        # ...then pool columns with mean+var+max (keeps the row's spectrum, not just its mean).
        self.row_proj = nn.Linear(3 * d_model, d_model)
        self.isab = _ISAB(d_model, n_heads, m_induce)
        # symmetric pairwise head: row embeddings (i, j) + current Gram entry -> ΔK[i, j].
        self.pair = nn.Sequential(nn.Linear(2 * d_model + 1, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, grams: list[torch.Tensor], k_out: torch.Tensor) -> torch.Tensor:
        b, k, _ = grams[0].shape
        eye = torch.eye(k, device=grams[0].device, dtype=grams[0].dtype).expand(b, -1, -1)
        emb = self.embed(torch.stack([*grams, eye], dim=-1))           # (B, K, K, d)
        rows = torch.cat([emb.mean(2), emb.var(2), emb.amax(2)], dim=-1)
        e = self.isab(self.row_proj(rows))                             # (B, K, d)
        ei = e.unsqueeze(2).expand(b, k, k, e.size(-1))
        ej = e.unsqueeze(1).expand(b, k, k, e.size(-1))
        pair = torch.cat([ei * ej, ei + ej, k_out.unsqueeze(-1)], dim=-1)
        out = self.pair(pair).squeeze(-1)
        return 0.5 * (out + out.transpose(1, 2))                       # enforce symmetry
