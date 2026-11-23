"""MLP for property price regression (predicts log price).

    categorical ids ──> Embedding per column ──┐
                                               ├─ concat ─> [Linear-BN-Act-Dropout] x N ─> Linear(1)
    numeric + flags ──> Linear-BN-Act ─────────┘
"""
from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(0.01)}


class PriceMLP(nn.Module):
    def __init__(self, cardinalities, embedding_dims, n_numeric, numeric_hidden=32,
                 hidden=(256, 128, 64), dropout=0.25, embedding_dropout=0.1,
                 activation="leaky_relu", target_mean=0.0):
        super().__init__()
        act = ACTIVATIONS[activation]

        self.embeddings = nn.ModuleList(nn.Embedding(n, d) for n, d in zip(cardinalities, embedding_dims))
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.numeric = nn.Sequential(nn.Linear(n_numeric, numeric_hidden), nn.BatchNorm1d(numeric_hidden), act())

        layers, width = [], sum(embedding_dims) + numeric_hidden
        for h in hidden:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), act(), nn.Dropout(dropout)]
            width = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        # start predictions at the mean log price so early epochs don't chase the offset
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, target_mean)

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        emb = [e(cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat([self.embedding_dropout(torch.cat(emb, dim=1)), self.numeric(num)], dim=1)
        return self.head(self.body(x)).squeeze(1)
