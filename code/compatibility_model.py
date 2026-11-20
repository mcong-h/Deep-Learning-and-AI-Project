"""Pairwise neural network for garment compatibility prediction."""

from __future__ import annotations

import torch
from torch import nn


class CompatibilityMLP(nn.Module):
    """Predict whether two garments are unsafe to wash together.

    The pair representation is symmetric:
        [z_a + z_b, |z_a - z_b|, z_a * z_b]

    Swapping garment A and garment B therefore produces the same representation.
    """

    def __init__(
        self,
        garment_feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        pair_dim = garment_feature_dim * 3

        self.network = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    @staticmethod
    def pair_features(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                z_a + z_b,
                torch.abs(z_a - z_b),
                z_a * z_b,
            ],
            dim=-1,
        )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        x = self.pair_features(z_a, z_b)
        return self.network(x).squeeze(-1)
