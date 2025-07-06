"""
AnfisNet – simple 5‑class classifier (raw logits, no Softmax).
"""

import torch
import torch.nn as nn


class AnfisNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden1: int = 64,
        hidden2: int = 32,
        output_dim: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),  # raw logits
        )

    def forward(self, x):
        return self.net(x)
