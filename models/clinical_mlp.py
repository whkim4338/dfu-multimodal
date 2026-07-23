from __future__ import annotations

import torch
from torch import nn


class ClinicalMLP(nn.Module):
    """Clinical feature(value+missing_flag, 60~62d) -> 256d embedding.

    2-layer MLP. 소규모 데이터 기준으로 안쪽 layer는 dropout을 높게, 바깥쪽은 낮게 뒀다
    (Gorishniy et al., 2021의 관찰을 반영).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 256,
        dropout1: float = 0.3,
        dropout2: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [B, in_dim], got shape={tuple(x.shape)}")
        return self.net(x)
