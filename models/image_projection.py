from __future__ import annotations

import torch
from torch import nn


class ImageProjection(nn.Module):
    """frozen DINOv3 embedding(768d: CLS 384 + mean-pool patch 384) -> 256d.

    DINOv3 자체는 이미 강한 표현을 뽑아주므로 1-layer projection만 둔다(ClinicalMLP처럼
    2-layer로 깊게 가지 않음 — 소규모 데이터에서 과적합 위험을 줄이기 위함).
    L2 정규화를 Linear 이전에 적용한다: BatchNorm과 달리 샘플 단위로 독립 계산되기 때문에
    배치가 작을 때도 불안정하지 않다.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [B, in_dim], got shape={tuple(x.shape)}")
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return self.net(x)
