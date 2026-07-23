from __future__ import annotations

import torch
from torch import nn

from models.fusion_strategies.base import FusionStrategy


class GatedFusion(FusionStrategy):
    """Highway-style modality gating — 두 branch의 concat으로부터 채널별 게이트(0~1)를 학습해
    image_repr/clinical_repr를 재가중한 뒤 합친다. 한쪽 모달리티가 특정 태스크에서 노이즈가
    많을 때 자동으로 덜 반영되게 하는 것이 목적이다.

    image_repr/clinical_repr는 서로 다른 서브네트워크(1-layer projection vs 2-layer MLP)의
    ReLU 출력이라 norm이 보장되지 않는다 — 그래서 게이트 계산 직전에 branch별 LayerNorm(학습 가능한 affine 포함)으로
    두 branch를 정규화한 뒤 게이트를 곱한다. 이 정규화는 이 클래스 내부에만 적용하고
    ImageProjection/ClinicalMLP 자체는 건드리지 않는다 — ConcatFusion이나 향후 다른 전략까지
    불필요하게 제약하지 않기 위함.
    """

    def __init__(self, branch_out_dim: int = 256, out_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__(branch_out_dim, out_dim)
        self.image_norm = nn.LayerNorm(branch_out_dim)
        self.clinical_norm = nn.LayerNorm(branch_out_dim)
        self.gate = nn.Linear(branch_out_dim * 2, branch_out_dim)
        self.net = nn.Sequential(
            nn.Linear(branch_out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, image_repr: torch.Tensor, clinical_repr: torch.Tensor) -> torch.Tensor:
        image_n = self.image_norm(image_repr)
        clinical_n = self.clinical_norm(clinical_repr)
        gate = torch.sigmoid(self.gate(torch.cat([image_n, clinical_n], dim=-1)))
        fused = gate * image_n + (1 - gate) * clinical_n
        return self.net(fused)
