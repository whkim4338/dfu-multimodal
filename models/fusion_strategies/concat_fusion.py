from __future__ import annotations

import torch
from torch import nn

from models.fusion_strategies.base import FusionStrategy


class ConcatFusion(FusionStrategy):
    """ concat(image_repr, clinical_repr) -> Linear -> ReLU -> Dropout.

    Dropout은 여기(공유부, 입력에 가까움)에만 두고, 태스크별 TaskAdapter는 파라미터가
    적어 dropout을 생략한다 (Gorishniy et al., 2021의 '안쪽은 높게, 바깥쪽은 낮거나 0' 관찰을
    trunk/adapter 구조에 맞게 적용).
    """

    def __init__(self, branch_out_dim: int = 256, out_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__(branch_out_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(branch_out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, image_repr: torch.Tensor, clinical_repr: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([image_repr, clinical_repr], dim=-1))
