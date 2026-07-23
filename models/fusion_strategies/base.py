"""
models/fusion_strategies/base.py — fusion 방식을 교체 가능한 부품으로 만드는 추상 인터페이스.

image_repr/clinical_repr(둘 다 (B, branch_out_dim))을 받아 shared_repr((B, out_dim))을
반환하는 규격만 지키면, 어떤 fusion 방식이든 이 인터페이스를 상속해 `fusion_strategies/`에
추가하고 `__init__.py`의 FUSION_STRATEGIES에 한 줄 등록하는 것만으로 --fusion-strategy
인자에서 바로 선택할 수 있다. fusion_model.py/trainers 어느 쪽도 수정할 필요가 없다.

GBDT처럼 미분 불가능하고 shared trunk 구조 자체가 성립하지 않는 방식은 이 인터페이스로
추상화하지 않는다(Project.md 4.3 참고) — 그런 방식은 gbdt/ 패키지의 완전 별도 파이프라인으로 다룬다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class FusionStrategy(nn.Module, ABC):
    """image_repr(B, branch_out_dim), clinical_repr(B, branch_out_dim) -> shared_repr(B, out_dim)."""

    def __init__(self, branch_out_dim: int, out_dim: int) -> None:
        super().__init__()
        self.branch_out_dim = branch_out_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, image_repr: torch.Tensor, clinical_repr: torch.Tensor) -> torch.Tensor: ...
