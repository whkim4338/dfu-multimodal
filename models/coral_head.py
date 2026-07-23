from __future__ import annotations

import torch
from torch import nn


class CoralHead(nn.Module):
    """Cao, Mirjalili & Raschka (2020) CORAL — rank-consistent ordinal regression head.

    num_classes개의 순서형 등급(예: Wagner 0~5, num_classes=6)을 num_classes-1개의 binary
    task로 변환한다. penultimate weight는 전부 공유하고 bias만 태스크별로 독립시켜서
    (bias가 학습 후 항상 내림차순이 되도록 강제됨, 원 논문 Theorem 1) rank consistency를
    보장한다.
    """

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes는 2 이상이어야 합니다: {num_classes}")
        self.num_classes = num_classes
        self.shared = nn.Linear(in_dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """반환: (B, num_classes-1) logits. 각 열 k는 'true rank > k'에 대한 logit."""
        shared_logit = self.shared(x)  # (B, 1)
        return shared_logit + self.biases  # broadcast -> (B, num_classes-1)


def coral_label_extension(rank_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """정수 등급(0..num_classes-1)을 CORAL의 확장 binary 라벨로 변환.

    extended[:, k] = 1{rank_labels > k}, k = 0..num_classes-2
    """
    thresholds = torch.arange(num_classes - 1, device=rank_labels.device)
    return (rank_labels.unsqueeze(-1) > thresholds).float()


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """logits(B, num_classes-1) -> 정수 등급 예측(B,). 0.5 threshold 후 개수를 합산."""
    binary_preds = (torch.sigmoid(logits) > 0.5).float()
    return binary_preds.sum(dim=-1)


def coral_loss(
    logits: torch.Tensor,
    rank_labels: torch.Tensor,
    num_classes: int,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """CORAL weighted binary cross-entropy (원 논문 Eq. 4, task weight는 균등 1.0 고정).

    sample_mask: (B,) 이 태스크의 라벨이 있는 행만 1인 mask. 없는 행은 loss에서 제외.
    반환: 스칼라 loss (mask=0인 행이 전부면 0.0을 반환, NaN이 되지 않도록 방지).
    """
    extended = coral_label_extension(rank_labels, num_classes)  # (B, num_classes-1)
    per_task_loss = nn.functional.binary_cross_entropy_with_logits(
        logits, extended, reduction="none"
    )  # (B, num_classes-1)
    per_sample_loss = per_task_loss.sum(dim=-1)  # (B,)

    if sample_mask is None:
        return per_sample_loss.mean()

    denom = sample_mask.sum().clamp_min(1.0)
    return (per_sample_loss * sample_mask).sum() / denom
