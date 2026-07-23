from __future__ import annotations

import torch
from torch import nn

from models.coral_head import coral_loss
from models.fusion_model import SINBAD_COMPONENTS


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """라벨이 없는(mask=0) 행을 제외한 BCE. 배치 전체가 mask=0이면 0.0 반환(NaN 방지)."""
    per_sample = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (per_sample * mask).sum() / denom


class MultiTaskLoss(nn.Module):
    """Wagner(CORAL) + SINBAD 6개(masked BCE)를 합산.

    태스크 가중치(lambda)는 전부 1.0으로 균등 고정 — 데이터가 늘어나 태스크 간 loss scale
    불균형이 관찰되면 uncertainty-based weighting(Kendall et al., 2018) 도입을 검토할 것.
    """

    def __init__(self, wagner_num_classes: int = 6) -> None:
        super().__init__()
        self.wagner_num_classes = wagner_num_classes

    def forward(self, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
        wagner_loss = coral_loss(
            outputs["wagner"], batch["wagner_label"], self.wagner_num_classes, batch["wagner_mask"]
        )

        sinbad_losses = {
            name: masked_bce_loss(
                outputs["sinbad"][name], batch["sinbad_labels"][name], batch["sinbad_masks"][name]
            )
            for name in SINBAD_COMPONENTS
        }
        sinbad_loss = torch.stack(list(sinbad_losses.values())).sum()

        total = wagner_loss + sinbad_loss

        return {
            "total": total,
            "wagner": wagner_loss,
            "sinbad": sinbad_loss,
            **{f"sinbad_{name}": loss for name, loss in sinbad_losses.items()},
        }
