from __future__ import annotations

import torch
from torch import nn

from models.clinical_mlp import ClinicalMLP
from models.coral_head import CoralHead
from models.fusion_strategies import FUSION_STRATEGIES
from models.image_projection import ImageProjection

SINBAD_COMPONENTS = ["site", "ischemia", "neuropathy", "infection", "area", "depth"]


class TaskAdapter(nn.Module):
    """공유 표현(trunk_out_dim) -> 태스크별 표현(adapter_out_dim). 파라미터가 적어 dropout 없음."""

    def __init__(self, in_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DFUMultimodalModel(nn.Module):
    """image_emb(768d) + clinical_feat(60~62d) -> Wagner/SINBAD×6.

    fusion 방식(concat/gated/...)은 `fusion_strategy` 인자로 교체 가능 —
    models/fusion_strategies/FUSION_STRATEGIES 레지스트리에서 이름으로 조회해 조립한다.
    이미지/임상 branch, TaskAdapter, head 구조는 fusion 방식과 무관하게 재사용된다.
    """

    def __init__(
        self,
        clinical_in_dim: int,
        image_in_dim: int = 768,
        fusion_strategy: str = "concat",
        branch_out_dim: int = 256,
        trunk_out_dim: int = 128,
        adapter_out_dim: int = 64,
        wagner_num_classes: int = 6,
    ) -> None:
        super().__init__()
        if fusion_strategy not in FUSION_STRATEGIES:
            raise ValueError(
                f"알 수 없는 fusion_strategy: {fusion_strategy!r}. 지원: {list(FUSION_STRATEGIES)}"
            )

        self.fusion_strategy_name = fusion_strategy
        self.image_projection = ImageProjection(in_dim=image_in_dim, out_dim=branch_out_dim)
        self.clinical_mlp = ClinicalMLP(in_dim=clinical_in_dim, out_dim=branch_out_dim)
        self.fusion = FUSION_STRATEGIES[fusion_strategy](branch_out_dim=branch_out_dim, out_dim=trunk_out_dim)

        self.wagner_adapter = TaskAdapter(trunk_out_dim, adapter_out_dim)
        self.wagner_head = CoralHead(adapter_out_dim, num_classes=wagner_num_classes)

        self.sinbad_adapter = TaskAdapter(trunk_out_dim, adapter_out_dim)
        self.sinbad_heads = nn.ModuleDict(
            {name: nn.Linear(adapter_out_dim, 1) for name in SINBAD_COMPONENTS}
        )

    def forward(self, image_emb: torch.Tensor, clinical_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        image_repr = self.image_projection(image_emb)
        clinical_repr = self.clinical_mlp(clinical_feat)
        shared_repr = self.fusion(image_repr, clinical_repr)

        wagner_logits = self.wagner_head(self.wagner_adapter(shared_repr))

        sinbad_hidden = self.sinbad_adapter(shared_repr)
        sinbad_logits = {name: head(sinbad_hidden).squeeze(-1) for name, head in self.sinbad_heads.items()}

        return {
            "wagner": wagner_logits,             # (B, num_classes-1)
            "sinbad": sinbad_logits,              # dict[str, (B,)]
        }

    def sinbad_total_score(self, sinbad_logits: dict[str, torch.Tensor]) -> torch.Tensor:
        """SINBAD 6개 component 예측(각각 0.5 threshold)을 합산해 총점(0~6) 산출.

        SINBAD 총점 자체에는 별도 ordinal head를 두지 않고, 6개 독립 binary head의 합으로
        유도한다.
        """
        preds = [
            (torch.sigmoid(sinbad_logits[name]) > 0.5).float() for name in SINBAD_COMPONENTS
        ]
        return torch.stack(preds, dim=-1).sum(dim=-1)
