from models.fusion_strategies.base import FusionStrategy
from models.fusion_strategies.concat_fusion import ConcatFusion
from models.fusion_strategies.gated_fusion import GatedFusion

# 새 fusion 전략을 추가할 때는: (1) 이 디렉토리에 FusionStrategy를 상속한 클래스를 추가하고,
# (2) 아래 레지스트리에 한 줄 등록하면 된다. fusion_model.py/trainers/cli 어느 것도 고칠 필요 없다.
FUSION_STRATEGIES: dict[str, type[FusionStrategy]] = {
    "concat": ConcatFusion,
    "gated": GatedFusion,
}

__all__ = [
    "FusionStrategy",
    "ConcatFusion",
    "GatedFusion",
    "FUSION_STRATEGIES",
]
