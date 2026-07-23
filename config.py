"""
config.py — 이미지 인코더 설정. `dfu_fusion/config.py`와 동일한 로직을 이 패키지 안으로
그대로 이식했다 (완전 독립 패키지 컨벤션 — 다른 dfu_* 패키지를 import하지 않는다).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImageEncoderConfig:
    """DINOv3 이미지 인코더 설정.

    input_resolution: patch_size(16)의 배수면 임의의 값으로 바꿀 수 있다. DINOv3는 RoPE 기반
        위치 인코딩이라 학습 시 기본 해상도(224)와 달라도 비교적 안정적으로 동작하지만, crop
        해상도가 아직 실험적으로 확정되지 않아 기본값을 두지 않고 CLI에서 매번 명시하도록
        강제한다(cli/fusion_args.py 참고).
    """

    model_name: str = "dinov3_vits16"
    input_resolution: int = 224
    patch_size: int = 16
    weights_path: str | None = None       # 로컬 HF 스냅샷 디렉토리 (assets/dinov3-hf)
    pooling: str = "mean"                  # "mean" | "cls_only"
    device: str = "cpu"
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.input_resolution % self.patch_size != 0:
            raise ValueError(
                f"input_resolution({self.input_resolution})은 patch_size({self.patch_size})의 "
                f"배수여야 합니다."
            )
        if self.pooling not in {"mean", "cls_only"}:
            raise ValueError(f"지원하지 않는 pooling 방식: {self.pooling}")

    @property
    def embedding_dim(self) -> int:
        dims = {"dinov3_vits16": 384, "dinov3_vitb16": 768, "dinov3_vitl16": 1024}
        if self.model_name not in dims:
            raise ValueError(f"알 수 없는 model_name: {self.model_name}. 지원: {list(dims)}")
        base_dim = dims[self.model_name]
        return base_dim * 2 if self.pooling == "mean" else base_dim
