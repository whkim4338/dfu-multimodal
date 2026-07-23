"""
models/dinov3_backbone.py — frozen DINOv3(HuggingFace DINOv3ViTModel) + CLS+mean-pool concat.

로컬 HF 스냅샷 로딩, freeze/set_trainable 인터페이스를 따르되, DINOv3ViTBackbone(dense
prediction용, CLS 미노출) 대신 전체 토큰 시퀀스를 반환하는 DINOv3ViTModel을 쓴다. 클래스명/
출력 필드명(last_hidden_state, pooler_output, config.num_register_tokens)은 실제 설치된
transformers 라이브러리로 직접 검증했다.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import DINOv3ViTModel

from config import ImageEncoderConfig

_HUB_NAME_MAP = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


class DINOv3ImageEncoder(nn.Module):
    """frozen DINOv3에서 CLS+mean-pooled-patch concat embedding을 뽑는 wrapper."""

    def __init__(self, config: ImageEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.model_path = Path(config.weights_path).resolve() if config.weights_path else None
        self.freeze = True

        if self.model_path is not None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"DINOv3 로컬 스냅샷을 찾을 수 없습니다: {self.model_path}. "
                    "assets/dinov3-hf 디렉토리가 존재하는지, --weights-path 경로가 맞는지 확인하세요."
                )
            self.encoder = DINOv3ViTModel.from_pretrained(str(self.model_path), local_files_only=True)
        else:
            if config.model_name not in _HUB_NAME_MAP:
                raise ValueError(f"알 수 없는 model_name: {config.model_name}")
            self.encoder = DINOv3ViTModel.from_pretrained(_HUB_NAME_MAP[config.model_name])

        self.set_trainable(False)

    def set_trainable(self, trainable: bool) -> None:
        self.freeze = not trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    @property
    def num_register_tokens(self) -> int:
        return getattr(self.encoder.config, "num_register_tokens", 0)

    def _forward_backbone(self, images: torch.Tensor):
        if self.freeze:
            with torch.no_grad():
                return self.encoder(pixel_values=images)
        return self.encoder(pixel_values=images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) -> embedding: (B, config.embedding_dim)"""
        outputs = self._forward_backbone(images)

        if getattr(outputs, "pooler_output", None) is not None:
            cls_token = outputs.pooler_output
        else:
            cls_token = outputs.last_hidden_state[:, 0, :]

        if self.config.pooling == "cls_only":
            return cls_token

        n_register = self.num_register_tokens
        patch_tokens = outputs.last_hidden_state[:, 1 + n_register:, :]
        pooled_patch = patch_tokens.mean(dim=1)

        return torch.cat([cls_token, pooled_patch], dim=-1)
