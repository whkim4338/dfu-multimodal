"""
tests/mock_backbone.py — 테스트 전용. 실제 DINOv3 가중치 없이 dinov3_backbone.py를 검증하기
위한 헬퍼. 손으로 짠 가짜 클래스 대신 `DINOv3ViTModel`을 아주 작은 랜덤 설정으로
인스턴스화하는 쪽이 인터페이스 불일치 위험이 없다 (last_hidden_state 레이아웃이
[CLS, register, patch] 순서라는 것도 이 방식으로 직접 검증됨).
"""

from transformers import DINOv3ViTConfig, DINOv3ViTModel


def build_tiny_dinov3(hidden_size=32, num_register_tokens=4, patch_size=16, image_size=64):
    """실제 DINOv3ViTModel 클래스를 무작위 초기화(사전학습 가중치 아님)로 생성.

    목적은 가중치의 정확성 검증이 아니라, dinov3_backbone.py가 실제 클래스의 forward()
    출력 인터페이스(last_hidden_state/pooler_output 필드명, 토큰 순서)와 어긋나지 않는지
    확인하는 것이다.
    """
    config = DINOv3ViTConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=2,
        num_attention_heads=2,
        patch_size=patch_size,
        image_size=image_size,
        num_register_tokens=num_register_tokens,
    )
    model = DINOv3ViTModel(config)
    model.eval()
    return model
