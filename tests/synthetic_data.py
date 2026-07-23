"""
tests/synthetic_data.py — smoke test 3종이 공유하는 합성 데이터/mock encoder 헬퍼.
실제 DINOv3 가중치나 실 환자 데이터 없이 파이프라인 전체를 검증하기 위함.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import ImageEncoderConfig
from models.dinov3_backbone import DINOv3ImageEncoder
from tests.mock_backbone import build_tiny_dinov3


def build_mock_encoder(config: ImageEncoderConfig, hidden_size: int = 32) -> DINOv3ImageEncoder:
    """가중치 로딩 없이 DINOv3ImageEncoder와 동일한 인터페이스를 갖는 인스턴스를 만든다."""
    encoder = DINOv3ImageEncoder.__new__(DINOv3ImageEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.config = config
    encoder.model_path = None
    encoder.freeze = True
    encoder.encoder = build_tiny_dinov3(
        hidden_size=hidden_size, num_register_tokens=4,
        patch_size=config.patch_size, image_size=config.input_resolution,
    )
    encoder.set_trainable(False)
    return encoder


def make_patient_visit_images(image_root: Path, spec: dict) -> pd.DataFrame:
    """spec 예: {"0001": {0: 2, 1: 1}, "0002": {0: 1}}
    -> 환자 0001은 visit0에 2장·visit1에 1장, 환자 0002는 visit0에 1장.
    실제 폴더 구조(image_root/{id}/{visit}/{파일명})까지 함께 만든다."""
    rng = np.random.default_rng(0)
    rows = []
    for patient_id, visits in spec.items():
        for visit_num, n_images in visits.items():
            visit_dir = image_root / patient_id / str(visit_num)
            visit_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_images):
                fname = f"img_{i}.jpg"
                arr = (rng.random((80, 96, 3)) * 255).astype(np.uint8)
                Image.fromarray(arr).save(visit_dir / fname)
                rows.append({"id": patient_id, "visit": visit_num, "img2d": fname})
    return pd.DataFrame(rows)


def make_synthetic_clinical_df(keys_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """keys_df(id/visit/img2d)에 raw clinical feature + 라벨 컬럼을 덧붙인다 (NaN 일부 포함,
    태스크별 라벨 결측 처리 경로도 같이 exercise됨)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in keys_df.iterrows():
        rows.append({
            "id": row["id"], "visit": row["visit"], "img2d": row["img2d"],
            "sex": rng.integers(0, 2),
            "age_band": rng.integers(1, 16),
            "bmi": rng.normal(24, 3),
            "hba1c": rng.normal(7, 1.5),
            "amputation_history": rng.integers(0, 2) if rng.random() > 0.3 else np.nan,
            "wag": int(rng.integers(0, 6)) if rng.random() > 0.15 else np.nan,
            "snb_site": rng.integers(0, 2),
            "snb_isc": rng.integers(0, 2),
            "snb_neuro": rng.integers(0, 2),
            "snb_bac": rng.integers(0, 2),
            "snb_area": rng.integers(0, 2),
            "snb_depth": rng.integers(0, 2),
        })
    return pd.DataFrame(rows)
