"""
tests/test_concat_fusion_smoke.py — concat 전략 + 공통 데이터 파이프라인(이미지 로딩,
embedding 캐시, 환자 단위 split) 검증. 실제 DINOv3 가중치 없이 mock encoder로 돈다.

concat/gated는 image/clinical branch, split, 전처리 로직을 전부 공유하므로, 이 파일에서
공통 인프라(이미지 폴더 구조, embedding 캐시 재사용, group split)까지 함께 확인하고,
test_gated_fusion_smoke.py/test_gbdt_fusion_smoke.py는 각 전략 고유 동작만 확인한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ImageEncoderConfig
from data.clinical_loader import LABEL_COLS, load_clinical_csv
from data.clinical_transform import preprocess_clinical_for_mlp
from data.embedding_cache import get_or_extract_embeddings
from data.image_dataset import WoundImageDataset, build_image_path
from tests.synthetic_data import build_mock_encoder, make_patient_visit_images, make_synthetic_clinical_df
from trainers.common import FusionTrainerArgs, split_by_group
from trainers.neural_trainer import train_neural_model


def test_visit_folder_structure(tmp_path):
    image_root = tmp_path / "images"
    keys_df = make_patient_visit_images(image_root, {"0001": {0: 2, 1: 1}, "0002": {0: 1}})

    assert build_image_path(image_root, "0001", 0, "img_0.jpg").exists()
    assert build_image_path(image_root, "0001", 1, "img_0.jpg").exists()
    assert build_image_path(image_root, "0002", 0, "img_0.jpg").exists()

    config = ImageEncoderConfig(model_name="dinov3_vits16", input_resolution=64, pooling="mean")
    dataset = WoundImageDataset(keys_df, image_root, config.input_resolution, config.normalize_mean, config.normalize_std)
    assert len(dataset) == 4  # 0001: visit0 2장 + visit1 1장, 0002: visit0 1장

    tensor, patient_id, visit, img2d = dataset[0]
    assert tensor.shape == (3, 64, 64)


def test_embedding_cache_reuse(tmp_path):
    image_root = tmp_path / "images"
    keys_df = make_patient_visit_images(image_root, {"0001": {0: 1, 1: 1}, "0002": {0: 1}})
    cache_path = tmp_path / "emb_cache.parquet"

    config = ImageEncoderConfig(model_name="dinov3_vits16", input_resolution=64, pooling="mean")
    encoder = build_mock_encoder(config, hidden_size=32)

    call_count = {"n": 0}
    original_forward = encoder.forward

    def counting_forward(images):
        call_count["n"] += 1
        return original_forward(images)

    encoder.forward = counting_forward

    emb1 = get_or_extract_embeddings(
        keys_df.copy(), image_root, config, cache_path=cache_path, batch_size=2, num_workers=0, encoder=encoder,
    )
    assert len(emb1) == 3  # 0001-v0, 0001-v1, 0002-v0
    first_call_count = call_count["n"]
    assert first_call_count > 0

    emb2 = get_or_extract_embeddings(
        keys_df.copy(), image_root, config, cache_path=cache_path, batch_size=2, num_workers=0, encoder=encoder,
    )
    assert len(emb2) == 3
    assert call_count["n"] == first_call_count, "캐시가 있는데도 재인코딩이 일어났습니다."


def test_group_split_keeps_all_visits_together():
    """한 환자의 서로 다른 visit이 train/val에 걸쳐 섞이지 않는지 확인."""
    rng = np.random.default_rng(0)
    rows = []
    for p in range(10):
        n_visits = rng.integers(1, 4)
        for v in range(int(n_visits)):
            rows.append({"id": f"P{p}", "visit": v, "img2d": f"img_{v}.jpg"})
    df = pd.DataFrame(rows)

    train_df, val_df, _ = split_by_group(df, group_col="id", val_ratio=0.3, test_ratio=0.0, seed=42)
    train_ids, val_ids = set(train_df["id"]), set(val_df["id"])
    assert not (train_ids & val_ids), f"같은 환자가 train/val에 걸쳐 나타났습니다: {train_ids & val_ids}"


def test_full_pipeline_concat(tmp_path):
    image_root = tmp_path / "images"
    rng0 = np.random.default_rng(0)
    spec = {f"{i:04d}": {v: 1 for v in range(int(rng0.integers(1, 3)))} for i in range(12)}
    keys_df = make_patient_visit_images(image_root, spec)

    clinical_df = make_synthetic_clinical_df(keys_df, seed=0)
    csv_path = tmp_path / "clinical.csv"
    clinical_df.to_csv(csv_path, index=False)

    df = load_clinical_csv(csv_path)
    assert len(df) == len(keys_df)

    config = ImageEncoderConfig(model_name="dinov3_vits16", input_resolution=64, pooling="mean")
    encoder = build_mock_encoder(config, hidden_size=32)
    cache_path = tmp_path / "emb_cache.parquet"
    embeddings_df = get_or_extract_embeddings(
        df, image_root, config, cache_path=cache_path, batch_size=4, num_workers=0, encoder=encoder,
    )
    image_emb_cols = [c for c in embeddings_df.columns if c not in ("id", "visit", "img2d")]

    merged_df = df.merge(embeddings_df, on=["id", "visit", "img2d"], how="inner")
    assert len(merged_df) == len(df)

    args = FusionTrainerArgs(
        run_name="smoke_concat", checkpoint_dir=str(tmp_path / "checkpoints"),
        fusion_strategy="concat",
        batch_size=8, epochs=2, early_stopping_patience=2, val_ratio=0.25,
        branch_out_dim=32, trunk_out_dim=16, adapter_out_dim=8,
    )
    train_preview, _, _ = split_by_group(merged_df, args.group_col, args.val_ratio, args.test_ratio, args.seed)
    train_mask = merged_df["id"].isin(set(train_preview["id"]))

    id_and_meta = ["id", "visit", "img2d"] + LABEL_COLS + image_emb_cols
    processed_df, feature_cols = preprocess_clinical_for_mlp(merged_df, id_and_meta, train_mask)

    result = train_neural_model(processed_df, feature_cols, image_emb_cols, args, device="cpu")
    assert result["best_val_loss"] is not None
    assert result["model"].fusion_strategy_name == "concat"
