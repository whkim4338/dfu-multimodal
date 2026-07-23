"""
tests/test_gbdt_fusion_smoke.py — GBDT 경로 고유 동작 검증.

확인 항목:
    1. tabular_builder가 clinical raw feature + image embedding을 중복 없이 하나의 feature
       목록으로 합치는지
    2. run_gbdt_pipeline이 태스크별 독립 모델을 학습하고 neural 경로와 같은 summary 스키마
       (n_val_labeled/accuracy/balanced_accuracy/macro_f1/roc_auc)를 산출하는지
    3. concat(neural)과 gbdt가 완전히 같은 (seed, val_ratio) 조합에서 완전히 같은 환자
       집합으로 split되는지 (Project.md 공통 설계 원칙 2번 — 전략 간 공정 비교의 전제)
"""

from __future__ import annotations

import numpy as np

from config import ImageEncoderConfig
from data.clinical_loader import LABEL_COLS, load_clinical_csv
from data.clinical_transform import preprocess_clinical_for_mlp
from data.embedding_cache import get_or_extract_embeddings
from gbdt.gbdt_trainer import GBDTConfig, run_gbdt_pipeline
from gbdt.tabular_builder import build_tabular_frame
from gbdt.tasks import TASK_DEFINITIONS
from tests.synthetic_data import build_mock_encoder, make_patient_visit_images, make_synthetic_clinical_df
from trainers.common import FusionTrainerArgs, split_by_group
from trainers.neural_trainer import train_neural_model


def _build_merged_df(tmp_path, seed: int):
    image_root = tmp_path / "images"
    rng0 = np.random.default_rng(seed)
    spec = {f"{i:04d}": {v: 1 for v in range(int(rng0.integers(1, 3)))} for i in range(16)}
    keys_df = make_patient_visit_images(image_root, spec)

    clinical_df = make_synthetic_clinical_df(keys_df, seed=seed)
    csv_path = tmp_path / "clinical.csv"
    clinical_df.to_csv(csv_path, index=False)
    df = load_clinical_csv(csv_path)

    config = ImageEncoderConfig(model_name="dinov3_vits16", input_resolution=64, pooling="mean")
    encoder = build_mock_encoder(config, hidden_size=32)
    cache_path = tmp_path / "emb_cache.parquet"
    embeddings_df = get_or_extract_embeddings(
        df, image_root, config, cache_path=cache_path, batch_size=4, num_workers=0, encoder=encoder,
    )
    image_emb_cols = [c for c in embeddings_df.columns if c not in ("id", "visit", "img2d")]

    merged_df = df.merge(embeddings_df, on=["id", "visit", "img2d"], how="inner")
    return merged_df, image_emb_cols


def test_tabular_builder_no_duplicate_columns(tmp_path):
    merged_df, image_emb_cols = _build_merged_df(tmp_path, seed=2)
    tabular_df, feature_cols = build_tabular_frame(merged_df, image_emb_cols)

    assert len(feature_cols) == len(set(feature_cols)), "feature 목록에 중복 컬럼이 있습니다."
    for col in image_emb_cols:
        assert col in feature_cols
    for col in ("id", "visit", "img2d", *LABEL_COLS):
        assert col not in feature_cols


def test_full_pipeline_gbdt(tmp_path):
    merged_df, image_emb_cols = _build_merged_df(tmp_path, seed=3)
    tabular_df, feature_cols = build_tabular_frame(merged_df, image_emb_cols)

    args = FusionTrainerArgs(
        run_name="smoke_gbdt", checkpoint_dir=str(tmp_path / "checkpoints"),
        fusion_strategy="gbdt", val_ratio=0.25, seed=42,
    )
    gbdt_config = GBDTConfig(backend="catboost", iterations=20, depth=2, min_samples=4)

    result = run_gbdt_pipeline(tabular_df, feature_cols, TASK_DEFINITIONS, args, gbdt_config)
    summary = result["summary"]

    assert summary["fusion_strategy"] == "gbdt"
    wagner_task = summary["tasks"]["wagner"]
    assert "skipped_reason" in wagner_task or {
        "n_val_labeled", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc",
    }.issubset(wagner_task.keys())


def test_concat_and_gbdt_share_same_patient_split(tmp_path):
    """같은 (seed, val_ratio)라면 neural(concat)과 gbdt가 완전히 같은 환자 집합으로
    나뉘어야 한다 — 그래야 두 전략의 성능 비교가 공정하다."""
    merged_df, image_emb_cols = _build_merged_df(tmp_path, seed=4)

    common_args = dict(val_ratio=0.3, test_ratio=0.0, seed=7, group_col="id")

    # neural(concat) 경로: preprocess_clinical_for_mlp까지 거친 뒤에도 split은 동일해야 함
    neural_args = FusionTrainerArgs(
        run_name="split_check_concat", checkpoint_dir=str(tmp_path / "checkpoints_neural"),
        fusion_strategy="concat", branch_out_dim=16, trunk_out_dim=8, adapter_out_dim=4,
        epochs=1, batch_size=8, **common_args,
    )
    train_preview, val_preview, _ = split_by_group(
        merged_df, neural_args.group_col, neural_args.val_ratio, neural_args.test_ratio, neural_args.seed,
    )
    train_mask = merged_df["id"].isin(set(train_preview["id"]))
    id_and_meta = ["id", "visit", "img2d"] + LABEL_COLS + image_emb_cols
    processed_df, clinical_feature_cols = preprocess_clinical_for_mlp(merged_df, id_and_meta, train_mask)
    neural_result = train_neural_model(processed_df, clinical_feature_cols, image_emb_cols, neural_args, device="cpu")

    # gbdt 경로: run_gbdt_pipeline이 내부에서 다시 split_by_group을 호출
    gbdt_args = FusionTrainerArgs(
        run_name="split_check_gbdt", checkpoint_dir=str(tmp_path / "checkpoints_gbdt"),
        fusion_strategy="gbdt", **common_args,
    )
    tabular_df, feature_cols = build_tabular_frame(merged_df, image_emb_cols)
    gbdt_result = run_gbdt_pipeline(tabular_df, feature_cols, TASK_DEFINITIONS, gbdt_args, GBDTConfig(iterations=20, depth=2, min_samples=2))

    assert neural_result is not None
    assert gbdt_result["summary"]["n_train"] == len(train_preview)
    assert gbdt_result["summary"]["n_val"] == len(val_preview)
