"""
tests/test_gated_fusion_smoke.py — gated 전략 고유 동작 검증 (공통 데이터 인프라는
test_concat_fusion_smoke.py에서 이미 확인함).

확인 항목:
    1. GatedFusion의 branch별 LayerNorm + gate가 실제로 존재하고 gradient가 흐르는지
       (Project.md 4.2 — LayerNorm 없이 스케일이 다른 두 branch를 바로 게이트로 섞으면
       gate=0.5의 의미가 깨진다는 문제를 해결하기 위해 도입된 부분이라 회귀 테스트로 남겨둔다)
    2. concat과 동일한 전체 파이프라인(합성 데이터 + mock encoder)이 gated로도 끝까지 도는지
"""

from __future__ import annotations

import numpy as np
import torch

from config import ImageEncoderConfig
from data.clinical_loader import LABEL_COLS, load_clinical_csv
from data.clinical_transform import preprocess_clinical_for_mlp
from data.embedding_cache import get_or_extract_embeddings
from models.fusion_model import DFUMultimodalModel
from models.fusion_strategies import GatedFusion
from tests.synthetic_data import build_mock_encoder, make_patient_visit_images, make_synthetic_clinical_df
from trainers.common import FusionTrainerArgs, split_by_group
from trainers.losses import MultiTaskLoss
from trainers.neural_trainer import train_neural_model


def test_gated_fusion_gate_receives_gradient():
    model = DFUMultimodalModel(clinical_in_dim=20, image_in_dim=32, fusion_strategy="gated",
                                branch_out_dim=16, trunk_out_dim=8, adapter_out_dim=4)
    assert isinstance(model.fusion, GatedFusion)

    image_emb = torch.randn(6, 32)
    clinical_feat = torch.randn(6, 20)
    out = model(image_emb, clinical_feat)

    batch = {
        "wagner_label": torch.randint(0, 6, (6,)).float(),
        "wagner_mask": torch.ones(6),
        "sinbad_labels": {name: torch.randint(0, 2, (6,)).float() for name in
                          ["site", "ischemia", "neuropathy", "infection", "area", "depth"]},
        "sinbad_masks": {name: torch.ones(6) for name in
                         ["site", "ischemia", "neuropathy", "infection", "area", "depth"]},
    }
    loss_fn = MultiTaskLoss(wagner_num_classes=6)
    losses = loss_fn(out, batch)
    losses["total"].backward()

    assert model.fusion.gate.weight.grad is not None
    assert model.fusion.gate.weight.grad.abs().sum().item() > 0
    assert model.fusion.image_norm.weight.grad is not None
    assert model.fusion.clinical_norm.weight.grad is not None


def test_full_pipeline_gated(tmp_path):
    image_root = tmp_path / "images"
    rng0 = np.random.default_rng(1)
    spec = {f"{i:04d}": {v: 1 for v in range(int(rng0.integers(1, 3)))} for i in range(12)}
    keys_df = make_patient_visit_images(image_root, spec)

    clinical_df = make_synthetic_clinical_df(keys_df, seed=1)
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

    args = FusionTrainerArgs(
        run_name="smoke_gated", checkpoint_dir=str(tmp_path / "checkpoints"),
        fusion_strategy="gated",
        batch_size=8, epochs=2, early_stopping_patience=2, val_ratio=0.25,
        branch_out_dim=32, trunk_out_dim=16, adapter_out_dim=8,
    )
    train_preview, _, _ = split_by_group(merged_df, args.group_col, args.val_ratio, args.test_ratio, args.seed)
    train_mask = merged_df["id"].isin(set(train_preview["id"]))

    id_and_meta = ["id", "visit", "img2d"] + LABEL_COLS + image_emb_cols
    processed_df, feature_cols = preprocess_clinical_for_mlp(merged_df, id_and_meta, train_mask)

    result = train_neural_model(processed_df, feature_cols, image_emb_cols, args, device="cpu")
    assert result["best_val_loss"] is not None
    assert result["model"].fusion_strategy_name == "gated"
