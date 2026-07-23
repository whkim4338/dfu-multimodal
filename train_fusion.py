"""
train_fusion.py — dfu_multimodal 학습 진입점. --fusion-strategy로 fusion 방식을 교체한다.

입력은 딱 두 가지뿐이다:
    --clinical-csv   model_input_variables.csv (id, visit, img2d, raw clinical feature, 라벨 컬럼)
    --image-root     {id}/{visit}/{img2d} 구조로 이미지가 있는 루트 디렉토리
                     (환자 한 명이 여러 방문·여러 사진을 가질 수 있음을 전제)

흐름:
    1. clinical CSV 로딩
    2. 환자(id) 단위 split (같은 환자 이미지가 train/val에 안 섞이도록)
    3. DINOv3 embedding 확보 (캐시에 없는 것만 새로 인코딩)
    4. clinical 병합 + train split으로만 전처리 fit
    5. --fusion-strategy 값에 따라 neural 경로(concat/gated) 또는 GBDT 경로(Phase 5)로 분기

사용법 (dfu_multimodal 디렉토리 안에서 실행):
    python -m train_fusion \
        --resolution 224 \
        --fusion-strategy concat \
        --run-name concat_v1 --epochs 30 --device cuda

--clinical-csv/--image-root 기본값은 이 프로젝트 안의 dataset/ 폴더를 가리키므로 toy 데이터로
실험할 때는 생략 가능하다. 다른 CSV/이미지를 쓰려면 명시적으로 덮어쓰면 된다.
"""

from __future__ import annotations

import argparse

import pandas as pd

from cli.fusion_args import add_fusion_args, args_to_trainer_args
from config import ImageEncoderConfig
from data.clinical_loader import LABEL_COLS, load_clinical_csv
from data.clinical_transform import preprocess_clinical_for_mlp
from data.embedding_cache import get_or_extract_embeddings
from gbdt.gbdt_trainer import GBDTConfig, run_gbdt_pipeline
from gbdt.tabular_builder import build_tabular_frame
from gbdt.tasks import TASK_DEFINITIONS
from models.fusion_strategies import FUSION_STRATEGIES
from trainers.common import split_by_group
from trainers.neural_trainer import train_neural_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    return add_fusion_args(parser)


_MISSING_IMG_TOKENS = {"none", "null", "nan", ""}


def load_and_merge_embeddings(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    """clinical CSV 로딩 + 이미지 확보/캐싱 + 병합. neural/GBDT 두 경로가 공유하는 부분만
    담당한다 — clinical feature를 어떻게 다듬을지(MLP 표준화 vs GBDT raw)는 각자 경로에서
    이어서 처리한다."""
    df = load_clinical_csv(args.clinical_csv)
    df = df.dropna(subset=["img2d", "visit"]).reset_index(drop=True)  # img2d/visit 없는 행은 이미지 매칭 불가하므로 제외
    # img2d가 NaN이 아니라 문자열 "none"/"null"/"nan"으로 채워진 행도 있다(원본 eCRF 전처리
    # 관례) — 실제 파일이 없는데 문자열 값이 있어 dropna로는 안 걸러지므로 여기서 추가로 제외.
    # CSV의 img2d는 "id/visit/파일명" 상대경로라 마지막 조각(파일명)만 떼어 비교해야 한다.
    img2d_basename = df["img2d"].astype(str).str.strip().str.lower().str.rsplit("/", n=1).str[-1]
    is_missing_token = img2d_basename.isin(_MISSING_IMG_TOKENS)
    if is_missing_token.any():
        print(f"[load_and_merge_embeddings] img2d가 'none' 등 결측 표기인 행 {int(is_missing_token.sum())}개 제외")
    df = df[~is_missing_token].reset_index(drop=True)

    image_config = ImageEncoderConfig(
        model_name=args.model_name,
        input_resolution=args.resolution,
        weights_path=args.weights_path,
        pooling=args.pooling,
        device=args.device,
    )

    embeddings_df = get_or_extract_embeddings(
        df, args.image_root, image_config,
        cache_path=args.embedding_cache,
        batch_size=args.embed_batch_size,
        num_workers=args.embed_num_workers,
    )
    image_emb_cols = [c for c in embeddings_df.columns if c not in ("id", "visit", "img2d")]

    merged_df = df.merge(embeddings_df, on=["id", "visit", "img2d"], how="inner")
    n_dropped = len(df) - len(merged_df)
    if n_dropped > 0:
        print(f"[load_and_merge_embeddings] 이미지 확보 실패로 {n_dropped}행 제외됨")

    return merged_df, image_emb_cols


def prepare_data_neural(args: argparse.Namespace, trainer_args) -> tuple[pd.DataFrame, list[str], list[str]]:
    merged_df, image_emb_cols = load_and_merge_embeddings(args)

    # clinical fit은 반드시 train subset에서만 — 모든 fusion 전략이 공유하는 split_by_group으로
    # train_mask를 먼저 확보한다(neural_trainer 내부에서 같은 seed로 동일한 split을 재현하므로
    # 이중 계산이 아니다).
    train_df_preview, _, _ = split_by_group(
        merged_df, trainer_args.group_col, trainer_args.val_ratio, trainer_args.test_ratio, trainer_args.seed,
    )
    train_ids = set(train_df_preview[trainer_args.group_col])
    train_mask = merged_df[trainer_args.group_col].isin(train_ids)

    id_and_meta_cols = ["id", "visit", "img2d"] + LABEL_COLS + image_emb_cols
    processed_df, clinical_feature_cols = preprocess_clinical_for_mlp(merged_df, id_and_meta_cols, train_mask)

    print(f"[prepare_data_neural] clinical feature {len(clinical_feature_cols)}개, image embedding {len(image_emb_cols)}차원")
    print(f"[prepare_data_neural] 최종 학습 표본: {len(processed_df)}행, 환자 수: {processed_df['id'].nunique()}명")

    return processed_df, clinical_feature_cols, image_emb_cols


def prepare_data_gbdt(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    """GBDT 경로는 clinical_transform(표준화/median 대체)을 거치지 않고 raw 값을 그대로 쓴다
    (Project.md 4.3 — CatBoost/XGBoost/LightGBM은 결측치를 자체 처리)."""
    merged_df, image_emb_cols = load_and_merge_embeddings(args)
    tabular_df, feature_cols = build_tabular_frame(merged_df, image_emb_cols)

    print(f"[prepare_data_gbdt] feature {len(feature_cols)}개(clinical raw + image embedding)")
    print(f"[prepare_data_gbdt] 최종 학습 표본: {len(tabular_df)}행, 환자 수: {tabular_df['id'].nunique()}명")

    return tabular_df, feature_cols


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    trainer_args = args_to_trainer_args(args)

    if trainer_args.fusion_strategy in FUSION_STRATEGIES:
        df, clinical_feature_cols, image_emb_cols = prepare_data_neural(args, trainer_args)
        result = train_neural_model(df, clinical_feature_cols, image_emb_cols, trainer_args, device=args.device)
        print(f"학습 완료. best_val_loss={result['best_val_loss']:.4f}")
        print(f"체크포인트 저장 위치: {trainer_args.run_dir}/best.pt, {trainer_args.run_dir}/last.pt")
    elif trainer_args.fusion_strategy == "gbdt":
        df, feature_cols = prepare_data_gbdt(args)
        gbdt_config = GBDTConfig(
            backend=args.gbdt_backend,
            iterations=args.gbdt_iterations,
            depth=args.gbdt_depth,
            learning_rate=args.gbdt_learning_rate,
            random_state=trainer_args.seed,
            min_samples=args.gbdt_min_samples,
            cat_features=args.gbdt_cat_features,
            min_child_samples=args.gbdt_min_child_samples,
        )
        run_gbdt_pipeline(df, feature_cols, TASK_DEFINITIONS, trainer_args, gbdt_config)
        print(f"학습 완료. 결과 저장 위치: {trainer_args.run_dir}/summary.json")
    else:
        raise ValueError(f"알 수 없는 fusion_strategy: {trainer_args.fusion_strategy!r}")


if __name__ == "__main__":
    main()
