from __future__ import annotations

import argparse

from models.fusion_strategies import FUSION_STRATEGIES
from trainers.common import FusionTrainerArgs

# neural(FUSION_STRATEGIES) + gbdt(완전 별도 파이프라인, Phase 5) 전부 --fusion-strategy 하나로 선택.
_ALL_FUSION_STRATEGIES = [*FUSION_STRATEGIES, "gbdt"]


def add_fusion_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # --- 입력 (기본값은 이 프로젝트 안의 dataset/ 폴더 기준) ---
    parser.add_argument("--clinical-csv", default="dataset/model_input_variables_full.csv",
                         help="model_input_variables.csv (id/visit/img2d/라벨 컬럼 포함, raw 상태)")
    parser.add_argument("--image-root", default="dataset/toy_image_dataset",
                         help="이미지 루트 디렉토리. 실제 파일은 {image-root}/{id}/{visit}/{img2d}에 있어야 함")
    parser.add_argument("--embedding-cache", default=None,
                         help="DINOv3 embedding 캐시 parquet 경로 (선택). 지정하면 재실행 시 재추출 안 함")

    # --- DINOv3 ---
    parser.add_argument("--resolution", type=int, required=True,
                         help="DINOv3 입력 해상도. patch_size(16)의 배수여야 함")
    parser.add_argument("--model-name", default="dinov3_vits16",
                         choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"])
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls_only"])
    parser.add_argument("--weights-path", default="assets/dinov3-hf",
                         help="로컬 HF 스냅샷 경로 (기본값: 이 프로젝트 안의 assets/dinov3-hf)")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--embed-num-workers", type=int, default=2)

    # --- Fusion 전략 ---
    parser.add_argument("--fusion-strategy", default="concat", choices=_ALL_FUSION_STRATEGIES,
                         help="concat/gated는 신경망 경로(trainers/neural_trainer.py), "
                              "gbdt는 완전 별도 파이프라인(gbdt/gbdt_trainer.py)")

    # --- GBDT 전용 (--fusion-strategy gbdt일 때만 사용) ---
    parser.add_argument("--gbdt-backend", default="catboost", choices=["catboost", "xgboost", "lightgbm"])
    parser.add_argument("--gbdt-iterations", type=int, default=300)
    parser.add_argument("--gbdt-depth", type=int, default=4)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbdt-min-samples", type=int, default=10)
    parser.add_argument("--gbdt-cat-features", nargs="*", default=["age_band"],
                         help="categorical로 취급할 clinical feature 컬럼 목록")
    parser.add_argument("--gbdt-min-child-samples", type=int, default=5,
                         help="LightGBM 전용(min_data_in_leaf). 소규모 데이터 대응으로 기본값을 낮춰둠")

    # --- 학습 ---
    parser.add_argument("--run-name", default=None, help="미지정 시 timestamp 자동 생성")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    return parser


def args_to_trainer_args(args: argparse.Namespace) -> FusionTrainerArgs:
    kwargs = dict(
        checkpoint_dir=args.checkpoint_dir,
        fusion_strategy=args.fusion_strategy,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if args.run_name is not None:
        kwargs["run_name"] = args.run_name
    return FusionTrainerArgs(**kwargs)
