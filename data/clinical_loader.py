"""
data/clinical_loader.py — 단일 CSV(model_input_variables.csv에 id/img2d/라벨 컬럼이
포함된 버전) 로딩.

⚠️ 이 CSV는 raw feature 상태(NaN 유지, imputation/standardization 이전)여야 한다.
   median 대체나 standardization까지 이미 적용된 CSV를 넣으면, 그 통계가 전체 데이터
   기준으로 계산된 것이라 train/val 분리 이전에 leakage가 생긴다 — 반드시 이 로더 -> 이후
   preprocess_clinical_for_mlp(train split로만 fit)까지 거친 뒤에 학습해야 한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ID_COL = "id"
VISIT_COL = "visit"
IMAGE_COL = "img2d"

LABEL_COLS = [
    "wag", "snb_site", "snb_isc", "snb_neuro", "snb_bac", "snb_area", "snb_depth",
    "snb_score",
]

REQUIRED_COLS = {ID_COL, VISIT_COL, IMAGE_COL}


def load_clinical_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"파일 없음: {csv_path}")

    df = pd.read_csv(csv_path, dtype={ID_COL: str})

    missing_required = REQUIRED_COLS - set(df.columns)
    if missing_required:
        raise ValueError(
            f"CSV에 필수 컬럼이 없습니다: {missing_required}. "
            f"이 파이프라인은 '{ID_COL}'(환자 id), '{VISIT_COL}'(방문 회차), "
            f"'{IMAGE_COL}'(이미지 파일명)이 전부 있는 CSV를 기대합니다."
        )

    missing_labels = set(LABEL_COLS) - set(df.columns)
    if missing_labels:
        import warnings
        warnings.warn(f"CSV에 없는 라벨 컬럼: {missing_labels} — 해당 태스크는 전부 결측으로 처리됩니다.")

    n_missing_img2d = df[IMAGE_COL].isna().sum()
    if n_missing_img2d > 0:
        import warnings
        warnings.warn(f"{IMAGE_COL}가 비어있는 행 {n_missing_img2d}개 — 이 행들은 이미지 병합 단계에서 제외됩니다.")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """id/img2d/라벨을 제외한 순수 clinical feature 컬럼만 추출."""
    excluded = REQUIRED_COLS | set(LABEL_COLS)
    return [c for c in df.columns if c not in excluded]
