"""
gbdt/tabular_builder.py — image embedding + clinical raw feature를 하나의 flat feature
목록으로 합친다. clinical_transform.py의 표준화/결측 대체는 거치지 않는다 — CatBoost/
XGBoost/LightGBM은 결측치를 자체적으로 처리하므로 dfu_gbdt와 동일하게 raw 값을 그대로
넘긴다 (Project.md 4.3).
"""

from __future__ import annotations

import pandas as pd

from data.clinical_loader import get_feature_columns


def build_tabular_frame(merged_df: pd.DataFrame, image_emb_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """merged_df(clinical raw feature + image embedding + 라벨이 전부 있는 DataFrame)를 받아
    GBDT가 바로 쓸 수 있는 (DataFrame, feature 컬럼 목록)을 반환한다.

    merged_df에는 image embedding 컬럼(emb_0..emb_N)도 이미 섞여 있으므로, clinical feature만
    골라내려면 get_feature_columns()가 뽑은 목록에서 image_emb_cols를 다시 제외해야 한다.
    """
    clinical_feature_cols = [c for c in get_feature_columns(merged_df) if c not in set(image_emb_cols)]
    feature_cols = clinical_feature_cols + image_emb_cols
    return merged_df, feature_cols
