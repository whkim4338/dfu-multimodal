from __future__ import annotations

import pandas as pd

# (혹시 컬럼(변수) 추가되면 여기도 같이 갱신)
BINARY_COLS = [
    "sex", "amputation_history", "esrd_proxy_ckd", "pad_symptom_proxy",
    "exudate_presence", "insulin_use", "exercise_any", "diabetic_neuropathy_dx",
    "ckd_or_nephropathy", "cardiovascular_disease_dx", "pad_diagnosis", "hypertension_dx",
]
# smoking_status(0=비흡연/1=과거흡연/2=현재흡연)는 순서형이라 age_band처럼 연속형 취급
# (BINARY_COLS에 안 넣으면 get_continuous_cols()가 자동으로 연속형으로 잡는다)

# 전부 결측인 컬럼 목록 (갱신 필요)
FULLY_MISSING_DROP_COLS = ["dfu_recurrence", "lops", "tg", "edema"]


def get_continuous_cols(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    return [
        c for c in df.columns
        if c not in excluded and c not in BINARY_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def fit_preprocessor(fit_df: pd.DataFrame, continuous_cols: list[str], binary_cols: list[str]) -> dict:
    """반드시 train subset에만 호출할 것 (leakage 방지)."""
    stats = {"continuous": {}, "binary_fill_value": 0.5}
    for col in continuous_cols:
        median, mean, std = fit_df[col].median(), fit_df[col].mean(), fit_df[col].std()
        stats["continuous"][col] = {
            "median": None if pd.isna(median) else float(median),
            "mean": None if pd.isna(mean) else float(mean),
            "std": None if (pd.isna(std) or std == 0) else float(std),
        }
    return stats


def transform_with_stats(
    target_df: pd.DataFrame, stats: dict, continuous_cols: list[str], binary_cols: list[str]
) -> pd.DataFrame:
    out = target_df.copy()
    for col in continuous_cols:
        s = stats["continuous"][col]
        out[f"{col}_missing_flag"] = out[col].isna().astype(float)
        out[col] = out[col].fillna(s["median"] if s["median"] is not None else 0.0)
        if s["mean"] is not None and s["std"] is not None:
            out[col] = (out[col] - s["mean"]) / s["std"]
        else:
            out[col] = 0.0
    for col in binary_cols:
        out[f"{col}_missing_flag"] = out[col].isna().astype(float)
        out[col] = out[col].fillna(stats["binary_fill_value"])
    return out


def preprocess_clinical_for_mlp(
    df: pd.DataFrame, id_and_meta_cols: list[str], train_mask: pd.Series
) -> tuple[pd.DataFrame, list[str]]:
    """전체 df(라벨/메타 포함)를 받아 완전결측 컬럼 제거 -> train만으로 fit ->
    전체 transform까지 한 번에 수행. 반환: (전처리된 df, feature 컬럼 목록).
    """
    drop_present = [c for c in FULLY_MISSING_DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_present)

    excluded = set(id_and_meta_cols)
    continuous_cols = get_continuous_cols(df, excluded)
    binary_cols = [c for c in BINARY_COLS if c in df.columns]

    stats = fit_preprocessor(df[train_mask], continuous_cols, binary_cols)
    df = transform_with_stats(df, stats, continuous_cols, binary_cols)

    feature_cols = continuous_cols + [f"{c}_missing_flag" for c in continuous_cols]
    feature_cols += binary_cols + [f"{c}_missing_flag" for c in binary_cols]
    return df, feature_cols
