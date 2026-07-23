from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from models.fusion_model import SINBAD_COMPONENTS

# SINBAD component 이름 -> 실제 eCRF 컬럼명
SINBAD_COLUMN_MAP = {
    "site": "snb_site",
    "ischemia": "snb_isc",
    "neuropathy": "snb_neuro",
    "infection": "snb_bac",
    "area": "snb_area",
    "depth": "snb_depth",
}


class DFUMultimodalDataset(Dataset):
    """clinical feature + image embedding + 라벨 컬럼이 전부 있는 DataFrame을 받아,
    태스크별 라벨/마스크가 포함된 텐서로 변환한다 (neural fusion 전략 전용 — GBDT 경로는
    gbdt/tabular_builder.py가 별도로 처리).

    라벨 결측 처리: 태스크마다 라벨이 없는 행은 값 대신 0을 채우고 별도 mask=0으로 표시한다
    (실제 loss/평가에서는 mask로 걸러지므로 0이라는 값 자체는 의미 없음 — 단순히 텐서 collate가
    가능하도록 자리만 채우는 것).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        clinical_feature_cols: list[str],
        image_emb_cols: list[str],
        wagner_col: str = "wag",
        group_col: str = "id",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.clinical_feature_cols = clinical_feature_cols
        self.image_emb_cols = image_emb_cols
        self.wagner_col = wagner_col
        self.group_col = group_col

        missing_sinbad = [c for c in SINBAD_COLUMN_MAP.values() if c not in df.columns]
        if missing_sinbad:
            raise ValueError(f"DataFrame에 SINBAD 컬럼이 없습니다: {missing_sinbad}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        clinical_feat = torch.tensor(row[self.clinical_feature_cols].to_numpy(dtype="float32"))
        image_emb = torch.tensor(row[self.image_emb_cols].to_numpy(dtype="float32"))

        wagner_value = row[self.wagner_col]
        wagner_mask = 0.0 if pd.isna(wagner_value) else 1.0
        wagner_label = 0.0 if pd.isna(wagner_value) else float(wagner_value)

        sinbad_labels, sinbad_masks = {}, {}
        for name, col in SINBAD_COLUMN_MAP.items():
            value = row[col]
            sinbad_masks[name] = 0.0 if pd.isna(value) else 1.0
            sinbad_labels[name] = 0.0 if pd.isna(value) else float(value)

        return {
            "id": row[self.group_col],
            "clinical_feat": clinical_feat,
            "image_emb": image_emb,
            "wagner_label": torch.tensor(wagner_label),
            "wagner_mask": torch.tensor(wagner_mask),
            "sinbad_labels": {k: torch.tensor(v) for k, v in sinbad_labels.items()},
            "sinbad_masks": {k: torch.tensor(v) for k, v in sinbad_masks.items()},
        }


def multimodal_collate(batch: list[dict]) -> dict:
    """dict 안에 dict(sinbad_labels/masks)가 중첩돼 있어 default_collate 대신 직접 구현."""
    out = {
        "id": [b["id"] for b in batch],
        "clinical_feat": torch.stack([b["clinical_feat"] for b in batch]),
        "image_emb": torch.stack([b["image_emb"] for b in batch]),
        "wagner_label": torch.stack([b["wagner_label"] for b in batch]),
        "wagner_mask": torch.stack([b["wagner_mask"] for b in batch]),
    }
    out["sinbad_labels"] = {
        name: torch.stack([b["sinbad_labels"][name] for b in batch]) for name in SINBAD_COMPONENTS
    }
    out["sinbad_masks"] = {
        name: torch.stack([b["sinbad_masks"][name] for b in batch]) for name in SINBAD_COMPONENTS
    }
    return out
