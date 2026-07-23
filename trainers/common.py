from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class FusionTrainerArgs:
    """neural fusion 경로(concat/gated/...) 학습 설정. image-size는 여기 없음 
    — DINOv3 forward pass 자체가 embedding_cache 단계에서 이미 끝나 있기 때문."""

    run_name: str = field(default_factory=lambda: dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    checkpoint_dir: str = "checkpoints"
    fusion_strategy: str = "concat"

    batch_size: int = 32
    epochs: int = 30
    lr: float = 5e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 7

    val_ratio: float = 0.2
    test_ratio: float = 0.0  # 데이터가 아주 적을 때는 val만 두고 0으로 둘 수 있음
    group_col: str = "id"
    seed: int = 42

    wagner_num_classes: int = 6
    branch_out_dim: int = 256
    trunk_out_dim: int = 128
    adapter_out_dim: int = 64

    @property
    def run_dir(self) -> str:
        return f"{self.checkpoint_dir}/{self.run_name}"


def split_by_group(
    df: pd.DataFrame,
    group_col: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """환자(id) 단위 GroupShuffleSplit.

    neural 경로(trainers/neural_trainer.py)와 GBDT 경로(gbdt/gbdt_trainer.py) 양쪽이 반드시
    이 함수 하나만 거치게 해서, 같은 (group_col, val_ratio, test_ratio, seed)라면 fusion
    전략이 달라도 완전히 같은 환자 집합이 train/val로 나뉘도록 보장한다. 
    """
    groups = df[group_col]
    n_unique = groups.nunique()
    if n_unique < 2:
        return df, df.iloc[0:0], None  # 평가 불가, 전체로만 학습

    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=groups))
    train_df, val_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

    test_df = None
    if test_ratio > 0:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        tr_idx, te_idx = next(gss2.split(train_df, groups=train_df[group_col]))
        test_df = train_df.iloc[te_idx].reset_index(drop=True)
        train_df = train_df.iloc[tr_idx].reset_index(drop=True)

    return train_df, val_df, test_df
