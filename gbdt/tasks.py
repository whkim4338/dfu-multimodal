"""
gbdt/tasks.py — 7개 예측 태스크의 정의 (neural 경로의 Wagner+SINBAD 6개와 동일 태스크 집합).

GBDT는 신경망처럼 공유 trunk + 여러 head 구조가 자연스럽지 않아서, 태스크마다 완전히
독립적인 모델을 학습한다. 이 딕셔너리 하나가 "무엇을 예측할지"의 유일한 출처(source of
truth)이다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    label_col: str
    objective: str  # "binary" | "multiclass"
    excluded_features: tuple[str, ...] = ()  # 이 태스크에서만 제외할 feature (target과 겹치는 것 등)
    num_classes: int | None = None  # multiclass일 때만 사용. XGBoost/LightGBM은 명시적 num_class가 필요해서
                                      # 특정 split에 없는 등급이 있어도 항상 6(Wagner 0~5)으로 고정해둔다.


TASK_DEFINITIONS: dict[str, TaskSpec] = {
    "wagner": TaskSpec(label_col="wag", objective="multiclass", num_classes=6),
    "sinbad_site": TaskSpec(label_col="snb_site", objective="binary"),
    "sinbad_ischemia": TaskSpec(label_col="snb_isc", objective="binary"),
    "sinbad_neuropathy": TaskSpec(label_col="snb_neuro", objective="binary"),
    "sinbad_infection": TaskSpec(label_col="snb_bac", objective="binary"),
    "sinbad_area": TaskSpec(label_col="snb_area", objective="binary"),
    "sinbad_depth": TaskSpec(label_col="snb_depth", objective="binary"),
}
