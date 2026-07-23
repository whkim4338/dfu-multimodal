"""
trainers/metrics.py — neural/GBDT 공용 평가 지표 계산 (Project.md 공통 설계 원칙 7:
전략이 달라도 같은 스크립트로 비교표를 만들 수 있도록 스키마를 통일한다).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> dict[str, object]:
    """accuracy/balanced_accuracy/macro_f1(+ binary면 roc_auc)을 계산.

    y_score: binary 태스크의 양성 클래스 확률. 두 클래스가 모두 존재할 때만 roc_auc를
    계산하고, 그렇지 않으면(한쪽 클래스만 있는 소규모 split 등) None을 반환한다.
    """
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["roc_auc"] = None
    return metrics
