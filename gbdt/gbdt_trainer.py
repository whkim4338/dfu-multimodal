"""
gbdt/gbdt_trainer.py — 태스크별 GBDT 학습. CatBoost / XGBoost / LightGBM 세 백엔드를 동일한
인터페이스로 지원한다.

neural 경로(trainers/neural_trainer.py)와의 핵심 차이 — Project.md 4.3:
    1. 신경망 shared trunk/CORAL head 없이 태스크마다 완전히 독립적인 모델을 학습한다.
    2. split을 태스크마다 새로 만들지 않는다. trainers/common.py의 split_by_group을
       한 번만 호출해 얻은 환자 집합(train_ids/val_ids)을 모든 태스크가 그대로 공유한다 —
       그래야 concat/gated/gbdt 세 전략이 완전히 같은 환자로 나뉜 fold에서 비교된다.
    3. 평가 지표는 trainers/metrics.py의 classification_metrics를 그대로 재사용해
       neural 경로와 같은 스키마(accuracy/balanced_accuracy/macro_f1/roc_auc)로 저장한다.

백엔드별 차이(내부에서 흡수, 호출부는 몰라도 됨):
    - CatBoost: cat_features 컬럼을 문자열로 캐스팅해서 넘김 (float dtype 거부 버그 회피)
    - XGBoost/LightGBM: cat_features 컬럼을 pandas 'category' dtype으로 캐스팅
    - LightGBM: 기본 min_data_in_leaf(20)가 소규모 데이터에서 트리가 거의 안 갈라지는
      문제를 일으킬 수 있어, min_child_samples를 GBDTConfig에서 조절 가능하게 뒀다
      (기본값 5로 낮춰둠).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gbdt.tasks import TaskSpec
from trainers.common import FusionTrainerArgs, split_by_group
from trainers.metrics import classification_metrics

SUPPORTED_BACKENDS = ("catboost", "xgboost", "lightgbm")


@dataclass
class GBDTConfig:
    backend: str = "catboost"  # "catboost" | "xgboost" | "lightgbm"
    iterations: int = 300
    depth: int = 4
    learning_rate: float = 0.05
    random_state: int = 42
    min_samples: int = 10
    cat_features: list[str] = field(default_factory=lambda: ["age_band"])
    min_child_samples: int = 5  # LightGBM 전용. 소규모 데이터 대응으로 기본값(20)보다 낮춤

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"지원하지 않는 backend: {self.backend!r} (지원: {SUPPORTED_BACKENDS})")


@dataclass
class TaskResult:
    task_name: str
    backend: str
    model: object | None
    n_total: int
    n_available: int
    n_train_labeled: int
    n_val_labeled: int
    metrics: dict | None
    skipped_reason: str | None = None


# ---------------------------------------------------------------
# 백엔드별 fit 함수 — 여기서만 라이브러리별 차이를 흡수한다
# ---------------------------------------------------------------

def _fit_catboost(X_train, y_train, cat_features, config: GBDTConfig, spec: TaskSpec, reference_categories=None):
    # reference_categories는 xgboost/lightgbm 전용 (CatBoost는 문자열 캐스팅만으로 충분해
    # unseen category 문제가 없음) — 시그니처만 맞추기 위해 받고 사용하지 않는다.
    from catboost import CatBoostClassifier

    X_train = X_train.copy()
    for col in cat_features:
        X_train[col] = X_train[col].astype(str)  # float dtype cat_features를 CatBoost가 거부하는 문제 회피

    loss_function = "MultiClass" if spec.objective == "multiclass" else "Logloss"
    model = CatBoostClassifier(
        iterations=config.iterations, depth=config.depth, learning_rate=config.learning_rate,
        loss_function=loss_function, cat_features=cat_features,
        random_state=config.random_state, verbose=False,
    )
    model.fit(X_train, y_train)
    return model


def _cast_categorical(
    X: pd.DataFrame, cat_features: list[str], reference_categories: dict[str, list] | None = None
) -> pd.DataFrame:
    """cat_features 컬럼을 pandas 'category' dtype으로 변환.

    reference_categories를 주면(보통 train+val을 합친 전체 X에서 뽑은 카테고리 목록) train/val이
    같은 카테고리 집합을 공유하게 강제한다 — XGBoost의 네이티브 categorical 처리는 predict
    시점에 train에 없던 카테고리가 val에 나타나면 에러를 내기 때문에(dfu_gbdt에서 실제로
    겪은 버그: age_band=15.0이 train split엔 없고 val split에만 있던 경우), 이 파라미터 없이는
    특정 seed에서 크래시가 날 수 있다.
    """
    X = X.copy()
    for col in cat_features:
        X[col] = X[col].astype(str)
        if reference_categories is not None and col in reference_categories:
            X[col] = pd.Categorical(X[col], categories=reference_categories[col])
        else:
            X[col] = X[col].astype("category")
    return X


def _fit_xgboost(X_train, y_train, cat_features, config: GBDTConfig, spec: TaskSpec, reference_categories=None):
    from xgboost import XGBClassifier

    X_train = _cast_categorical(X_train, cat_features, reference_categories)

    kwargs = dict(
        n_estimators=config.iterations, max_depth=config.depth, learning_rate=config.learning_rate,
        enable_categorical=True, tree_method="hist",
        random_state=config.random_state, eval_metric="logloss",
    )
    if spec.objective == "multiclass":
        kwargs["objective"] = "multi:softprob"
        kwargs["num_class"] = spec.num_classes
        kwargs["eval_metric"] = "mlogloss"
    else:
        kwargs["objective"] = "binary:logistic"

    model = XGBClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model


def _fit_lightgbm(X_train, y_train, cat_features, config: GBDTConfig, spec: TaskSpec, reference_categories=None):
    from lightgbm import LGBMClassifier

    X_train = _cast_categorical(X_train, cat_features, reference_categories)

    kwargs = dict(
        n_estimators=config.iterations, max_depth=config.depth, learning_rate=config.learning_rate,
        random_state=config.random_state, verbose=-1,
        min_child_samples=config.min_child_samples,
    )
    if spec.objective == "multiclass":
        kwargs["objective"] = "multiclass"
        kwargs["num_class"] = spec.num_classes
    else:
        kwargs["objective"] = "binary"

    model = LGBMClassifier(**kwargs)
    model.fit(X_train, y_train, categorical_feature=cat_features if cat_features else "auto")
    return model


_FIT_FUNCTIONS = {
    "catboost": _fit_catboost,
    "xgboost": _fit_xgboost,
    "lightgbm": _fit_lightgbm,
}


def _predict_for_eval(model, X_val, backend: str, cat_features: list[str], reference_categories=None):
    if backend != "catboost":
        X_val = _cast_categorical(X_val, cat_features, reference_categories)
    else:
        X_val = X_val.copy()
        for col in cat_features:
            X_val[col] = X_val[col].astype(str)
    preds = model.predict(X_val)
    preds = np.asarray(preds).reshape(-1)  # xgboost/lightgbm는 (N,), catboost는 (N,1)일 수 있어 통일
    proba = model.predict_proba(X_val)
    return preds, proba


# ---------------------------------------------------------------
# 공통 학습/평가 로직
# ---------------------------------------------------------------

def train_task_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    task_name: str,
    spec: TaskSpec,
    config: GBDTConfig,
    n_total: int,
) -> TaskResult:
    """train_df/val_df는 이미 (1) split_by_group으로 나뉜 환자 집합 기준, (2) 이 태스크의
    라벨이 있는 행만 남긴 상태로 호출부(train_all_tasks)에서 넘어온다 — 여기서는 그 fold를
    그대로 믿고 fit/eval만 한다.
    """
    n_train_labeled, n_val_labeled = len(train_df), len(val_df)
    n_available = n_train_labeled + n_val_labeled

    if n_available < config.min_samples:
        return TaskResult(
            task_name=task_name, backend=config.backend, model=None,
            n_total=n_total, n_available=n_available,
            n_train_labeled=n_train_labeled, n_val_labeled=n_val_labeled, metrics=None,
            skipped_reason=f"표본 부족 ({n_available} < min_samples={config.min_samples})",
        )

    task_feature_cols = [c for c in feature_cols if c not in spec.excluded_features]
    if spec.excluded_features:
        print(f"[{task_name}] target과 겹치는 feature 제외: {list(spec.excluded_features)}")

    cat_features = [c for c in config.cat_features if c in task_feature_cols]

    X_train, y_train = train_df[task_feature_cols], train_df[spec.label_col].astype(int)
    X_val, y_val = val_df[task_feature_cols], val_df[spec.label_col].astype(int)

    # train/val이 같은 categorical 카테고리 집합을 공유하도록, 둘을 합친 기준으로 카테고리
    # 목록을 미리 확정해둔다 (XGBoost가 val에만 있는 미지의 카테고리를 만나면 에러를 내는
    # 문제 예방).
    combined_X = pd.concat([X_train, X_val]) if len(X_val) else X_train
    reference_categories = {col: sorted(combined_X[col].astype(str).unique()) for col in cat_features}

    fit_fn = _FIT_FUNCTIONS[config.backend]
    model = fit_fn(X_train, y_train, cat_features, config, spec, reference_categories)

    metrics = None
    if len(X_val) > 0:
        preds, proba = _predict_for_eval(model, X_val, config.backend, cat_features, reference_categories)
        y_score = proba[:, 1] if (spec.objective == "binary" and proba.ndim == 2) else None
        metrics = classification_metrics(y_val.to_numpy(), preds, y_score)

    return TaskResult(
        task_name=task_name, backend=config.backend, model=model,
        n_total=n_total, n_available=n_available,
        n_train_labeled=n_train_labeled, n_val_labeled=n_val_labeled, metrics=metrics,
    )


def train_all_tasks(
    df: pd.DataFrame,
    feature_cols: list[str],
    task_definitions: dict[str, TaskSpec],
    config: GBDTConfig,
    train_ids: set[str],
    val_ids: set[str],
    group_col: str,
) -> list[TaskResult]:
    n_total = len(df)
    results = []
    for task_name, spec in task_definitions.items():
        task_df = df[df[spec.label_col].notna()]
        train_df = task_df[task_df[group_col].isin(train_ids)]
        val_df = task_df[task_df[group_col].isin(val_ids)]

        result = train_task_model(train_df, val_df, feature_cols, task_name, spec, config, n_total)
        if result.skipped_reason:
            print(f"[{task_name}] 스킵: {result.skipped_reason}")
        else:
            m = result.metrics or {}
            print(
                f"[{task_name}][{config.backend}] 라벨 {result.n_available}건 "
                f"(train {result.n_train_labeled} / val {result.n_val_labeled}), "
                f"accuracy={m.get('accuracy')}, roc_auc={m.get('roc_auc')}"
            )
        results.append(result)
    return results


def save_models(results: list[TaskResult], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        if result.model is None:
            continue
        path = run_dir / f"{result.task_name}_{result.backend}.pkl"
        with open(path, "wb") as f:
            pickle.dump(result.model, f)
        print(f"[save_models] 저장 완료: {path}")


def build_summary(results: list[TaskResult], fusion_strategy: str, n_total: int, n_train: int, n_val: int) -> dict:
    """neural 경로(trainers/neural_trainer.py)의 summary.json과 같은 최상위 스키마
    (fusion_strategy/n_total/n_train/n_val/tasks)를 쓴다 — n_train/n_val은 라벨 유무와
    무관하게 환자 단위 split으로 나뉜 전체 행 수(neural과 동일 정의), 태스크별 실제 라벨
    보유 행 수는 tasks 안의 n_train_labeled/n_val_labeled에 담긴다."""
    tasks = {}
    for result in results:
        if result.skipped_reason:
            tasks[result.task_name] = {"skipped_reason": result.skipped_reason}
        else:
            tasks[result.task_name] = {
                "n_available": result.n_available,
                "n_train_labeled": result.n_train_labeled,
                "n_val_labeled": result.n_val_labeled,
                **(result.metrics or {}),
            }
    return {
        "fusion_strategy": fusion_strategy,
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "tasks": tasks,
    }


def run_gbdt_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    task_definitions: dict[str, TaskSpec],
    trainer_args: FusionTrainerArgs,
    gbdt_config: GBDTConfig,
) -> dict:
    """GBDT 경로 전체 실행: 공유 split_by_group -> 태스크별 독립 학습/평가 -> 모델/요약 저장."""
    train_df, val_df, _ = split_by_group(
        df, trainer_args.group_col, trainer_args.val_ratio, trainer_args.test_ratio, trainer_args.seed,
    )
    train_ids = set(train_df[trainer_args.group_col])
    val_ids = set(val_df[trainer_args.group_col])

    results = train_all_tasks(df, feature_cols, task_definitions, gbdt_config, train_ids, val_ids, trainer_args.group_col)

    run_dir = Path(trainer_args.run_dir)
    save_models(results, run_dir)
    summary = build_summary(results, trainer_args.fusion_strategy, len(df), len(train_df), len(val_df))
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"[run_gbdt_pipeline] 결과 요약 저장: {run_dir / 'summary.json'}")

    return {"results": results, "summary": summary}
