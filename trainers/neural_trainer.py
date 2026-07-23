from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.fusion_dataset import DFUMultimodalDataset, multimodal_collate
from models.coral_head import coral_predict
from models.fusion_model import DFUMultimodalModel, SINBAD_COMPONENTS
from trainers.common import FusionTrainerArgs, split_by_group
from trainers.losses import MultiTaskLoss
from trainers.metrics import classification_metrics


def _to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_losses(model: DFUMultimodalModel, loader: DataLoader, loss_fn: MultiTaskLoss, device: str) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = _to_device(batch, device)
        outputs = model(batch["image_emb"], batch["clinical_feat"])
        losses = loss_fn(outputs, batch)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_task_metrics(model: DFUMultimodalModel, loader: DataLoader, device: str) -> dict[str, dict | None]:
    """태스크별(Wagner + SINBAD 6개) accuracy/balanced_accuracy/macro_f1/roc_auc 계산.

    라벨 mask=0인 행은 각 태스크 평가에서 제외한다(다른 태스크 평가에는 영향 없음 —
    Project.md 공통 설계 원칙 3과 동일한 태스크별 결측 처리). 라벨이 하나도 없는 태스크는
    None을 반환한다.
    """
    model.eval()
    wagner_true: list[float] = []
    wagner_pred: list[float] = []
    sinbad_true = {name: [] for name in SINBAD_COMPONENTS}
    sinbad_pred = {name: [] for name in SINBAD_COMPONENTS}
    sinbad_score = {name: [] for name in SINBAD_COMPONENTS}

    for batch in loader:
        batch = _to_device(batch, device)
        outputs = model(batch["image_emb"], batch["clinical_feat"])

        wagner_mask = batch["wagner_mask"].bool().cpu().numpy()
        if wagner_mask.any():
            pred_batch = coral_predict(outputs["wagner"]).cpu().numpy()
            true_batch = batch["wagner_label"].cpu().numpy()
            wagner_true.extend(true_batch[wagner_mask].tolist())
            wagner_pred.extend(pred_batch[wagner_mask].tolist())

        for name in SINBAD_COMPONENTS:
            mask = batch["sinbad_masks"][name].bool().cpu().numpy()
            if not mask.any():
                continue
            score = torch.sigmoid(outputs["sinbad"][name]).cpu().numpy()
            pred = (score > 0.5).astype(float)
            true = batch["sinbad_labels"][name].cpu().numpy()
            sinbad_true[name].extend(true[mask].tolist())
            sinbad_pred[name].extend(pred[mask].tolist())
            sinbad_score[name].extend(score[mask].tolist())

    results: dict[str, dict | None] = {}
    if wagner_true:
        metrics = classification_metrics(np.array(wagner_true), np.array(wagner_pred))
        results["wagner"] = {"n_val_labeled": len(wagner_true), **metrics}
    else:
        results["wagner"] = None

    for name in SINBAD_COMPONENTS:
        if sinbad_true[name]:
            metrics = classification_metrics(
                np.array(sinbad_true[name]), np.array(sinbad_pred[name]), np.array(sinbad_score[name])
            )
            results[name] = {"n_val_labeled": len(sinbad_true[name]), **metrics}
        else:
            results[name] = None

    return results


def train_neural_model(
    df: pd.DataFrame,
    clinical_feature_cols: list[str],
    image_emb_cols: list[str],
    args: FusionTrainerArgs,
    device: str = "cpu",
) -> dict:
    """concat/gated 등 neural fusion 전략 학습 실행. 반환: {'model', 'best_val_loss', 'history', 'task_metrics'}."""
    train_df, val_df, _ = split_by_group(df, args.group_col, args.val_ratio, args.test_ratio, args.seed)

    train_ds = DFUMultimodalDataset(train_df, clinical_feature_cols, image_emb_cols)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=multimodal_collate
    )

    has_val = len(val_df) > 0
    val_loader = None
    if has_val:
        val_ds = DFUMultimodalDataset(val_df, clinical_feature_cols, image_emb_cols)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=multimodal_collate)

    model = DFUMultimodalModel(
        clinical_in_dim=len(clinical_feature_cols),
        image_in_dim=len(image_emb_cols),
        fusion_strategy=args.fusion_strategy,
        branch_out_dim=args.branch_out_dim,
        trunk_out_dim=args.trunk_out_dim,
        adapter_out_dim=args.adapter_out_dim,
        wagner_num_classes=args.wagner_num_classes,
    ).to(device)

    loss_fn = MultiTaskLoss(wagner_num_classes=args.wagner_num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_losses: dict[str, float] = {}
        n_batches = 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            outputs = model(batch["image_emb"], batch["clinical_feat"])
            losses = loss_fn(outputs, batch)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            for k, v in losses.items():
                train_losses[k] = train_losses.get(k, 0.0) + v.item()
            n_batches += 1
        train_losses = {k: v / max(n_batches, 1) for k, v in train_losses.items()}

        if has_val:
            val_losses = evaluate_losses(model, val_loader, loss_fn, device)
            monitor_loss = val_losses["total"]
        else:
            val_losses = {}
            monitor_loss = train_losses["total"]

        history.append({"epoch": epoch, "train": train_losses, "val": val_losses})
        print(
            f"[{args.fusion_strategy}][epoch {epoch}] train_total={train_losses['total']:.4f}"
            + (f" val_total={val_losses['total']:.4f}" if has_val else " (val 없음, train으로 모니터링)")
        )

        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            epochs_without_improvement = 0
            save_checkpoint(model, args, epoch, {"best_val_loss": best_val_loss}, run_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        save_checkpoint(model, args, epoch, {"monitor_loss": monitor_loss}, run_dir / "last.pt")

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"[epoch {epoch}] early stopping (patience={args.early_stopping_patience})")
            break

    task_metrics = evaluate_task_metrics(model, val_loader, device) if has_val else {}

    summary = {
        "fusion_strategy": args.fusion_strategy,
        "n_total": len(df),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "best_val_loss": best_val_loss,
        "tasks": {
            name: (metrics if metrics is not None else {"skipped_reason": "val에 라벨 있는 행 없음"})
            for name, metrics in task_metrics.items()
        },
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"[train_neural_model] 결과 요약 저장: {run_dir / 'summary.json'}")

    return {"model": model, "best_val_loss": best_val_loss, "history": history, "task_metrics": task_metrics}


def save_checkpoint(model: DFUMultimodalModel, args: FusionTrainerArgs, epoch: int, metrics: dict, path: Path) -> None:
    torch.save(
        {
            "head_state_dict": model.state_dict(),
            "args": args,
            "metrics": metrics,
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[DFUMultimodalModel, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args: FusionTrainerArgs = checkpoint["args"]
    model = DFUMultimodalModel(
        clinical_in_dim=checkpoint["head_state_dict"]["clinical_mlp.net.0.weight"].shape[1],
        image_in_dim=checkpoint["head_state_dict"]["image_projection.net.0.weight"].shape[1],
        fusion_strategy=args.fusion_strategy,
        branch_out_dim=args.branch_out_dim,
        trunk_out_dim=args.trunk_out_dim,
        adapter_out_dim=args.adapter_out_dim,
        wagner_num_classes=args.wagner_num_classes,
    )
    missing, unexpected = model.load_state_dict(checkpoint["head_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"체크포인트 로딩 불일치 — missing={missing}, unexpected={unexpected}")
    model.to(device)
    return model, checkpoint
