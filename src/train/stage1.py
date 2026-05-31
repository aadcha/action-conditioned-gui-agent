"""Stage 1 training loop on cached features.

Designed to run anywhere — Modal worker, local CPU, GPU notebook. Operates on
in-memory feature tensors so it doesn't care where they were extracted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.data.taxonomy import ID_TO_ACTION
from src.models.stage1_classifier import Stage1Classifier, Stage1Config


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 128
    class_weighted: bool = True
    seed: int = 42


def _make_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, scaled to mean 1.0 so the loss scale isn't shifted."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    # avoid div-by-zero for absent classes
    safe_counts = counts.clamp(min=1.0)
    weights = safe_counts.sum() / (num_classes * safe_counts)
    # zero out classes that weren't seen (recall they would never contribute anyway)
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    # normalize so mean weight over present classes is 1.0
    present = (counts > 0).sum().clamp(min=1)
    scale = present / weights[counts > 0].sum().clamp(min=1e-8)
    return weights * scale


def evaluate(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    num_classes: int = 8,
) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()
    y_true = labels.cpu().numpy()
    labels_present = sorted(set(y_true.tolist()))
    target_names = [ID_TO_ACTION[i] for i in labels_present]

    cls_report = classification_report(
        y_true,
        preds,
        labels=labels_present,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, labels=labels_present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, labels=labels_present, average="weighted", zero_division=0)),
        "per_class": {
            target_names[i]: cls_report[target_names[i]] for i in range(len(target_names))
        },
        "confusion_matrix": confusion_matrix(y_true, preds, labels=labels_present).tolist(),
        "confusion_matrix_labels": target_names,
        "labels_present": labels_present,
        "prediction_distribution": {ID_TO_ACTION[c]: int(v) for c, v in Counter(preds.tolist()).items()},
        "ground_truth_distribution": {ID_TO_ACTION[c]: int(v) for c, v in Counter(y_true.tolist()).items()},
    }


def train_stage1(
    features_train: torch.Tensor,
    labels_train: torch.Tensor,
    features_val: torch.Tensor,
    labels_val: torch.Tensor,
    model_cfg: Stage1Config | None = None,
    train_cfg: TrainConfig | None = None,
    device: torch.device | str | None = None,
) -> dict:
    """Train the MLP, return metrics + best-epoch state.

    Reports val macro-F1 every epoch, returns the best one.
    """
    if model_cfg is None:
        model_cfg = Stage1Config(feature_dim=features_train.shape[1])
    elif model_cfg.feature_dim != features_train.shape[1]:
        # auto-adjust to match the cached features
        model_cfg = Stage1Config(**{**asdict(model_cfg), "feature_dim": features_train.shape[1]})
    if train_cfg is None:
        train_cfg = TrainConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    model = Stage1Classifier(model_cfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    if train_cfg.class_weighted:
        weights = _make_class_weights(labels_train, model_cfg.num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(features_train, labels_train),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    history: list[dict] = []
    best_val_macro_f1 = -1.0
    best_metrics: dict = {}
    best_epoch = -1
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        val_metrics = evaluate(model, features_val, labels_val, device, model_cfg.num_classes)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
        })
        print(
            f"[stage1] epoch {epoch}/{train_cfg.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_metrics": best_metrics,
        "history": history,
        "train_cfg": asdict(train_cfg),
        "model_cfg": asdict(model_cfg),
        "best_state": best_state,
        "num_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
