"""Model evaluation utilities (multi-class)."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABELS = ["Rendah", "Sedang", "Tinggi"]


def _to_native(obj: Any):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    return obj


def evaluate_model(model, X, y, labels: list[str] | None = None) -> dict[str, Any]:
    labels = labels or LABELS
    pred = model.predict(X)
    cm = confusion_matrix(y, pred, labels=labels)
    report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    return _to_native(
        {
            "accuracy": accuracy_score(y, pred),
            "precision_macro": precision_score(y, pred, labels=labels, average="macro", zero_division=0),
            "recall_macro": recall_score(y, pred, labels=labels, average="macro", zero_division=0),
            "f1_macro": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
            "precision_weighted": precision_score(y, pred, labels=labels, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y, pred, labels=labels, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y, pred, labels=labels, average="weighted", zero_division=0),
            "confusion_matrix": cm,
            "labels": labels,
            "classification_report": report,
        }
    )


def compare_train_test(train_metrics: dict, test_metrics: dict) -> dict[str, Any]:
    def gap(k):
        return float(train_metrics.get(k, 0)) - float(test_metrics.get(k, 0))

    return {
        "accuracy_gap": gap("accuracy"),
        "f1_macro_gap": gap("f1_macro"),
        "overfit_warning": gap("accuracy") > 0.05 or gap("f1_macro") > 0.05,
    }
