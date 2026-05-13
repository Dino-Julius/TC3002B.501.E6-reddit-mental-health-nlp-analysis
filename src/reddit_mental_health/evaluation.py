"""Métricas de evaluación para el baseline binario."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from reddit_mental_health.config import BaselineConfig, ensure_parent_dir


def calcular_metricas(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    score: list[float] | np.ndarray,
    config: BaselineConfig,
) -> dict[str, object]:
    """Calcula ROC AUC y métricas complementarias del protocolo."""

    y_real = np.asarray(y_true, dtype=int)
    y_estimado = np.asarray(y_pred, dtype=int)
    puntajes = np.asarray(score, dtype=float)

    roc_auc = None
    if len(np.unique(y_real)) == 2:
        roc_auc = float(roc_auc_score(y_real, puntajes))

    matriz = confusion_matrix(
        y_real,
        y_estimado,
        labels=[config.negative_value, config.positive_value],
    )
    tn, fp, fn, tp = matriz.ravel()

    return {
        "roc_auc": roc_auc,
        "recall": float(recall_score(y_real, y_estimado, zero_division=0)),
        "precision": float(precision_score(y_real, y_estimado, zero_division=0)),
        "f1": float(f1_score(y_real, y_estimado, zero_division=0)),
        "confusion_matrix": {
            "labels": [config.negative_label, config.positive_label],
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def guardar_metricas(metricas: dict[str, object], path: str | Path) -> None:
    """Guarda las métricas en JSON legible y reproducible."""

    salida = Path(path)
    ensure_parent_dir(salida)
    salida.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = ["calcular_metricas", "guardar_metricas"]
