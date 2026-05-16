"""
Este módulo prueba las métricas de evaluación del baseline.
"""

from __future__ import annotations

import pytest

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.evaluation import calcular_metricas


def test_calcular_metricas_incluye_claves_esperadas() -> None:
    """
    Verifica que las métricas regresen las claves públicas esperadas.
    """

    config = BaselineConfig()
    metricas = calcular_metricas(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 0, 1],
        score=[0.1, 0.7, 0.3, 0.9],
        config=config,
    )

    assert {
        "roc_auc",
        "protocol_auc",
        "true_positive_rate",
        "false_positive_rate",
        "recall",
        "precision",
        "f1",
        "confusion_matrix",
    }.issubset(metricas)


def test_calcular_metricas_protocolo_con_matriz_conocida() -> None:
    """
    Verifica el AUC de protocolo con la matriz oficial conocida.
    """

    config = BaselineConfig()
    y_true = [0] * 135 + [1] * 166
    y_pred = [0] * 97 + [1] * 38 + [0] * 54 + [1] * 112
    score = [0.1] * 97 + [0.8] * 38 + [0.2] * 54 + [0.9] * 112

    metricas = calcular_metricas(y_true, y_pred, score, config)

    assert metricas["true_positive_rate"] == pytest.approx(112 / (112 + 54))
    assert metricas["false_positive_rate"] == pytest.approx(38 / (38 + 97))
    assert metricas["protocol_auc"] == pytest.approx(0.6966086568503251)
    assert metricas["confusion_matrix"] == {
        "labels": ["no", "yes"],
        "true_negative": 97,
        "false_positive": 38,
        "false_negative": 54,
        "true_positive": 112,
    }


def test_calcular_metricas_maneja_una_sola_clase() -> None:
    """
    Verifica comportamiento seguro cuando solo existe una clase real.
    """

    config = BaselineConfig()
    metricas_positivos = calcular_metricas(
        y_true=[1, 1, 1],
        y_pred=[1, 1, 1],
        score=[0.9, 0.8, 0.7],
        config=config,
    )
    assert metricas_positivos["roc_auc"] is None
    assert metricas_positivos["false_positive_rate"] == 0.0
    assert metricas_positivos["true_positive_rate"] == 1.0
    assert metricas_positivos["protocol_auc"] == 1.0

    metricas_negativos = calcular_metricas(
        y_true=[0, 0, 0],
        y_pred=[0, 0, 0],
        score=[0.1, 0.2, 0.3],
        config=config,
    )
    assert metricas_negativos["roc_auc"] is None
    assert metricas_negativos["true_positive_rate"] == 0.0
    assert metricas_negativos["false_positive_rate"] == 0.0
    assert metricas_negativos["protocol_auc"] == 0.5
