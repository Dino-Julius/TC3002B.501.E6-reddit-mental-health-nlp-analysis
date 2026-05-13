"""Modelo baseline lineal para detección binaria de suicidalidad."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from reddit_mental_health.config import BaselineConfig, ensure_parent_dir
from reddit_mental_health.features import ajustar_transformar_tfidf, transformar_tfidf


@dataclass
class BaselineModel:
    """Artefacto completo del baseline: configuración, TF-IDF y clasificador."""

    config: BaselineConfig
    vectorizer: object
    classifier: LogisticRegression


def construir_clasificador(config: BaselineConfig) -> LogisticRegression:
    """Crea la regresión logística recomendada por el diseño conceptual."""

    return LogisticRegression(
        C=config.logistic_c,
        class_weight=config.class_weight,
        max_iter=config.logistic_max_iter,
        random_state=config.random_state,
    )


def entrenar_baseline(
    textos_entrenamiento: Iterable[str],
    y_entrenamiento: Iterable[int],
    config: BaselineConfig,
) -> BaselineModel:
    """Entrena TF-IDF + Regresión Logística sobre el conjunto de entrenamiento."""

    vectorizador, x_entrenamiento = ajustar_transformar_tfidf(
        textos_entrenamiento,
        config,
    )
    clasificador = construir_clasificador(config)
    y = np.asarray(list(y_entrenamiento), dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("El entrenamiento requiere al menos dos clases.")

    clasificador.fit(x_entrenamiento, y)
    return BaselineModel(
        config=config,
        vectorizer=vectorizador,
        classifier=clasificador,
    )


def predecir_baseline(
    modelo: BaselineModel,
    textos: Iterable[str],
) -> tuple[list[int], list[float]]:
    """Genera predicciones binarias y puntajes continuos para ROC AUC."""

    x = transformar_tfidf(modelo.vectorizer, textos)
    y_pred = modelo.classifier.predict(x).astype(int)

    if hasattr(modelo.classifier, "predict_proba"):
        clases = list(modelo.classifier.classes_)
        indice_positivo = clases.index(modelo.config.positive_value)
        score = modelo.classifier.predict_proba(x)[:, indice_positivo]
    else:
        score = modelo.classifier.decision_function(x)

    return y_pred.tolist(), score.astype(float).tolist()


def guardar_modelo(modelo: BaselineModel, path: str | Path) -> None:
    """Persiste el artefacto entrenado para evaluación posterior."""

    salida = Path(path)
    ensure_parent_dir(salida)
    joblib.dump(modelo, salida)


def cargar_modelo(path: str | Path) -> BaselineModel:
    """Carga un baseline entrenado desde disco."""

    return joblib.load(path)


__all__ = [
    "BaselineModel",
    "cargar_modelo",
    "construir_clasificador",
    "entrenar_baseline",
    "guardar_modelo",
    "predecir_baseline",
]
