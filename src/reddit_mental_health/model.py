"""Modelo baseline lineal para detección binaria de suicidalidad."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

from reddit_mental_health.config import BaselineConfig, ensure_parent_dir
from reddit_mental_health.features import ajustar_transformar_tfidf, transformar_tfidf


@dataclass
class BaselineModel:
    """Artefacto completo del baseline: configuración, TF-IDF y clasificador."""

    config: BaselineConfig
    vectorizer: object
    classifier: ClassifierMixin


def construir_clasificador(config: BaselineConfig) -> ClassifierMixin:
    """Crea el clasificador solicitado por la configuración experimental."""

    if config.classifier_name == "logistic_regression":
        return LogisticRegression(
            C=config.logistic_c,
            class_weight=config.class_weight,
            max_iter=config.logistic_max_iter,
            random_state=config.random_state,
        )
    if config.classifier_name == "linear_svm":
        return LinearSVC(
            C=config.linear_svm_c,
            class_weight=config.class_weight,
            dual="auto",
            max_iter=config.linear_svm_max_iter,
            random_state=config.random_state,
        )
    if config.classifier_name == "sgd_logistic":
        return SGDClassifier(
            alpha=config.sgd_alpha,
            class_weight=config.class_weight,
            loss="log_loss",
            max_iter=config.sgd_max_iter,
            random_state=config.random_state,
        )
    if config.classifier_name == "multinomial_nb":
        return MultinomialNB(alpha=config.naive_bayes_alpha)
    if config.classifier_name == "complement_nb":
        return ComplementNB(alpha=config.naive_bayes_alpha)

    raise ValueError(f"Clasificador no soportado: {config.classifier_name}")


def _score_clase_positiva(
    clasificador: Any,
    x: object,
    positive_value: int,
) -> np.ndarray:
    """Obtiene un puntaje continuo para la clase positiva."""

    clases = list(clasificador.classes_)
    indice_positivo = clases.index(positive_value)
    if hasattr(clasificador, "predict_proba"):
        return clasificador.predict_proba(x)[:, indice_positivo]

    decision = np.asarray(clasificador.decision_function(x), dtype=float)
    if decision.ndim == 1:
        return decision
    return decision[:, indice_positivo]


def entrenar_baseline(
    textos_entrenamiento: Iterable[str],
    y_entrenamiento: Iterable[int],
    config: BaselineConfig,
) -> BaselineModel:
    """Entrena TF-IDF + el clasificador configurado sobre entrenamiento."""

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
    score = _score_clase_positiva(
        modelo.classifier,
        x,
        modelo.config.positive_value,
    )

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
