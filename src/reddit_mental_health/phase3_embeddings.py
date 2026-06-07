"""
Flujo de Fase 3 basado en embeddings locales de Ollama.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from reddit_mental_health.config import BaselineConfig, ensure_parent_dir
from reddit_mental_health.evaluation import calcular_metricas, guardar_metricas
from reddit_mental_health.preprocessing import preprocesar_publicaciones


class EmbeddingClient(Protocol):
    """
    Contrato mínimo para clientes de embeddings.
    """

    def embed(self, model: str, text: str) -> list[float]:
        """
        Genera un embedding para un texto.
        """


def _cargar_cache(path: Path) -> dict[str, list[float]]:
    """
    Carga un caché JSON de embeddings por text_id.
    """

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"El caché de embeddings no es un objeto JSON: {path}")
    cache: dict[str, list[float]] = {}
    for key, value in payload.items():
        if not isinstance(value, list) or not value:
            raise ValueError(f"Embedding inválido en caché para text_id={key}.")
        cache[str(key)] = [float(item) for item in value]
    return cache


def _guardar_cache(cache: dict[str, list[float]], path: Path) -> None:
    """
    Guarda el caché de embeddings como JSON reproducible.
    """

    ensure_parent_dir(path)
    path.write_text(json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8")


def generar_embeddings(
    frame: pd.DataFrame,
    config: BaselineConfig,
    client: EmbeddingClient,
    model_name: str,
    cache_path: Path,
    max_chars: int,
) -> np.ndarray:
    """
    Genera embeddings para un frame usando caché por text_id.
    """

    textos = preprocesar_publicaciones(frame, config)
    cache = _cargar_cache(cache_path)
    embeddings: list[list[float]] = []
    cache_changed = False

    for text_id, texto in zip(frame[config.text_id_column], textos, strict=True):
        key = str(text_id)
        embedding = cache.get(key)
        if embedding is None:
            embedding = client.embed(model_name, texto[:max_chars])
            cache[key] = embedding
            cache_changed = True
        embeddings.append(embedding)

    if cache_changed:
        _guardar_cache(cache, cache_path)

    if not embeddings:
        raise ValueError("No hay publicaciones para generar embeddings.")
    dimensions = {len(embedding) for embedding in embeddings}
    if len(dimensions) != 1:
        raise ValueError("Los embeddings tienen dimensiones inconsistentes.")
    return np.asarray(embeddings, dtype=float)


def entrenar_clasificador_embeddings(
    x_train: np.ndarray,
    y_train: Sequence[int],
    random_state: int,
) -> LogisticRegression:
    """
    Entrena una regresión logística sobre embeddings densos.
    """

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_state,
    )
    classifier.fit(x_train, np.asarray(y_train, dtype=int))
    return classifier


def construir_predicciones_embeddings(
    test_data: pd.DataFrame,
    y_pred: Sequence[int],
    score: Sequence[float],
    config: BaselineConfig,
    include_y_true: bool,
) -> pd.DataFrame:
    """
    Construye salida trazable para el método de embeddings.
    """

    salida = test_data[[config.text_id_column, config.user_column]].copy()
    salida["y_pred"] = list(map(int, y_pred))
    salida["label_pred"] = salida["y_pred"].map({0: "no", 1: "yes"})
    salida["score"] = list(map(float, score))
    if include_y_true and config.target_column in test_data.columns:
        salida["y_true"] = test_data[config.target_column].to_numpy()
    return salida


def evaluar_y_guardar_embeddings(
    predicciones: pd.DataFrame,
    config: BaselineConfig,
    metrics_path: Path,
) -> dict[str, object]:
    """
    Calcula y persiste métricas para predicciones etiquetadas.
    """

    metricas = calcular_metricas(
        predicciones["y_true"],
        predicciones["y_pred"],
        predicciones["score"],
        config,
    )
    guardar_metricas(metricas, metrics_path)
    return metricas


__all__ = [
    "EmbeddingClient",
    "construir_predicciones_embeddings",
    "entrenar_clasificador_embeddings",
    "evaluar_y_guardar_embeddings",
    "generar_embeddings",
]
