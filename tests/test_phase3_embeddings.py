"""
Pruebas del flujo de embeddings de Fase 3.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.phase3_embeddings import (
    construir_clasificador_embeddings,
    construir_predicciones_embeddings,
    entrenar_clasificador_embeddings_por_nombre,
    generar_embeddings,
    listar_clasificadores_embeddings,
    obtener_score_clase_positiva,
)


class FakeEmbeddingClient:
    """
    Cliente determinista para probar caché de embeddings.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, model: str, text: str) -> list[float]:
        self.calls.append(f"{model}:{text}")
        return [float(len(text)), 1.0]


def test_generar_embeddings_usa_y_persiste_cache(tmp_path) -> None:
    """
    Verifica generación y reutilización de caché por text_id.
    """

    config = BaselineConfig()
    frame = pd.DataFrame(
        {
            "text_id": ["t1", "t2"],
            "user_id": ["u1", "u2"],
            "title": ["Risk", "Calm"],
            "text": ["help", "stable"],
        }
    )
    cache_path = tmp_path / "embeddings.json"
    client = FakeEmbeddingClient()

    embeddings = generar_embeddings(
        frame,
        config,
        client,
        "model-a",
        cache_path,
        max_chars=4,
    )
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    second = generar_embeddings(
        frame,
        config,
        client,
        "model-a",
        cache_path,
        max_chars=4,
    )

    assert embeddings.shape == (2, 2)
    assert second.shape == (2, 2)
    assert sorted(cached) == ["t1", "t2"]
    assert len(client.calls) == 2
    assert all(call.endswith(":risk") or call.endswith(":calm") for call in client.calls)


def test_construir_predicciones_embeddings_incluye_y_true() -> None:
    """
    Verifica salida tabular compatible con evaluación.
    """

    config = BaselineConfig()
    frame = pd.DataFrame(
        {
            "text_id": ["t1", "t2"],
            "user_id": ["u1", "u2"],
            "y": [1, 0],
        }
    )

    predicciones = construir_predicciones_embeddings(
        frame,
        y_pred=[1, 0],
        score=[0.8, 0.2],
        config=config,
        include_y_true=True,
    )

    assert predicciones.columns.tolist() == [
        "text_id",
        "user_id",
        "y_pred",
        "label_pred",
        "score",
        "y_true",
    ]
    assert predicciones["label_pred"].tolist() == ["yes", "no"]


def test_listar_clasificadores_embeddings_define_matriz() -> None:
    """
    Verifica la matriz de clasificadores densos para Fase 3.
    """

    assert listar_clasificadores_embeddings() == (
        "logistic_regression",
        "linear_svm",
        "sgd_logistic",
    )


def test_construir_clasificador_embeddings_rechaza_desconocido() -> None:
    """
    Verifica error claro para clasificadores no compatibles con embeddings.
    """

    with pytest.raises(ValueError, match="Clasificador de embeddings no soportado"):
        construir_clasificador_embeddings("complement_nb", random_state=42)


def test_entrenar_clasificadores_embeddings_y_scores() -> None:
    """
    Verifica entrenamiento y score continuo para los tres clasificadores.
    """

    x_train = np.array(
        [
            [0.0, 0.1],
            [0.1, 0.0],
            [2.0, 2.1],
            [2.1, 2.0],
        ]
    )
    y_train = [0, 0, 1, 1]
    x_test = np.array([[0.2, 0.1], [2.2, 2.1]])

    for classifier_name in listar_clasificadores_embeddings():
        classifier = entrenar_clasificador_embeddings_por_nombre(
            x_train,
            y_train,
            classifier_name,
            random_state=42,
        )
        y_pred = classifier.predict(x_test)
        score = obtener_score_clase_positiva(classifier, x_test, positive_value=1)

        assert y_pred.shape == (2,)
        assert score.shape == (2,)
