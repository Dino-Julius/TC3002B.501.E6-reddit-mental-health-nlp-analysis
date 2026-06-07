"""
Pruebas del flujo de embeddings de Fase 3.
"""

from __future__ import annotations

import json

import pandas as pd

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.phase3_embeddings import (
    construir_predicciones_embeddings,
    generar_embeddings,
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
