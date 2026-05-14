"""Pruebas de construcción y predicción con clasificadores baseline."""

from __future__ import annotations

import pytest

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.experiments import listar_clasificadores
from reddit_mental_health.model import (
    construir_clasificador,
    entrenar_baseline,
    predecir_baseline,
)


@pytest.mark.parametrize("classifier_name", listar_clasificadores())
def test_clasificadores_baseline_entrenan_y_generan_scores(
    classifier_name: str,
) -> None:
    """Cada clasificador del catálogo produce predicciones y puntajes."""

    config = BaselineConfig(
        classifier_name=classifier_name,
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 1),
        max_features=100,
        random_state=42,
    )
    textos = [
        "ayuda riesgo urgente",
        "vida tranquila estable",
        "riesgo alto ayuda",
        "tranquilo hoy estable",
        "urgente crisis riesgo",
        "vida calma tranquila",
    ]
    etiquetas = [1, 0, 1, 0, 1, 0]

    modelo = entrenar_baseline(textos, etiquetas, config)
    y_pred, score = predecir_baseline(modelo, textos)

    assert len(y_pred) == len(textos)
    assert len(score) == len(textos)
    assert set(y_pred).issubset({0, 1})
    assert all(isinstance(valor, float) for valor in score)


def test_construir_clasificador_rechaza_nombre_no_soportado() -> None:
    """Reporta nombres inválidos antes de entrenar."""

    config = BaselineConfig(classifier_name="random_forest")

    with pytest.raises(ValueError, match="Clasificador no soportado"):
        construir_clasificador(config)
