"""
Este módulo prueba la extracción de características TF-IDF.
"""

from __future__ import annotations

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.features import (
    ajustar_transformar_tfidf,
    transformar_tfidf,
)


def test_ajustar_transformar_tfidf_entrena_vectorizador() -> None:
    """
    Verifica que el vectorizador pueda ajustarse y transformar entrenamiento.
    """

    config = BaselineConfig(min_df=1, max_df=1.0, ngram_range=(1, 1))

    vectorizador, matriz = ajustar_transformar_tfidf(
        ["texto de ayuda", "texto estable"],
        config,
    )

    assert matriz.shape[0] == 2
    assert len(vectorizador.get_feature_names_out()) > 0


def test_transformar_tfidf_usa_vectorizador_ajustado() -> None:
    """
    Verifica que validación use un vectorizador ya ajustado.
    """

    config = BaselineConfig(min_df=1, max_df=1.0, ngram_range=(1, 1))
    vectorizador, matriz_entrenamiento = ajustar_transformar_tfidf(
        ["ayuda urgente", "vida estable"],
        config,
    )

    matriz_validacion = transformar_tfidf(vectorizador, ["ayuda estable"])

    assert matriz_entrenamiento.shape[1] == matriz_validacion.shape[1]
    assert matriz_validacion.shape[0] == 1
