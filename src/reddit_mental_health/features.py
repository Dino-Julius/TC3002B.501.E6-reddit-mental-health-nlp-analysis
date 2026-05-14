"""Extracción de características TF-IDF para el baseline."""

from __future__ import annotations

from collections.abc import Iterable

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from reddit_mental_health.config import BaselineConfig


def construir_vectorizador(config: BaselineConfig) -> TfidfVectorizer:
    """Crea el vectorizador TF-IDF definido por la configuración experimental."""

    return TfidfVectorizer(
        analyzer=config.analyzer,
        lowercase=False,
        max_df=config.max_df,
        max_features=config.max_features,
        min_df=config.min_df,
        ngram_range=config.ngram_range,
    )


def ajustar_transformar_tfidf(
    textos: Iterable[str],
    config: BaselineConfig,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Ajusta el vectorizador solo con entrenamiento y regresa la matriz TF-IDF."""

    vectorizador = construir_vectorizador(config)
    matriz = vectorizador.fit_transform(textos)
    return vectorizador, matriz


def transformar_tfidf(
    vectorizador: TfidfVectorizer,
    textos: Iterable[str],
) -> csr_matrix:
    """Aplica un vectorizador ya ajustado a validación o prueba."""

    return vectorizador.transform(textos)


__all__ = [
    "ajustar_transformar_tfidf",
    "construir_vectorizador",
    "transformar_tfidf",
]
