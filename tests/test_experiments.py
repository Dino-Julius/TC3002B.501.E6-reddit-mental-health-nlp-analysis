"""Pruebas del catálogo de experimentos baseline."""

from __future__ import annotations

import pytest

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.experiments import (
    construir_config_experimento,
    listar_clasificadores,
    listar_configuraciones_features,
)


def test_catalogo_incluye_matriz_experimental_esperada() -> None:
    """Expone cinco clasificadores y cuatro configuraciones de features."""

    assert listar_clasificadores() == (
        "logistic_regression",
        "linear_svm",
        "sgd_logistic",
        "multinomial_nb",
        "complement_nb",
    )
    assert listar_configuraciones_features() == (
        "word_unigram",
        "word_unigram_bigram",
        "word_unigram_trigram",
        "char_wb_3_5",
    )


def test_construir_config_experimento_aplica_features_y_clasificador() -> None:
    """Crea una configuración concreta sin mutar la base."""

    base_config = BaselineConfig()
    config = construir_config_experimento(
        base_config,
        classifier_name="linear_svm",
        feature_config_name="char_wb_3_5",
    )

    assert base_config.classifier_name == "logistic_regression"
    assert config.classifier_name == "linear_svm"
    assert config.feature_config_name == "char_wb_3_5"
    assert config.analyzer == "char_wb"
    assert config.ngram_range == (3, 5)


def test_construir_config_experimento_rechaza_nombres_invalidos() -> None:
    """Falla claramente cuando el usuario pide una opción inexistente."""

    with pytest.raises(ValueError, match="Clasificador desconocido"):
        construir_config_experimento(
            BaselineConfig(),
            classifier_name="random_forest",
            feature_config_name="word_unigram",
        )

    with pytest.raises(ValueError, match="Configuración de features desconocida"):
        construir_config_experimento(
            BaselineConfig(),
            classifier_name="logistic_regression",
            feature_config_name="bert",
        )
