"""
Este módulo define catálogos reproducibles para experimentos.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from reddit_mental_health.config import BaselineConfig


@dataclass(frozen=True)
class FeatureConfigSpec:
    """
    Define una variante de extracción TF-IDF.
    """

    name: str
    analyzer: str
    ngram_range: tuple[int, int]
    max_features: int
    min_df: int
    max_df: float


@dataclass(frozen=True)
class ClassifierSpec:
    """
    Define un clasificador disponible para la matriz experimental.
    """

    name: str
    description: str


CLASSIFIER_SPECS: dict[str, ClassifierSpec] = {
    "logistic_regression": ClassifierSpec(
        name="logistic_regression",
        description="Regresión logística lineal con pesos balanceados.",
    ),
    "linear_svm": ClassifierSpec(
        name="linear_svm",
        description="SVM lineal con margen máximo y puntaje por decision_function.",
    ),
    "sgd_logistic": ClassifierSpec(
        name="sgd_logistic",
        description="Regresión logística entrenada con descenso estocástico.",
    ),
    "multinomial_nb": ClassifierSpec(
        name="multinomial_nb",
        description="Naive Bayes multinomial para conteos o pesos no negativos.",
    ),
    "complement_nb": ClassifierSpec(
        name="complement_nb",
        description="Naive Bayes complementario, útil con clases desbalanceadas.",
    ),
}


FEATURE_CONFIG_SPECS: dict[str, FeatureConfigSpec] = {
    "word_unigram": FeatureConfigSpec(
        name="word_unigram",
        analyzer="word",
        ngram_range=(1, 1),
        max_features=20_000,
        min_df=2,
        max_df=0.95,
    ),
    "word_unigram_bigram": FeatureConfigSpec(
        name="word_unigram_bigram",
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20_000,
        min_df=2,
        max_df=0.95,
    ),
    "word_unigram_trigram": FeatureConfigSpec(
        name="word_unigram_trigram",
        analyzer="word",
        ngram_range=(1, 3),
        max_features=30_000,
        min_df=2,
        max_df=0.95,
    ),
    "char_wb_3_5": FeatureConfigSpec(
        name="char_wb_3_5",
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=40_000,
        min_df=2,
        max_df=0.95,
    ),
}


def listar_clasificadores() -> tuple[str, ...]:
    """
    Lista los clasificadores disponibles en orden estable.
    """

    return tuple(CLASSIFIER_SPECS)


def listar_configuraciones_features() -> tuple[str, ...]:
    """
    Lista las configuraciones de características disponibles.
    """

    return tuple(FEATURE_CONFIG_SPECS)


def construir_config_experimento(
    base_config: BaselineConfig,
    classifier_name: str,
    feature_config_name: str,
) -> BaselineConfig:
    """
    Aplica una dupla clasificador/features sobre la configuración base.
    """

    if classifier_name not in CLASSIFIER_SPECS:
        disponibles = ", ".join(listar_clasificadores())
        raise ValueError(
            f"Clasificador desconocido: {classifier_name}. Disponibles: {disponibles}",
        )
    if feature_config_name not in FEATURE_CONFIG_SPECS:
        disponibles = ", ".join(listar_configuraciones_features())
        raise ValueError(
            "Configuración de features desconocida: "
            f"{feature_config_name}. Disponibles: {disponibles}",
        )

    feature_spec = FEATURE_CONFIG_SPECS[feature_config_name]
    return replace(
        base_config,
        classifier_name=classifier_name,
        feature_config_name=feature_config_name,
        analyzer=feature_spec.analyzer,
        ngram_range=feature_spec.ngram_range,
        max_features=feature_spec.max_features,
        min_df=feature_spec.min_df,
        max_df=feature_spec.max_df,
    )


__all__ = [
    "CLASSIFIER_SPECS",
    "FEATURE_CONFIG_SPECS",
    "ClassifierSpec",
    "FeatureConfigSpec",
    "construir_config_experimento",
    "listar_clasificadores",
    "listar_configuraciones_features",
]
