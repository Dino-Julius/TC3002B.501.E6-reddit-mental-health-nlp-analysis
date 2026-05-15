"""
Este módulo prueba la validación cruzada sin fuga por usuario.
"""

from __future__ import annotations

import pandas as pd

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.cross_validation import (
    FOLD_RESULT_COLUMNS,
    construir_folds_cv,
    ejecutar_cv_combinacion,
    resolver_matriz_cv,
    resumir_resultados_cv,
    seleccionar_mejor_modelo_cv,
)


def _crear_frame_cv() -> pd.DataFrame:
    """
    Crea datos sintéticos balanceados por usuario para pruebas CV.
    """

    filas = []
    for indice in range(12):
        positivo = indice % 2 == 0
        filas.append(
            {
                "user_id": f"u{indice}",
                "text_id": f"t{indice}",
                "title": "riesgo ayuda" if positivo else "calma estable",
                "text": "riesgo ayuda urgente" if positivo else "calma estable vida",
                "y": 1 if positivo else 0,
            }
        )
    return pd.DataFrame(filas)


def test_folds_cv_no_comparten_user_id() -> None:
    """
    Verifica que ningún fold comparta usuarios entre train y validación.
    """

    config = BaselineConfig()
    frame = _crear_frame_cv()

    folds = construir_folds_cv(frame, config, n_splits=3, random_state=42)

    assert len(folds) == 3
    for train_idx, validation_idx in folds:
        usuarios_train = set(frame.iloc[train_idx]["user_id"])
        usuarios_validacion = set(frame.iloc[validation_idx]["user_id"])
        assert usuarios_train.isdisjoint(usuarios_validacion)


def test_cv_combinacion_regresa_una_fila_por_fold() -> None:
    """
    Verifica que una combinación produzca una fila de métricas por fold.
    """

    config = BaselineConfig()
    frame = _crear_frame_cv()

    resultados = ejecutar_cv_combinacion(
        frame,
        config,
        classifier_name="logistic_regression",
        feature_config_name="word_unigram",
        n_splits=3,
        random_state=42,
    )

    assert len(resultados) == 3
    assert set(resultados[0]) == set(FOLD_RESULT_COLUMNS)
    assert {fila["user_overlap_count"] for fila in resultados} == {0}


def test_resumen_cv_incluye_media_y_desviacion_protocol_auc() -> None:
    """
    Verifica que el agregado CV incluya media y desviación estándar.
    """

    config = BaselineConfig()
    frame = _crear_frame_cv()
    fold_results = pd.DataFrame(
        ejecutar_cv_combinacion(
            frame,
            config,
            classifier_name="logistic_regression",
            feature_config_name="word_unigram",
            n_splits=3,
            random_state=42,
        )
    )

    resumen = resumir_resultados_cv(fold_results, n_splits=3)

    assert "mean_protocol_auc" in resumen.columns
    assert "std_protocol_auc" in resumen.columns
    assert resumen.iloc[0]["folds_completed"] == 3
    assert resumen.iloc[0]["mean_protocol_auc"] >= 0


def test_matriz_all_all_produce_veinte_combinaciones() -> None:
    """
    Verifica que all x all expanda la matriz experimental completa.
    """

    matriz = resolver_matriz_cv("all", "all")

    assert len(matriz) == 20
    assert ("logistic_regression", "char_wb_3_5") in matriz


def test_seleccion_mejor_modelo_usa_mean_protocol_auc() -> None:
    """
    Verifica que la selección priorice mean_protocol_auc.
    """

    resumen = pd.DataFrame(
        [
            {
                "classifier_name": "logistic_regression",
                "feature_config_name": "char_wb_3_5",
                "n_splits": 5,
                "mean_protocol_auc": 0.72,
                "std_protocol_auc": 0.03,
                "mean_roc_auc": 0.70,
                "mean_recall": 0.75,
                "mean_precision": 0.71,
                "mean_f1": 0.73,
                "folds_completed": 5,
            },
            {
                "classifier_name": "linear_svm",
                "feature_config_name": "word_unigram",
                "n_splits": 5,
                "mean_protocol_auc": 0.81,
                "std_protocol_auc": 0.08,
                "mean_roc_auc": 0.79,
                "mean_recall": 0.82,
                "mean_precision": 0.80,
                "mean_f1": 0.81,
                "folds_completed": 5,
            },
        ]
    )

    mejor = seleccionar_mejor_modelo_cv(resumen)

    assert mejor["selected_by"] == "mean_protocol_auc"
    assert mejor["classifier_name"] == "linear_svm"
    assert mejor["feature_config_name"] == "word_unigram"
