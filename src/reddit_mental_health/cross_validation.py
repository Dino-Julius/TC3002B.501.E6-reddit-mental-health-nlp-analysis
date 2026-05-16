"""
Este módulo ejecuta validación cruzada estratificada sin fuga por usuario.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.evaluation import calcular_metricas
from reddit_mental_health.experiments import (
    construir_config_experimento,
    listar_clasificadores,
    listar_configuraciones_features,
)
from reddit_mental_health.model import entrenar_baseline, predecir_baseline
from reddit_mental_health.preprocessing import preprocesar_publicaciones


FOLD_RESULT_COLUMNS = [
    "classifier_name",
    "feature_config_name",
    "fold",
    "train_rows",
    "validation_rows",
    "train_users",
    "validation_users",
    "user_overlap_count",
    "protocol_auc",
    "roc_auc",
    "recall",
    "precision",
    "f1",
    "true_positive_rate",
    "false_positive_rate",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
]

SUMMARY_CV_COLUMNS = [
    "classifier_name",
    "feature_config_name",
    "n_splits",
    "mean_protocol_auc",
    "std_protocol_auc",
    "mean_roc_auc",
    "std_roc_auc",
    "mean_recall",
    "std_recall",
    "mean_precision",
    "std_precision",
    "mean_f1",
    "std_f1",
    "mean_true_positive_rate",
    "std_true_positive_rate",
    "mean_false_positive_rate",
    "std_false_positive_rate",
    "folds_completed",
]

METRIC_COLUMNS = [
    "protocol_auc",
    "roc_auc",
    "recall",
    "precision",
    "f1",
    "true_positive_rate",
    "false_positive_rate",
]

BEST_MODEL_NOTE = (
    "La selección se hizo exclusivamente con data_train.csv; "
    "data_test_fold1.csv no se usó para selección de modelo."
)


def _resolver_valores(nombre: str, disponibles: tuple[str, ...], etiqueta: str) -> tuple[str, ...]:
    """
    Expande all o valida un nombre concreto del catálogo experimental.
    """

    if nombre == "all":
        return disponibles
    if nombre not in disponibles:
        opciones = ", ".join((*disponibles, "all"))
        raise ValueError(f"{etiqueta} desconocido: {nombre}. Disponibles: {opciones}")
    return (nombre,)


def resolver_matriz_cv(
    classifier_name: str,
    feature_config_name: str,
) -> list[tuple[str, str]]:
    """
    Construye las combinaciones de validación cruzada solicitadas.
    """

    clasificadores = _resolver_valores(
        classifier_name,
        listar_clasificadores(),
        "Clasificador",
    )
    configuraciones = _resolver_valores(
        feature_config_name,
        listar_configuraciones_features(),
        "Configuración de features",
    )
    return list(itertools.product(clasificadores, configuraciones))


def construir_folds_cv(
    frame: pd.DataFrame,
    config: BaselineConfig,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Construye folds StratifiedGroupKFold y valida ausencia de fuga por user_id.
    """

    for columna in (config.user_column, config.target_column):
        if columna not in frame.columns:
            raise ValueError(f"No existe la columna requerida para CV: {columna}")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds = []
    y = frame[config.target_column]
    grupos = frame[config.user_column]

    for train_idx, validation_idx in splitter.split(frame, y=y, groups=grupos):
        usuarios_train = set(frame.iloc[train_idx][config.user_column])
        usuarios_validacion = set(frame.iloc[validation_idx][config.user_column])
        if usuarios_train.intersection(usuarios_validacion):
            raise RuntimeError("StratifiedGroupKFold generó fuga por user_id.")
        folds.append((train_idx, validation_idx))

    return folds


def _aplanar_metricas(metricas: dict[str, Any]) -> dict[str, Any]:
    """
    Convierte las métricas anidadas al esquema tabular de folds.
    """

    matriz = metricas.get("confusion_matrix", {})
    return {
        "protocol_auc": metricas.get("protocol_auc"),
        "roc_auc": metricas.get("roc_auc"),
        "recall": metricas.get("recall"),
        "precision": metricas.get("precision"),
        "f1": metricas.get("f1"),
        "true_positive_rate": metricas.get("true_positive_rate"),
        "false_positive_rate": metricas.get("false_positive_rate"),
        "true_negative": matriz.get("true_negative"),
        "false_positive": matriz.get("false_positive"),
        "false_negative": matriz.get("false_negative"),
        "true_positive": matriz.get("true_positive"),
    }


def ejecutar_cv_combinacion(
    frame: pd.DataFrame,
    base_config: BaselineConfig,
    classifier_name: str,
    feature_config_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Evalúa una dupla clasificador/features en todos los folds de CV.
    """

    config = construir_config_experimento(
        replace(base_config, random_state=random_state),
        classifier_name,
        feature_config_name,
    )
    folds = construir_folds_cv(
        frame,
        config,
        n_splits=n_splits,
        random_state=random_state,
    )

    resultados = []
    for fold_number, (train_idx, validation_idx) in enumerate(folds, start=1):
        entrenamiento = frame.iloc[train_idx].copy()
        validacion = frame.iloc[validation_idx].copy()
        entrenamiento[config.split_column] = "entrenamiento"
        validacion[config.split_column] = "validacion"

        textos_entrenamiento = preprocesar_publicaciones(entrenamiento, config)
        textos_validacion = preprocesar_publicaciones(validacion, config)
        modelo = entrenar_baseline(
            textos_entrenamiento,
            entrenamiento[config.target_column],
            config,
        )
        y_pred, score = predecir_baseline(modelo, textos_validacion)
        metricas = calcular_metricas(
            validacion[config.target_column],
            y_pred,
            score,
            config,
        )

        usuarios_train = set(entrenamiento[config.user_column])
        usuarios_validacion = set(validacion[config.user_column])
        fila = {
            "classifier_name": classifier_name,
            "feature_config_name": feature_config_name,
            "fold": fold_number,
            "train_rows": int(len(entrenamiento)),
            "validation_rows": int(len(validacion)),
            "train_users": int(entrenamiento[config.user_column].nunique()),
            "validation_users": int(validacion[config.user_column].nunique()),
            "user_overlap_count": int(len(usuarios_train.intersection(usuarios_validacion))),
        }
        fila.update(_aplanar_metricas(metricas))
        resultados.append(fila)

    return resultados


def ejecutar_matriz_cv(
    frame: pd.DataFrame,
    base_config: BaselineConfig,
    matriz: list[tuple[str, str]],
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Ejecuta validación cruzada para una matriz de combinaciones.
    """

    registros = []
    for classifier_name, feature_config_name in matriz:
        registros.extend(
            ejecutar_cv_combinacion(
                frame,
                base_config,
                classifier_name,
                feature_config_name,
                n_splits=n_splits,
                random_state=random_state,
            )
        )
    return pd.DataFrame(registros, columns=FOLD_RESULT_COLUMNS)


def _mean_std(serie: pd.Series) -> tuple[float | None, float | None]:
    """
    Calcula media y desviación estándar ignorando valores no numéricos.
    """

    valores = pd.to_numeric(serie, errors="coerce").dropna()
    if valores.empty:
        return None, None
    media = float(valores.mean())
    desviacion = float(valores.std(ddof=1)) if len(valores) > 1 else 0.0
    return media, desviacion


def resumir_resultados_cv(
    fold_results: pd.DataFrame,
    n_splits: int,
) -> pd.DataFrame:
    """
    Agrega resultados por combinación usando medias y desviaciones por fold.
    """

    registros = []
    grupos = fold_results.groupby(
        ["classifier_name", "feature_config_name"],
        sort=True,
        dropna=False,
    )
    for (classifier_name, feature_config_name), grupo in grupos:
        fila: dict[str, Any] = {
            "classifier_name": classifier_name,
            "feature_config_name": feature_config_name,
            "n_splits": int(n_splits),
            "folds_completed": int(len(grupo)),
        }
        for metrica in METRIC_COLUMNS:
            media, desviacion = _mean_std(grupo[metrica])
            fila[f"mean_{metrica}"] = media
            fila[f"std_{metrica}"] = desviacion
        registros.append(fila)

    resumen = pd.DataFrame(registros)
    if resumen.empty:
        return pd.DataFrame(columns=SUMMARY_CV_COLUMNS)
    return resumen[SUMMARY_CV_COLUMNS].sort_values(
        ["mean_protocol_auc", "std_protocol_auc", "classifier_name", "feature_config_name"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def seleccionar_mejor_modelo_cv(summary_cv: pd.DataFrame) -> dict[str, Any]:
    """
    Selecciona la mejor combinación por mean_protocol_auc.
    """

    if summary_cv.empty:
        raise ValueError("No hay resultados CV para seleccionar modelo.")

    resumen = summary_cv.copy()
    resumen["mean_protocol_auc"] = pd.to_numeric(
        resumen["mean_protocol_auc"],
        errors="coerce",
    )
    resumen = resumen.dropna(subset=["mean_protocol_auc"])
    if resumen.empty:
        raise ValueError("No hay mean_protocol_auc válido para seleccionar modelo.")

    resumen = resumen.sort_values(
        ["mean_protocol_auc", "std_protocol_auc", "classifier_name", "feature_config_name"],
        ascending=[False, True, True, True],
        na_position="last",
    )
    mejor = resumen.iloc[0]
    return {
        "classifier_name": str(mejor["classifier_name"]),
        "feature_config_name": str(mejor["feature_config_name"]),
        "selected_by": "mean_protocol_auc",
        "mean_protocol_auc": float(mejor["mean_protocol_auc"]),
        "std_protocol_auc": None
        if pd.isna(mejor["std_protocol_auc"])
        else float(mejor["std_protocol_auc"]),
        "mean_roc_auc": None
        if pd.isna(mejor["mean_roc_auc"])
        else float(mejor["mean_roc_auc"]),
        "mean_recall": None
        if pd.isna(mejor["mean_recall"])
        else float(mejor["mean_recall"]),
        "mean_precision": None
        if pd.isna(mejor["mean_precision"])
        else float(mejor["mean_precision"]),
        "mean_f1": None if pd.isna(mejor["mean_f1"]) else float(mejor["mean_f1"]),
        "note": BEST_MODEL_NOTE,
    }


__all__ = [
    "BEST_MODEL_NOTE",
    "FOLD_RESULT_COLUMNS",
    "SUMMARY_CV_COLUMNS",
    "construir_folds_cv",
    "ejecutar_cv_combinacion",
    "ejecutar_matriz_cv",
    "resolver_matriz_cv",
    "resumir_resultados_cv",
    "seleccionar_mejor_modelo_cv",
]
