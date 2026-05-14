"""
Este módulo carga, valida y prepara publicaciones tabulares de Reddit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reddit_mental_health.config import BaselineConfig


def columnas_requeridas(config: BaselineConfig, require_label: bool = True) -> list[str]:
    """
    Devuelve las columnas mínimas esperadas por el diseño conceptual.
    """

    columnas = [
        config.user_column,
        config.text_id_column,
        config.title_column,
        config.text_column,
    ]
    if require_label:
        columnas.append(config.label_column)
    return columnas


def validar_columnas(
    frame: pd.DataFrame,
    config: BaselineConfig,
    require_label: bool = True,
) -> None:
    """
    Valida que el CSV contenga las columnas necesarias para el pipeline.
    """

    faltantes = [
        columna
        for columna in columnas_requeridas(config, require_label=require_label)
        if columna not in frame.columns
    ]
    if faltantes:
        raise ValueError(
            "El archivo no contiene las columnas requeridas: "
            + ", ".join(faltantes)
        )


def normalizar_etiquetas(frame: pd.DataFrame, config: BaselineConfig) -> pd.DataFrame:
    """
    Convierte las etiquetas yes/no a valores binarios reproducibles.
    """

    if config.label_column not in frame.columns:
        return frame

    etiquetas = {
        config.positive_label: config.positive_value,
        config.negative_label: config.negative_value,
        "1": config.positive_value,
        "0": config.negative_value,
        1: config.positive_value,
        0: config.negative_value,
    }
    salida = frame.copy()
    valores = salida[config.label_column].map(
        lambda valor: valor.strip().lower() if isinstance(valor, str) else valor
    )
    salida[config.target_column] = valores.map(etiquetas)

    invalidas = salida[salida[config.target_column].isna()][config.label_column].unique()
    if len(invalidas) > 0:
        raise ValueError(
            "La columna de etiquetas contiene valores no reconocidos: "
            + ", ".join(map(str, invalidas))
        )

    salida[config.target_column] = salida[config.target_column].astype(int)
    return salida


def cargar_publicaciones_csv(
    path: str | Path,
    config: BaselineConfig,
    require_label: bool = True,
) -> pd.DataFrame:
    """
    Carga un CSV de publicaciones y deja las etiquetas listas para modelar.
    """

    frame = pd.read_csv(path)
    validar_columnas(frame, config, require_label=require_label)

    # Los campos textuales vacíos se conservan como cadenas para no perder filas.
    for columna in (config.title_column, config.text_column):
        if columna in frame.columns:
            frame[columna] = frame[columna].fillna("")

    return normalizar_etiquetas(frame, config)


def construir_salida_predicciones(
    frame: pd.DataFrame,
    y_pred: list[int],
    score: list[float],
    config: BaselineConfig,
    split: str | None = None,
    include_y_true: bool = True,
) -> pd.DataFrame:
    """
    Construye la tabla de salida trazable definida en el diseño.
    """

    columnas = [config.text_id_column, config.user_column]
    salida = frame[columnas].copy()
    if include_y_true and config.target_column in frame.columns:
        salida["y_true"] = frame[config.target_column].to_numpy()

    salida["y_pred"] = y_pred
    salida["score"] = score
    salida[config.split_column] = split or frame.get(config.split_column, "evaluacion")
    return salida


__all__ = [
    "cargar_publicaciones_csv",
    "columnas_requeridas",
    "construir_salida_predicciones",
    "normalizar_etiquetas",
    "validar_columnas",
]
