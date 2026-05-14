"""
Este módulo particiona datos sin fuga entre usuarios.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from reddit_mental_health.config import BaselineConfig, ensure_parent_dir


def separar_por_usuario(
    frame: pd.DataFrame,
    config: BaselineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa registros evitando usuarios compartidos entre train y validación.
    """

    if config.user_column not in frame.columns:
        raise ValueError(f"No existe la columna de usuario: {config.user_column}")

    # El split por user_id evita fuga de publicaciones del mismo usuario.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.validation_size,
        random_state=config.random_state,
    )
    train_idx, validation_idx = next(
        splitter.split(frame, groups=frame[config.user_column])
    )

    entrenamiento = frame.iloc[train_idx].copy()
    validacion = frame.iloc[validation_idx].copy()
    entrenamiento[config.split_column] = "entrenamiento"
    validacion[config.split_column] = "validacion"

    usuarios_train = set(entrenamiento[config.user_column])
    usuarios_validacion = set(validacion[config.user_column])
    fuga = usuarios_train.intersection(usuarios_validacion)
    if fuga:
        raise RuntimeError("El particionado generó fuga por user_id.")

    return entrenamiento, validacion


def _resumen_split(frame: pd.DataFrame, config: BaselineConfig) -> dict[str, Any]:
    """
    Resume tamaño, usuarios y distribución de clases de un subconjunto.
    """

    total_rows = int(len(frame))
    conteos = {config.negative_label: 0, config.positive_label: 0}
    porcentajes = {config.negative_label: 0.0, config.positive_label: 0.0}

    if config.target_column in frame.columns:
        serie = frame[config.target_column]
        conteos = {
            config.negative_label: int((serie == config.negative_value).sum()),
            config.positive_label: int((serie == config.positive_value).sum()),
        }
        if total_rows:
            porcentajes = {
                etiqueta: float(valor / total_rows)
                for etiqueta, valor in conteos.items()
            }

    return {
        "rows": total_rows,
        "unique_users": int(frame[config.user_column].nunique()),
        "class_distribution": conteos,
        "class_percentage": porcentajes,
    }


def resumir_calidad_split(
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    config: BaselineConfig,
) -> dict[str, Any]:
    """
    Resume la calidad del split entre entrenamiento y validación.
    """

    usuarios_train = set(entrenamiento[config.user_column])
    usuarios_validacion = set(validacion[config.user_column])
    overlap = usuarios_train.intersection(usuarios_validacion)

    return {
        "split_column": config.split_column,
        "user_column": config.user_column,
        "target_column": config.target_column,
        "train": _resumen_split(entrenamiento, config),
        "validation": _resumen_split(validacion, config),
        "user_id_overlap_count": int(len(overlap)),
    }


def guardar_diagnostico_split(
    diagnostico: dict[str, Any],
    path: str | Path,
) -> None:
    """
    Guarda el diagnóstico del split en JSON legible.
    """

    salida = Path(path)
    ensure_parent_dir(salida)
    salida.write_text(
        json.dumps(diagnostico, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "guardar_diagnostico_split",
    "resumir_calidad_split",
    "separar_por_usuario",
]
