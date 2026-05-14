"""
Este módulo particiona datos sin fuga entre usuarios.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from reddit_mental_health.config import BaselineConfig


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


__all__ = ["separar_por_usuario"]
