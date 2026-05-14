"""
Este módulo prueba el particionado sin fuga por usuario.
"""

from __future__ import annotations

import pandas as pd

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.splitting import separar_por_usuario


def test_separar_por_usuario_no_comparte_user_id() -> None:
    """
    Verifica que ningún user_id aparezca en entrenamiento y validación.
    """

    config = BaselineConfig(validation_size=0.4, random_state=7)
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3", "u4", "u5"],
            "text_id": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "title": ["a", "b", "c", "d", "e", "f"],
            "text": ["a", "b", "c", "d", "e", "f"],
            "y": [0, 0, 1, 0, 1, 1],
        }
    )

    entrenamiento, validacion = separar_por_usuario(frame, config)

    usuarios_train = set(entrenamiento["user_id"])
    usuarios_validacion = set(validacion["user_id"])
    assert usuarios_train.isdisjoint(usuarios_validacion)


def test_separar_por_usuario_asigna_columna_split() -> None:
    """
    Verifica que el split quede marcado en ambos subconjuntos.
    """

    config = BaselineConfig(validation_size=0.4, random_state=7)
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3", "u4", "u5"],
            "text_id": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "title": ["a", "b", "c", "d", "e", "f"],
            "text": ["a", "b", "c", "d", "e", "f"],
            "y": [0, 0, 1, 0, 1, 1],
        }
    )

    entrenamiento, validacion = separar_por_usuario(frame, config)

    assert set(entrenamiento["split"]) == {"entrenamiento"}
    assert set(validacion["split"]) == {"validacion"}
