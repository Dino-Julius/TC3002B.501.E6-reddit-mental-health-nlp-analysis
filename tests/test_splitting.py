"""
Este módulo prueba el particionado sin fuga por usuario.
"""

from __future__ import annotations

import json

import pandas as pd

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.splitting import (
    guardar_diagnostico_split,
    resumir_calidad_split,
    separar_por_usuario,
)


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


def test_resumir_calidad_split_reporta_overlap_y_clases() -> None:
    """
    Verifica que el diagnóstico reporte overlap cero y distribución de clases.
    """

    config = BaselineConfig()
    entrenamiento = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "y": [0, 1, 1],
            "split": ["entrenamiento"] * 3,
        }
    )
    validacion = pd.DataFrame(
        {
            "user_id": ["u4", "u5"],
            "y": [0, 1],
            "split": ["validacion"] * 2,
        }
    )

    diagnostico = resumir_calidad_split(entrenamiento, validacion, config)

    assert diagnostico["user_id_overlap_count"] == 0
    assert diagnostico["train"]["rows"] == 3
    assert diagnostico["train"]["unique_users"] == 3
    assert diagnostico["train"]["class_distribution"] == {"no": 1, "yes": 2}
    assert diagnostico["train"]["class_percentage"] == {
        "no": 1 / 3,
        "yes": 2 / 3,
    }
    assert diagnostico["validation"]["class_distribution"] == {"no": 1, "yes": 1}
    assert diagnostico["validation"]["class_percentage"] == {"no": 0.5, "yes": 0.5}


def test_guardar_diagnostico_split_escribe_json(tmp_path) -> None:
    """
    Verifica que el diagnóstico de split se guarde como JSON legible.
    """

    path = tmp_path / "split_diagnostics.json"
    diagnostico = {
        "train": {"rows": 3},
        "validation": {"rows": 2},
        "user_id_overlap_count": 0,
    }

    guardar_diagnostico_split(diagnostico, path)

    assert json.loads(path.read_text(encoding="utf-8")) == diagnostico
