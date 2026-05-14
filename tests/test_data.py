"""
Este módulo prueba carga y preparación tabular de publicaciones.
"""

from __future__ import annotations

import pandas as pd
import pytest

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.data import (
    cargar_publicaciones_csv,
    columnas_requeridas,
    normalizar_etiquetas,
    validar_columnas,
)


def test_columnas_requeridas_incluye_etiqueta_si_se_solicita() -> None:
    """
    Verifica las columnas mínimas con y sin etiqueta obligatoria.
    """

    config = BaselineConfig()

    assert columnas_requeridas(config, require_label=True) == [
        "user_id",
        "text_id",
        "title",
        "text",
        "is_suicide",
    ]
    assert columnas_requeridas(config, require_label=False) == [
        "user_id",
        "text_id",
        "title",
        "text",
    ]


def test_validar_columnas_falla_si_falta_requerida() -> None:
    """
    Verifica que falten columnas requeridas produzca un error claro.
    """

    config = BaselineConfig()
    frame = pd.DataFrame(
        {
            "user_id": ["u1"],
            "text_id": ["t1"],
            "title": ["titulo"],
            "is_suicide": ["yes"],
        }
    )

    with pytest.raises(ValueError, match="text"):
        validar_columnas(frame, config, require_label=True)


def test_normalizar_etiquetas_convierte_yes_no_a_binario() -> None:
    """
    Verifica que las etiquetas textuales se conviertan a valores binarios.
    """

    config = BaselineConfig()
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "text_id": ["t1", "t2"],
            "title": ["a", "b"],
            "text": ["c", "d"],
            "is_suicide": [" yes ", "No"],
        }
    )

    salida = normalizar_etiquetas(frame, config)

    assert salida["y"].tolist() == [1, 0]


def test_cargar_publicaciones_rellena_title_text_vacios(tmp_path) -> None:
    """
    Verifica que title y text faltantes se rellenen con cadenas vacías.
    """

    config = BaselineConfig()
    path = tmp_path / "publicaciones.csv"
    pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "text_id": ["t1", "t2"],
            "title": [None, "Titulo"],
            "text": ["Texto", None],
            "is_suicide": ["yes", "no"],
        }
    ).to_csv(path, index=False)

    salida = cargar_publicaciones_csv(path, config, require_label=True)

    assert salida["title"].tolist() == ["", "Titulo"]
    assert salida["text"].tolist() == ["Texto", ""]
    assert salida["y"].tolist() == [1, 0]
