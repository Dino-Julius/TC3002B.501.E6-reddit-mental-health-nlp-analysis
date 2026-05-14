"""
Este módulo prueba el preprocesamiento textual conservador.
"""

from __future__ import annotations

import pandas as pd

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.preprocessing import (
    combinar_title_text,
    normalizar_texto,
    preprocesar_publicaciones,
)


def test_combinar_title_text_une_titulo_y_cuerpo() -> None:
    """
    Verifica que title y text se unan con salto de línea.
    """

    config = BaselineConfig()
    frame = pd.DataFrame({"title": ["Titulo"], "text": ["Cuerpo"]})

    combinado = combinar_title_text(frame, config)

    assert combinado.tolist() == ["Titulo\nCuerpo"]


def test_normalizar_texto_minimiza_sin_borrar_negaciones() -> None:
    """
    Verifica minúsculas, URL tokenizada, espacios y negaciones preservadas.
    """

    texto = "NO visitar https://example.com   not\tnever\n"

    salida = normalizar_texto(texto)

    assert salida == "no visitar url_token not never"
    assert "no" in salida.split()
    assert "not" in salida.split()
    assert "never" in salida.split()


def test_preprocesar_publicaciones_normaliza_texto_combinado() -> None:
    """
    Verifica que las publicaciones se combinen y normalicen.
    """

    config = BaselineConfig()
    frame = pd.DataFrame(
        {
            "title": ["AYUDA"],
            "text": ["Texto   con   espacios y www.example.com"],
        }
    )

    salida = preprocesar_publicaciones(frame, config)

    assert salida.tolist() == ["ayuda texto con espacios y url_token"]
