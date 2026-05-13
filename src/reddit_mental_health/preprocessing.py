"""Preprocesamiento textual conservador para publicaciones de Reddit."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from reddit_mental_health.config import BaselineConfig


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
WHITESPACE_RE = re.compile(r"\s+")


def normalizar_texto(texto: object) -> str:
    """Normaliza texto sin borrar señales lingüísticas potencialmente útiles."""

    if texto is None or pd.isna(texto):
        return ""

    salida = str(texto).lower()
    salida = URL_RE.sub(" url_token ", salida)
    salida = CONTROL_RE.sub(" ", salida)
    salida = WHITESPACE_RE.sub(" ", salida)
    return salida.strip()


def combinar_title_text(frame: pd.DataFrame, config: BaselineConfig) -> pd.Series:
    """Concatena `title` y `text`, que son la entrada textual del baseline."""

    titulo = frame[config.title_column].fillna("").astype(str)
    cuerpo = frame[config.text_column].fillna("").astype(str)
    return titulo + "\n" + cuerpo


def preprocesar_publicaciones(
    frame: pd.DataFrame,
    config: BaselineConfig,
) -> pd.Series:
    """Aplica la normalización textual prevista por el diseño conceptual."""

    return combinar_title_text(frame, config).map(normalizar_texto)


def preprocesar_textos(textos: Iterable[object]) -> list[str]:
    """Normaliza una secuencia de textos ya combinados."""

    return [normalizar_texto(texto) for texto in textos]


__all__ = [
    "combinar_title_text",
    "normalizar_texto",
    "preprocesar_publicaciones",
    "preprocesar_textos",
]
