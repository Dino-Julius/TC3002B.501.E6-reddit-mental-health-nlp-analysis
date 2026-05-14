"""
Este módulo construye reportes básicos de interpretabilidad experimental.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reddit_mental_health.config import BaselineConfig, ensure_parent_dir
from reddit_mental_health.model import BaselineModel


NOTA_ETICA = (
    "Evidencia exploratoria del modelo; no debe interpretarse como diagnóstico "
    "clínico ni como explicación validada de riesgo individual."
)


def extraer_pesos_caracteristicas(
    modelo: BaselineModel,
    top_n: int = 25,
) -> dict[str, list[dict[str, float | str]]]:
    """
    Obtiene n-gramas asociados a cada clase a partir de señales del modelo.
    """

    nombres = modelo.vectorizer.get_feature_names_out()
    clasificador = modelo.classifier
    if hasattr(clasificador, "coef_"):
        pesos = clasificador.coef_[0]
    elif hasattr(clasificador, "feature_log_prob_"):
        clases = list(clasificador.classes_)
        indice_positivo = clases.index(modelo.config.positive_value)
        indice_negativo = clases.index(modelo.config.negative_value)
        pesos = (
            clasificador.feature_log_prob_[indice_positivo]
            - clasificador.feature_log_prob_[indice_negativo]
        )
    else:
        raise ValueError(
            "El clasificador no expone coeficientes ni probabilidades por feature.",
        )

    indices_positivos = np.argsort(pesos)[-top_n:][::-1]
    indices_negativos = np.argsort(pesos)[:top_n]

    def empaquetar(indices: np.ndarray) -> list[dict[str, float | str]]:
        return [
            {"termino": str(nombres[indice]), "peso": float(pesos[indice])}
            for indice in indices
        ]

    return {
        "terminos_asociados_a_yes": empaquetar(indices_positivos),
        "terminos_asociados_a_no": empaquetar(indices_negativos),
    }


def _recortar_fragmento(texto: object, limite: int = 320) -> str:
    """
    Recorta texto para reportes sin guardar publicaciones completas.
    """

    fragmento = "" if texto is None or pd.isna(texto) else str(texto)
    fragmento = " ".join(fragmento.split())
    if len(fragmento) <= limite:
        return fragmento
    return fragmento[: limite - 3].rstrip() + "..."


def _seleccionar_ejemplos(
    frame: pd.DataFrame,
    condicion: pd.Series,
    config: BaselineConfig,
    max_examples: int,
) -> list[dict[str, object]]:
    """
    Selecciona casos trazables para análisis cualitativo posterior.
    """

    columnas = [
        config.text_id_column,
        config.user_column,
        "y_true",
        "y_pred",
        "score",
        "fragmento",
    ]
    ejemplos = (
        frame.loc[condicion, columnas]
        .sort_values("score", ascending=False)
        .head(max_examples)
    )
    return ejemplos.to_dict(orient="records")


def construir_reporte_interpretabilidad(
    modelo: BaselineModel,
    predicciones: pd.DataFrame,
    textos_crudos: pd.Series,
    config: BaselineConfig,
    top_n: int = 25,
    max_examples: int = 5,
) -> dict[str, object]:
    """
    Construye un reporte inicial con pesos y casos de error revisables.
    """

    reporte: dict[str, object] = {
        "nota_etica": NOTA_ETICA,
        "pesos_caracteristicas": extraer_pesos_caracteristicas(modelo, top_n=top_n),
    }

    ejemplos = predicciones.copy()
    ejemplos["fragmento"] = textos_crudos.map(_recortar_fragmento).to_numpy()

    if {"y_true", "y_pred"}.issubset(ejemplos.columns):
        falsos_negativos = (ejemplos["y_true"] == config.positive_value) & (
            ejemplos["y_pred"] == config.negative_value
        )
        falsos_positivos = (ejemplos["y_true"] == config.negative_value) & (
            ejemplos["y_pred"] == config.positive_value
        )
        reporte["falsos_negativos_prioritarios"] = _seleccionar_ejemplos(
            ejemplos,
            falsos_negativos,
            config,
            max_examples,
        )
        reporte["falsos_positivos"] = _seleccionar_ejemplos(
            ejemplos,
            falsos_positivos,
            config,
            max_examples,
        )
    else:
        reporte["predicciones_alta_confianza"] = (
            ejemplos.sort_values("score", ascending=False)
            .head(max_examples)
            .to_dict(orient="records")
        )

    return reporte


def guardar_reporte_interpretabilidad(
    reporte: dict[str, object],
    path: str | Path,
) -> None:
    """
    Guarda el reporte interpretativo en JSON.
    """

    salida = Path(path)
    ensure_parent_dir(salida)
    salida.write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "construir_reporte_interpretabilidad",
    "extraer_pesos_caracteristicas",
    "guardar_reporte_interpretabilidad",
]
