"""
Flujo de Fase 3 basado en clasificación zero-shot con LLM local.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import pandas as pd

from reddit_mental_health.config import BaselineConfig, ensure_parent_dir
from reddit_mental_health.evaluation import calcular_metricas, guardar_metricas
from reddit_mental_health.ollama import (
    construir_prompt_zero_shot,
    label_to_prediction,
    parse_llm_classification,
)
from reddit_mental_health.preprocessing import preprocesar_publicaciones


class GenerativeClient(Protocol):
    """
    Contrato mínimo para clientes LLM generativos.
    """

    def generate_json(self, model: str, prompt: str) -> str:
        """
        Genera texto JSON para un prompt.
        """


def _clasificar_con_reintento(
    client: GenerativeClient,
    model_name: str,
    texto: str,
    max_attempts: int,
) -> tuple[dict[str, object], str | None]:
    """
    Clasifica un texto y reintenta si el JSON no cumple el contrato.
    """

    prompt = construir_prompt_zero_shot(texto)
    last_error: str | None = None
    for _attempt in range(max_attempts):
        raw_response = client.generate_json(model_name, prompt)
        try:
            classification = parse_llm_classification(raw_response)
            return (
                {
                    "label": classification.label,
                    "score": classification.score,
                    "evidence": classification.evidence,
                    "justification": classification.justification,
                    "raw_response": raw_response,
                },
                None,
            )
        except ValueError as exc:
            last_error = str(exc)
    return (
        {
            "label": None,
            "score": None,
            "evidence": [],
            "justification": "",
            "raw_response": raw_response if "raw_response" in locals() else "",
        },
        last_error or "No se pudo clasificar la publicación.",
    )


def clasificar_publicaciones_llm(
    test_data: pd.DataFrame,
    config: BaselineConfig,
    client: GenerativeClient,
    model_name: str,
    raw_responses_path: Path,
    max_attempts: int = 2,
) -> pd.DataFrame:
    """
    Clasifica publicaciones con LLM y guarda respuestas crudas en JSONL.
    """

    textos = preprocesar_publicaciones(test_data, config)
    ensure_parent_dir(raw_responses_path)
    rows: list[dict[str, object]] = []

    with raw_responses_path.open("w", encoding="utf-8") as output:
        for record, texto in zip(test_data.to_dict("records"), textos, strict=True):
            result, error = _clasificar_con_reintento(
                client,
                model_name,
                texto,
                max_attempts=max_attempts,
            )
            label = result["label"]
            y_pred = label_to_prediction(str(label)) if isinstance(label, str) else None
            score = result["score"] if isinstance(result["score"], float | int) else None
            row = {
                config.text_id_column: record[config.text_id_column],
                config.user_column: record[config.user_column],
                "y_pred": y_pred,
                "label_pred": label,
                "score": score,
                "evidence": result["evidence"],
                "justification": result["justification"],
                "error": error,
            }
            if config.target_column in record:
                row["y_true"] = record[config.target_column]
            rows.append(row)

            raw_payload = {
                config.text_id_column: record[config.text_id_column],
                config.user_column: record[config.user_column],
                "model": model_name,
                "raw_response": result["raw_response"],
                "error": error,
            }
            output.write(json.dumps(raw_payload, ensure_ascii=False) + "\n")

    return pd.DataFrame(rows)


def evaluar_y_guardar_llm(
    predicciones: pd.DataFrame,
    config: BaselineConfig,
    metrics_path: Path,
) -> dict[str, object]:
    """
    Calcula métricas LLM si todas las predicciones son válidas.
    """

    if predicciones["error"].notna().any():
        errores = int(predicciones["error"].notna().sum())
        raise ValueError(
            f"No se pueden calcular métricas: {errores} respuestas LLM fallaron."
        )
    metricas = calcular_metricas(
        predicciones["y_true"],
        predicciones["y_pred"].astype(int),
        predicciones["score"].astype(float),
        config,
    )
    guardar_metricas(metricas, metrics_path)
    return metricas


def serializar_evidence_para_csv(values: Sequence[object]) -> list[str]:
    """
    Serializa listas de evidencia para guardarlas en CSV sin perder estructura.
    """

    return [json.dumps(value, ensure_ascii=False) for value in values]


__all__ = [
    "GenerativeClient",
    "clasificar_publicaciones_llm",
    "evaluar_y_guardar_llm",
    "serializar_evidence_para_csv",
]
