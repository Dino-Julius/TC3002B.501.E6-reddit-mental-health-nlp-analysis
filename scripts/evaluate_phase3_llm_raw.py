"""
Recalcula predicciones y métricas LLM desde respuestas JSONL ya generadas.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reddit_mental_health.config import BaselineConfig, ensure_parent_dir
from reddit_mental_health.data import cargar_publicaciones_csv
from reddit_mental_health.ollama import label_to_prediction, parse_llm_classification
from reddit_mental_health.phase3_config import Phase3Config
from reddit_mental_health.phase3_llm import evaluar_y_guardar_llm, serializar_evidence_para_csv


def parse_args() -> argparse.Namespace:
    """
    Lee argumentos CLI para reparar resultados LLM desde JSONL.
    """

    phase3 = Phase3Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-input", type=Path, default=phase3.test_input)
    parser.add_argument("--raw-responses", type=Path, default=phase3.llm_raw_responses_path)
    parser.add_argument("--predictions-out", type=Path, default=phase3.llm_predictions_path)
    parser.add_argument("--metrics-out", type=Path, default=phase3.llm_metrics_path)
    return parser.parse_args()


def _leer_respuestas(path: Path) -> dict[str, str]:
    """
    Carga raw_response por text_id desde JSONL.
    """

    responses: dict[str, str] = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            payload = json.loads(line)
            text_id = str(payload["text_id"])
            raw_response = payload.get("raw_response", "")
            if not isinstance(raw_response, str):
                raw_response = ""
            responses[text_id] = raw_response
    return responses


def reconstruir_predicciones(
    test_data: pd.DataFrame,
    raw_responses: dict[str, str],
    config: BaselineConfig,
) -> pd.DataFrame:
    """
    Reconstruye el CSV de predicciones desde respuestas crudas.
    """

    rows: list[dict[str, object]] = []
    for record in test_data.to_dict("records"):
        text_id = str(record[config.text_id_column])
        raw_response = raw_responses.get(text_id, "")
        error = None
        label = None
        score = None
        evidence: list[str] = []
        justification = ""
        try:
            classification = parse_llm_classification(raw_response)
            label = classification.label
            score = classification.score
            evidence = classification.evidence
            justification = classification.justification
            y_pred = label_to_prediction(label)
        except ValueError as exc:
            error = str(exc)
            y_pred = None

        row = {
            config.text_id_column: record[config.text_id_column],
            config.user_column: record[config.user_column],
            "y_pred": y_pred,
            "label_pred": label,
            "score": score,
            "evidence": evidence,
            "justification": justification,
            "error": error,
        }
        if config.target_column in record:
            row["y_true"] = record[config.target_column]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """
    Ejecuta reparación y evaluación.
    """

    args = parse_args()
    config = BaselineConfig()
    test_data = cargar_publicaciones_csv(args.test_input, config, require_label=False)
    predicciones = reconstruir_predicciones(
        test_data,
        _leer_respuestas(args.raw_responses),
        config,
    )

    csv_predicciones = predicciones.copy()
    csv_predicciones["evidence"] = serializar_evidence_para_csv(
        csv_predicciones["evidence"]
    )
    ensure_parent_dir(args.predictions_out)
    csv_predicciones.to_csv(args.predictions_out, index=False, quoting=csv.QUOTE_MINIMAL)

    error_count = int(predicciones["error"].notna().sum())
    if error_count:
        print(
            f"No se calcularon métricas porque quedan errores LLM: {error_count}",
            file=sys.stderr,
        )
        return

    evaluar_y_guardar_llm(predicciones, config, args.metrics_out)
    print(f"Predicciones reparadas guardadas en: {args.predictions_out}")
    print(f"Métricas LLM guardadas en: {args.metrics_out}")


if __name__ == "__main__":
    main()
