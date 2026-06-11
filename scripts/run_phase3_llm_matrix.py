"""
Ejecuta matriz Phase 3 de LLMs locales con zero-shot y few-shot.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reddit_mental_health.config import BaselineConfig, ensure_parent_dir
from reddit_mental_health.data import cargar_publicaciones_csv
from reddit_mental_health.ollama import OllamaClient
from reddit_mental_health.phase3_config import Phase3Config
from reddit_mental_health.phase3_llm import (
    clasificar_publicaciones_llm,
    ejemplos_few_shot_a_json,
    evaluar_y_guardar_llm,
    seleccionar_ejemplos_few_shot,
    serializar_evidence_para_csv,
)


PROMPT_MODES = ("zero_shot", "few_shot")


def parse_args() -> argparse.Namespace:
    """
    Lee argumentos CLI de la matriz LLM.
    """

    phase3 = Phase3Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=phase3.train_input)
    parser.add_argument("--test-input", type=Path, default=phase3.test_input)
    parser.add_argument("--output-dir", type=Path, default=phase3.llm_matrix_dir)
    parser.add_argument("--ollama-url", default=phase3.ollama_base_url)
    parser.add_argument(
        "--llm-model",
        action="append",
        dest="llm_models",
        default=None,
        help="Modelo LLM local. Puede repetirse. Default: qwen2.5, llama3.2 y gemma3.",
    )
    parser.add_argument(
        "--prompt-mode",
        action="append",
        choices=PROMPT_MODES,
        dest="prompt_modes",
        default=None,
        help="Modo de prompt. Puede repetirse. Default: zero_shot y few_shot.",
    )
    parser.add_argument(
        "--few-shot-examples-per-class",
        type=int,
        default=phase3.few_shot_examples_per_class,
    )
    parser.add_argument("--few-shot-random-state", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--summary-csv-out", type=Path, default=phase3.llm_matrix_summary_csv_path)
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=phase3.llm_matrix_summary_json_path,
    )
    parser.add_argument(
        "--few-shot-examples-out",
        type=Path,
        default=phase3.few_shot_examples_path,
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Omite métricas aunque el fold tenga etiquetas.",
    )
    return parser.parse_args()


def _timestamp_utc() -> str:
    """
    Genera timestamp UTC para metadata.
    """

    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug_model(model_name: str) -> str:
    """
    Convierte nombres Ollama a slugs seguros para archivos.
    """

    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


def _metricas_a_fila(
    model_name: str,
    prompt_mode: str,
    metrics_path: Path,
    predictions_path: Path,
    raw_responses_path: Path,
    metricas: dict[str, Any] | None,
    error_count: int,
) -> dict[str, object]:
    """
    Normaliza resultados de una corrida LLM a una fila comparativa.
    """

    metricas = metricas or {}
    confusion = metricas.get("confusion_matrix", {})
    if not isinstance(confusion, dict):
        confusion = {}
    return {
        "method": f"phase3_llm_{prompt_mode}",
        "model": model_name,
        "representation": f"ollama_{_slug_model(model_name)}",
        "classifier": f"{prompt_mode}_prompt",
        "prompt_mode": prompt_mode,
        "metrics_path": str(metrics_path) if metricas else None,
        "predictions_path": str(predictions_path),
        "raw_responses_path": str(raw_responses_path),
        "error_count": error_count,
        "protocol_auc": metricas.get("protocol_auc"),
        "roc_auc": metricas.get("roc_auc"),
        "recall": metricas.get("recall"),
        "precision": metricas.get("precision"),
        "f1": metricas.get("f1"),
        "true_negative": confusion.get("true_negative"),
        "false_positive": confusion.get("false_positive"),
        "false_negative": confusion.get("false_negative"),
        "true_positive": confusion.get("true_positive"),
    }


def _guardar_json(payload: dict[str, object] | list[dict[str, object]], path: Path) -> None:
    """
    Guarda JSON con indentación estable.
    """

    ensure_parent_dir(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ejecutar_matriz_llm(args: argparse.Namespace) -> pd.DataFrame:
    """
    Ejecuta todas las combinaciones modelo x prompt y guarda resúmenes.
    """

    phase3 = Phase3Config()
    config = BaselineConfig()
    llm_models = tuple(args.llm_models or phase3.llm_matrix_models)
    prompt_modes = tuple(args.prompt_modes or PROMPT_MODES)

    train_data = cargar_publicaciones_csv(args.train_input, config, require_label=True)
    test_data = cargar_publicaciones_csv(args.test_input, config, require_label=False)
    few_shot_examples = seleccionar_ejemplos_few_shot(
        train_data,
        config,
        examples_per_class=args.few_shot_examples_per_class,
        random_state=args.few_shot_random_state,
    )
    _guardar_json(
        {
            "timestamp": _timestamp_utc(),
            "train_input": str(args.train_input),
            "examples_per_class": int(args.few_shot_examples_per_class),
            "random_state": int(args.few_shot_random_state),
            "examples": ejemplos_few_shot_a_json(few_shot_examples),
        },
        args.few_shot_examples_out,
    )

    client = OllamaClient(base_url=args.ollama_url)
    rows: list[dict[str, object]] = []
    has_labels = config.target_column in test_data.columns

    for model_name in llm_models:
        model_slug = _slug_model(model_name)
        for prompt_mode in prompt_modes:
            prefix = f"{model_slug}_{prompt_mode}"
            raw_responses_path = args.output_dir / f"{prefix}_responses.jsonl"
            predictions_path = args.output_dir / f"{prefix}_predictions.csv"
            metrics_path = args.output_dir / f"{prefix}_metrics.json"
            metadata_path = args.output_dir / f"{prefix}_metadata.json"

            predicciones = clasificar_publicaciones_llm(
                test_data,
                config,
                client,
                model_name,
                raw_responses_path,
                max_attempts=args.max_attempts,
                prompt_mode=prompt_mode,
                few_shot_examples=few_shot_examples if prompt_mode == "few_shot" else None,
            )

            csv_predicciones = predicciones.copy()
            csv_predicciones["evidence"] = serializar_evidence_para_csv(
                csv_predicciones["evidence"]
            )
            ensure_parent_dir(predictions_path)
            csv_predicciones.to_csv(predictions_path, index=False)

            error_count = int(predicciones["error"].notna().sum())
            metricas = None
            if has_labels and not args.no_evaluate and error_count == 0:
                metricas = evaluar_y_guardar_llm(predicciones, config, metrics_path)

            _guardar_json(
                {
                    "timestamp": _timestamp_utc(),
                    "method": f"ollama_llm_{prompt_mode}",
                    "llm_model": model_name,
                    "prompt_mode": prompt_mode,
                    "test_input": str(args.test_input),
                    "test_rows": int(len(test_data)),
                    "raw_responses_out": str(raw_responses_path),
                    "predictions_out": str(predictions_path),
                    "metrics_out": str(metrics_path) if metricas else None,
                    "max_attempts": int(args.max_attempts),
                    "error_count": error_count,
                    "few_shot_examples_out": str(args.few_shot_examples_out)
                    if prompt_mode == "few_shot"
                    else None,
                },
                metadata_path,
            )
            rows.append(
                _metricas_a_fila(
                    model_name,
                    prompt_mode,
                    metrics_path,
                    predictions_path,
                    raw_responses_path,
                    metricas,
                    error_count,
                )
            )
            print(
                f"{model_name} / {prompt_mode}: "
                f"predicciones={predictions_path} errores={error_count}"
            )

    summary = pd.DataFrame(rows)
    ensure_parent_dir(args.summary_csv_out)
    summary.to_csv(args.summary_csv_out, index=False)
    _guardar_json({"timestamp": _timestamp_utc(), "rows": rows}, args.summary_json_out)
    print(f"Resumen CSV guardado en: {args.summary_csv_out}")
    print(f"Resumen JSON guardado en: {args.summary_json_out}")
    return summary


def main() -> None:
    """
    Ejecuta el CLI.
    """

    ejecutar_matriz_llm(parse_args())


if __name__ == "__main__":
    main()
