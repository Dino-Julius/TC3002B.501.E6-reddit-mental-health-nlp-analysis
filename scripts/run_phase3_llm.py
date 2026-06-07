"""
Ejecuta el método de Fase 3 basado en LLM local zero-shot con Ollama.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reddit_mental_health.config import BaselineConfig, ensure_parent_dir
from reddit_mental_health.data import cargar_publicaciones_csv
from reddit_mental_health.ollama import OllamaClient
from reddit_mental_health.phase3_config import Phase3Config
from reddit_mental_health.phase3_llm import (
    clasificar_publicaciones_llm,
    evaluar_y_guardar_llm,
    serializar_evidence_para_csv,
)


def parse_args() -> argparse.Namespace:
    """
    Lee argumentos CLI del flujo LLM.
    """

    phase3 = Phase3Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-input", type=Path, default=phase3.test_input)
    parser.add_argument("--ollama-url", default=phase3.ollama_base_url)
    parser.add_argument("--llm-model", default=phase3.llm_model)
    parser.add_argument(
        "--raw-responses-out",
        type=Path,
        default=phase3.llm_raw_responses_path,
    )
    parser.add_argument("--predictions-out", type=Path, default=phase3.llm_predictions_path)
    parser.add_argument("--metrics-out", type=Path, default=phase3.llm_metrics_path)
    parser.add_argument("--metadata-out", type=Path, default=phase3.llm_metadata_path)
    parser.add_argument("--max-attempts", type=int, default=2)
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


def _guardar_json(payload: dict[str, object], path: Path) -> None:
    """
    Guarda metadata JSON.
    """

    ensure_parent_dir(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ejecutar_llm(args: argparse.Namespace) -> dict[str, object]:
    """
    Clasifica y evalúa fold de prueba con LLM local.
    """

    config = BaselineConfig()
    test_data = cargar_publicaciones_csv(args.test_input, config, require_label=False)
    client = OllamaClient(base_url=args.ollama_url)

    predicciones = clasificar_publicaciones_llm(
        test_data,
        config,
        client,
        args.llm_model,
        args.raw_responses_out,
        max_attempts=args.max_attempts,
    )

    csv_predicciones = predicciones.copy()
    csv_predicciones["evidence"] = serializar_evidence_para_csv(
        csv_predicciones["evidence"]
    )
    ensure_parent_dir(args.predictions_out)
    csv_predicciones.to_csv(args.predictions_out, index=False)

    metricas = None
    has_labels = config.target_column in predicciones.columns
    error_count = int(predicciones["error"].notna().sum())
    if has_labels and not args.no_evaluate and error_count == 0:
        metricas = evaluar_y_guardar_llm(predicciones, config, args.metrics_out)

    metadata = {
        "timestamp": _timestamp_utc(),
        "method": "ollama_llm_zero_shot",
        "llm_model": args.llm_model,
        "test_input": str(args.test_input),
        "test_rows": int(len(test_data)),
        "raw_responses_out": str(args.raw_responses_out),
        "predictions_out": str(args.predictions_out),
        "metrics_out": str(args.metrics_out) if metricas else None,
        "max_attempts": int(args.max_attempts),
        "error_count": error_count,
    }
    _guardar_json(metadata, args.metadata_out)

    print(f"Respuestas crudas guardadas en: {args.raw_responses_out}")
    print(f"Predicciones guardadas en: {args.predictions_out}")
    if metricas:
        print(f"Métricas guardadas en: {args.metrics_out}")
    elif error_count:
        print(
            "No se generaron métricas porque existen respuestas LLM inválidas; "
            f"errores: {error_count}",
            file=sys.stderr,
        )
    print(f"Metadata guardada en: {args.metadata_out}")
    return {"metadata": metadata, "metrics": metricas}


def main() -> None:
    """
    Ejecuta el CLI.
    """

    ejecutar_llm(parse_args())


if __name__ == "__main__":
    main()
