"""
Consolida métricas de Fase 3 en una tabla comparativa.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reddit_mental_health.config import PROJECT_ROOT, ensure_parent_dir
from reddit_mental_health.phase3_config import Phase3Config


DEFAULT_BASELINE_METRICS = (
    PROJECT_ROOT / "data" / "processed" / "phase3" / "baseline_fold2_metrics.json"
)


def parse_args() -> argparse.Namespace:
    """
    Lee argumentos CLI del comparador.
    """

    phase3 = Phase3Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS)
    parser.add_argument("--embeddings-metrics", type=Path, default=phase3.embeddings_metrics_path)
    parser.add_argument("--llm-metrics", type=Path, default=phase3.llm_metrics_path)
    parser.add_argument("--csv-out", type=Path, default=phase3.comparison_csv_path)
    parser.add_argument("--json-out", type=Path, default=phase3.comparison_json_path)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Omite métodos sin archivo de métricas en lugar de fallar.",
    )
    return parser.parse_args()


def _timestamp_utc() -> str:
    """
    Genera timestamp UTC para metadata.
    """

    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _leer_metricas(path: Path, allow_missing: bool) -> dict[str, Any] | None:
    """
    Lee un archivo de métricas JSON.
    """

    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(f"No existe el archivo de métricas: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Archivo de métricas inválido: {path}")
    return payload


def _fila_metodo(
    method: str,
    representation: str,
    classifier: str,
    metrics_path: Path,
    metrics: dict[str, Any],
) -> dict[str, object]:
    """
    Convierte métricas del protocolo en una fila tabular.
    """

    confusion = metrics.get("confusion_matrix", {})
    if not isinstance(confusion, dict):
        confusion = {}
    return {
        "method": method,
        "representation": representation,
        "classifier": classifier,
        "metrics_path": str(metrics_path),
        "protocol_auc": metrics.get("protocol_auc"),
        "roc_auc": metrics.get("roc_auc"),
        "recall": metrics.get("recall"),
        "precision": metrics.get("precision"),
        "f1": metrics.get("f1"),
        "true_negative": confusion.get("true_negative"),
        "false_positive": confusion.get("false_positive"),
        "false_negative": confusion.get("false_negative"),
        "true_positive": confusion.get("true_positive"),
    }


def construir_comparacion(args: argparse.Namespace) -> pd.DataFrame:
    """
    Construye la tabla comparativa desde métricas JSON.
    """

    specs = [
        (
            "phase2b_baseline",
            "tfidf_char_wb_3_5",
            "complement_nb",
            args.baseline_metrics,
        ),
        (
            "phase3_embeddings",
            "ollama_nomic_embed_text",
            "logistic_regression",
            args.embeddings_metrics,
        ),
        (
            "phase3_llm_zero_shot",
            "ollama_qwen2.5_3b_instruct",
            "zero_shot_prompt",
            args.llm_metrics,
        ),
    ]
    rows = []
    for method, representation, classifier, path in specs:
        metrics = _leer_metricas(path, allow_missing=args.allow_missing)
        if metrics is None:
            continue
        rows.append(_fila_metodo(method, representation, classifier, path, metrics))
    if not rows:
        raise ValueError("No se encontró ningún archivo de métricas para comparar.")
    return pd.DataFrame(rows)


def guardar_comparacion(frame: pd.DataFrame, csv_out: Path, json_out: Path) -> None:
    """
    Guarda comparación en CSV y JSON.
    """

    ensure_parent_dir(csv_out)
    frame.to_csv(csv_out, index=False)
    payload = {
        "timestamp": _timestamp_utc(),
        "rows": frame.to_dict("records"),
    }
    ensure_parent_dir(json_out)
    json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """
    Ejecuta el comparador CLI.
    """

    args = parse_args()
    comparison = construir_comparacion(args)
    guardar_comparacion(comparison, args.csv_out, args.json_out)
    print(f"Comparación CSV guardada en: {args.csv_out}")
    print(f"Comparación JSON guardada en: {args.json_out}")


if __name__ == "__main__":
    main()
