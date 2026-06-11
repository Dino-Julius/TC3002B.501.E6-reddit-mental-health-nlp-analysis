"""
Ejecuta una matriz de clasificadores sobre embeddings de Fase 3.
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

from reddit_mental_health.config import BaselineConfig, ensure_parent_dir
from reddit_mental_health.data import cargar_publicaciones_csv
from reddit_mental_health.ollama import OllamaClient
from reddit_mental_health.phase3_config import Phase3Config
from reddit_mental_health.phase3_embeddings import (
    construir_predicciones_embeddings,
    entrenar_clasificador_embeddings_por_nombre,
    evaluar_y_guardar_embeddings,
    generar_embeddings,
    listar_clasificadores_embeddings,
    obtener_score_clase_positiva,
)


def parse_args() -> argparse.Namespace:
    """
    Lee argumentos CLI del flujo de matriz de embeddings.
    """

    phase3 = Phase3Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=phase3.train_input)
    parser.add_argument("--test-input", type=Path, default=phase3.test_input)
    parser.add_argument("--ollama-url", default=phase3.ollama_base_url)
    parser.add_argument("--embedding-model", default=phase3.embedding_model)
    parser.add_argument(
        "--embedding-max-chars",
        type=int,
        default=phase3.embedding_max_chars,
        help="Trunca textos antes de pedir embeddings para respetar contexto local.",
    )
    parser.add_argument("--train-cache", type=Path, default=phase3.embeddings_train_cache)
    parser.add_argument("--test-cache", type=Path, default=phase3.embeddings_test_cache)
    parser.add_argument(
        "--classifier-name",
        choices=[*listar_clasificadores_embeddings(), "all"],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=phase3.embedding_classifiers_dir)
    parser.add_argument(
        "--summary-csv-out",
        type=Path,
        default=phase3.embedding_classifiers_summary_csv_path,
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=phase3.embedding_classifiers_summary_json_path,
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


def _guardar_json(payload: dict[str, Any], path: Path) -> None:
    """
    Guarda un objeto JSON legible.
    """

    ensure_parent_dir(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _clasificadores_a_ejecutar(classifier_name: str) -> tuple[str, ...]:
    """
    Expande 'all' a la matriz completa de clasificadores densos.
    """

    if classifier_name == "all":
        return listar_clasificadores_embeddings()
    return (classifier_name,)


def _fila_resumen(
    classifier_name: str,
    predictions_path: Path,
    metrics_path: Path | None,
    metrics: dict[str, Any] | None,
) -> dict[str, object]:
    """
    Construye una fila estable del resumen comparativo.
    """

    confusion = metrics.get("confusion_matrix", {}) if metrics else {}
    if not isinstance(confusion, dict):
        confusion = {}
    return {
        "method": f"phase3_embeddings_{classifier_name}",
        "representation": "ollama_nomic_embed_text",
        "classifier": classifier_name,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path) if metrics_path else None,
        "protocol_auc": metrics.get("protocol_auc") if metrics else None,
        "roc_auc": metrics.get("roc_auc") if metrics else None,
        "recall": metrics.get("recall") if metrics else None,
        "precision": metrics.get("precision") if metrics else None,
        "f1": metrics.get("f1") if metrics else None,
        "true_negative": confusion.get("true_negative"),
        "false_positive": confusion.get("false_positive"),
        "false_negative": confusion.get("false_negative"),
        "true_positive": confusion.get("true_positive"),
    }


def ejecutar_matriz_embeddings(args: argparse.Namespace) -> pd.DataFrame:
    """
    Ejecuta la matriz de clasificadores sobre embeddings ya cacheables.
    """

    config = BaselineConfig(input_path=args.train_input)
    train_data = cargar_publicaciones_csv(args.train_input, config, require_label=True)
    test_data = cargar_publicaciones_csv(args.test_input, config, require_label=False)

    client = OllamaClient(base_url=args.ollama_url)
    x_train = generar_embeddings(
        train_data,
        config,
        client,
        args.embedding_model,
        args.train_cache,
        max_chars=args.embedding_max_chars,
    )
    x_test = generar_embeddings(
        test_data,
        config,
        client,
        args.embedding_model,
        args.test_cache,
        max_chars=args.embedding_max_chars,
    )

    include_y_true = config.target_column in test_data.columns and not args.no_evaluate
    rows: list[dict[str, object]] = []
    for classifier_name in _clasificadores_a_ejecutar(args.classifier_name):
        classifier = entrenar_clasificador_embeddings_por_nombre(
            x_train,
            train_data[config.target_column],
            classifier_name,
            random_state=config.random_state,
        )
        y_pred = classifier.predict(x_test)
        score = obtener_score_clase_positiva(
            classifier,
            x_test,
            config.positive_value,
        )
        predicciones = construir_predicciones_embeddings(
            test_data,
            y_pred,
            score,
            config,
            include_y_true=include_y_true,
        )

        predictions_path = args.output_dir / f"{classifier_name}_predictions.csv"
        metrics_path = args.output_dir / f"{classifier_name}_metrics.json"
        ensure_parent_dir(predictions_path)
        predicciones.to_csv(predictions_path, index=False)

        metrics = None
        if include_y_true:
            metrics = evaluar_y_guardar_embeddings(predicciones, config, metrics_path)

        rows.append(
            _fila_resumen(
                classifier_name,
                predictions_path,
                metrics_path if metrics else None,
                metrics,
            )
        )
        print(f"Predicciones {classifier_name}: {predictions_path}")
        if metrics:
            print(f"Métricas {classifier_name}: {metrics_path}")

    summary = pd.DataFrame(rows)
    ensure_parent_dir(args.summary_csv_out)
    summary.to_csv(args.summary_csv_out, index=False)
    _guardar_json(
        {
            "timestamp": _timestamp_utc(),
            "method": "phase3_embedding_classifier_matrix",
            "embedding_model": args.embedding_model,
            "embedding_max_chars": int(args.embedding_max_chars),
            "train_input": str(args.train_input),
            "test_input": str(args.test_input),
            "train_rows": int(len(train_data)),
            "test_rows": int(len(test_data)),
            "embedding_dimensions": int(x_train.shape[1]),
            "classifiers": list(_clasificadores_a_ejecutar(args.classifier_name)),
            "rows": rows,
        },
        args.summary_json_out,
    )
    print(f"Resumen CSV: {args.summary_csv_out}")
    print(f"Resumen JSON: {args.summary_json_out}")
    return summary


def main() -> None:
    """
    Ejecuta el CLI.
    """

    ejecutar_matriz_embeddings(parse_args())


if __name__ == "__main__":
    main()
