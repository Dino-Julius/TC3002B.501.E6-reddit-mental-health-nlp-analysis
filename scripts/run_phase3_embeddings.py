"""
Ejecuta el método de Fase 3 basado en embeddings de Ollama.
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
from reddit_mental_health.phase3_embeddings import (
    construir_predicciones_embeddings,
    entrenar_clasificador_embeddings,
    evaluar_y_guardar_embeddings,
    generar_embeddings,
)


def parse_args() -> argparse.Namespace:
    """
    Lee argumentos CLI del flujo de embeddings.
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
        "--predictions-out",
        type=Path,
        default=phase3.embeddings_predictions_path,
    )
    parser.add_argument("--metrics-out", type=Path, default=phase3.embeddings_metrics_path)
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=phase3.embeddings_metadata_path,
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


def _guardar_json(payload: dict[str, object], path: Path) -> None:
    """
    Guarda metadata JSON.
    """

    ensure_parent_dir(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ejecutar_embeddings(args: argparse.Namespace) -> dict[str, object]:
    """
    Entrena y evalúa el método de embeddings.
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

    classifier = entrenar_clasificador_embeddings(
        x_train,
        train_data[config.target_column],
        random_state=config.random_state,
    )
    y_pred = classifier.predict(x_test)
    positive_index = list(classifier.classes_).index(config.positive_value)
    score = classifier.predict_proba(x_test)[:, positive_index]

    include_y_true = config.target_column in test_data.columns and not args.no_evaluate
    predicciones = construir_predicciones_embeddings(
        test_data,
        y_pred,
        score,
        config,
        include_y_true=include_y_true,
    )
    ensure_parent_dir(args.predictions_out)
    predicciones.to_csv(args.predictions_out, index=False)

    metricas = None
    if include_y_true:
        metricas = evaluar_y_guardar_embeddings(predicciones, config, args.metrics_out)

    metadata = {
        "timestamp": _timestamp_utc(),
        "method": "ollama_embeddings_logistic_regression",
        "embedding_model": args.embedding_model,
        "embedding_max_chars": int(args.embedding_max_chars),
        "classifier": "logistic_regression",
        "train_input": str(args.train_input),
        "test_input": str(args.test_input),
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "embedding_dimensions": int(x_train.shape[1]),
        "predictions_out": str(args.predictions_out),
        "metrics_out": str(args.metrics_out) if metricas else None,
        "train_cache": str(args.train_cache),
        "test_cache": str(args.test_cache),
    }
    _guardar_json(metadata, args.metadata_out)

    print(f"Predicciones guardadas en: {args.predictions_out}")
    if metricas:
        print(f"Métricas guardadas en: {args.metrics_out}")
    print(f"Metadata guardada en: {args.metadata_out}")
    return {"metadata": metadata, "metrics": metricas}


def main() -> None:
    """
    Ejecuta el CLI.
    """

    ejecutar_embeddings(parse_args())


if __name__ == "__main__":
    main()
