"""
Este módulo prueba el tracking ligero de experimentos.
"""

from __future__ import annotations

import csv
import json

from reddit_mental_health.tracking import (
    RunMetadata,
    actualizar_resumen,
    construir_run_id,
    construir_rutas_corrida,
    guardar_json,
    guardar_metadata,
    slugify,
)


def test_construir_run_id_normaliza_componentes() -> None:
    """
    Verifica que los identificadores de corrida sean estables y legibles.
    """

    run_id = construir_run_id(
        "Phase 2B Implementation",
        "Logistic Regression",
        "Word (1,2)",
        started_at="2026-05-14T01:02:03Z",
    )

    assert slugify("Word (1,2)") == "word-1-2"
    assert run_id == (
        "phase-2b-implementation__logistic-regression__word-1-2__"
        "20260514-010203z"
    )


def test_actualizar_resumen_aplana_metadata_y_metricas(tmp_path) -> None:
    """
    Verifica que el resumen se reconstruya desde una corrida guardada.
    """

    paths = construir_rutas_corrida(tmp_path, "run-001")
    metadata = RunMetadata(
        run_id="run-001",
        experiment_name="phase-2b-implementation",
        classifier_name="logistic_regression",
        feature_config_name="word_unigram_bigram",
        status="completed",
        started_at="2026-05-14T01:02:03Z",
        completed_at="2026-05-14T01:02:04Z",
        duration_seconds=1.0,
        input_path="data/raw/data_train.csv",
        random_state=42,
        validation_size=0.2,
        metrics_path=str(paths.metrics_path),
        predictions_path=str(paths.predictions_path),
        interpretability_path=str(paths.interpretability_path),
        model_path=str(paths.model_path),
    )
    metricas = {
        "roc_auc": 0.76,
        "protocol_auc": 0.69,
        "true_positive_rate": 0.67,
        "false_positive_rate": 0.28,
        "recall": 0.67,
        "precision": 0.74,
        "f1": 0.70,
        "confusion_matrix": {
            "true_negative": 97,
            "false_positive": 38,
            "false_negative": 54,
            "true_positive": 112,
        },
    }

    guardar_metadata(metadata, paths.metadata_path)
    guardar_json(metricas, paths.metrics_path)
    registros = actualizar_resumen(tmp_path)

    assert len(registros) == 1
    assert registros[0]["protocol_auc"] == 0.69
    assert registros[0]["true_positive"] == 112

    with (tmp_path / "summary.csv").open(encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    assert filas[0]["run_id"] == "run-001"
    assert filas[0]["classifier_name"] == "logistic_regression"

    summary_json = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["runs"][0]["feature_config_name"] == "word_unigram_bigram"
