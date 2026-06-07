"""
Pruebas del comparador de métricas Phase 3.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _cargar_script_compare_phase3() -> ModuleType:
    """
    Carga el comparador por ruta para probar helpers.
    """

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_phase3_results.py"
    spec = importlib.util.spec_from_file_location("compare_phase3_results", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar scripts/compare_phase3_results.py.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARE_PHASE3 = _cargar_script_compare_phase3()


def _guardar_metricas(path: Path, auc: float) -> None:
    """
    Guarda métricas mínimas para comparar.
    """

    path.write_text(
        json.dumps(
            {
                "protocol_auc": auc,
                "roc_auc": auc + 0.1,
                "recall": 0.5,
                "precision": 0.6,
                "f1": 0.55,
                "confusion_matrix": {
                    "true_negative": 1,
                    "false_positive": 2,
                    "false_negative": 3,
                    "true_positive": 4,
                },
            }
        ),
        encoding="utf-8",
    )


def test_construir_comparacion_une_metricas(tmp_path) -> None:
    """
    Verifica que el comparador genere una fila por método disponible.
    """

    baseline = tmp_path / "baseline.json"
    embeddings = tmp_path / "embeddings.json"
    llm = tmp_path / "llm.json"
    _guardar_metricas(baseline, 0.6)
    _guardar_metricas(embeddings, 0.7)
    _guardar_metricas(llm, 0.8)
    args = argparse.Namespace(
        baseline_metrics=baseline,
        embeddings_metrics=embeddings,
        llm_metrics=llm,
        allow_missing=False,
    )

    frame = COMPARE_PHASE3.construir_comparacion(args)

    assert frame["method"].tolist() == [
        "phase2b_baseline",
        "phase3_embeddings",
        "phase3_llm_zero_shot",
    ]
    assert frame["false_negative"].tolist() == [3, 3, 3]


def test_guardar_comparacion_exporta_csv_y_json(tmp_path) -> None:
    """
    Verifica exportación de la tabla final.
    """

    metrics = tmp_path / "baseline.json"
    _guardar_metricas(metrics, 0.6)
    args = argparse.Namespace(
        baseline_metrics=metrics,
        embeddings_metrics=tmp_path / "missing_embeddings.json",
        llm_metrics=tmp_path / "missing_llm.json",
        allow_missing=True,
    )
    frame = COMPARE_PHASE3.construir_comparacion(args)
    csv_out = tmp_path / "comparison.csv"
    json_out = tmp_path / "comparison.json"

    COMPARE_PHASE3.guardar_comparacion(frame, csv_out, json_out)

    assert csv_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["rows"][0]["method"] == "phase2b_baseline"
