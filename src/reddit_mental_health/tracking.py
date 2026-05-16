"""
Este módulo gestiona tracking local y ligero de corridas experimentales.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "run_id",
    "status",
    "experiment_name",
    "classifier_name",
    "feature_config_name",
    "started_at",
    "completed_at",
    "duration_seconds",
    "random_state",
    "validation_size",
    "roc_auc",
    "protocol_auc",
    "true_positive_rate",
    "false_positive_rate",
    "recall",
    "precision",
    "f1",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
]


@dataclass(frozen=True)
class RunPaths:
    """
    Define rutas estándar de artefactos para una corrida.
    """

    run_dir: Path
    metadata_path: Path
    metrics_path: Path
    predictions_path: Path
    interpretability_path: Path
    split_diagnostics_path: Path
    model_path: Path


@dataclass(frozen=True)
class RunMetadata:
    """
    Define metadata mínima para reproducir y comparar una corrida.
    """

    run_id: str
    experiment_name: str
    classifier_name: str
    feature_config_name: str
    status: str
    started_at: str
    completed_at: str | None
    duration_seconds: float | None
    input_path: str
    random_state: int
    validation_size: float
    metrics_path: str
    predictions_path: str
    interpretability_path: str
    model_path: str | None
    error: str | None = None


def timestamp_utc() -> str:
    """
    Genera un timestamp UTC compacto y serializable.
    """

    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: object) -> str:
    """
    Normaliza texto para usarlo de forma segura en nombres de archivos.
    """

    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "sin-nombre"


def construir_run_id(
    experiment_name: str,
    classifier_name: str,
    feature_config_name: str,
    started_at: str | None = None,
) -> str:
    """
    Construye un identificador legible para una corrida.
    """

    timestamp = started_at or timestamp_utc()
    timestamp_slug = timestamp.replace("-", "").replace(":", "").replace("+", "")
    timestamp_slug = timestamp_slug.replace("T", "-").replace("Z", "z")
    partes = [
        slugify(experiment_name),
        slugify(classifier_name),
        slugify(feature_config_name),
        timestamp_slug,
    ]
    return "__".join(partes)


def construir_rutas_corrida(output_dir: str | Path, run_id: str) -> RunPaths:
    """
    Devuelve las rutas estándar para una corrida dentro del directorio base.
    """

    run_dir = Path(output_dir) / run_id
    return RunPaths(
        run_dir=run_dir,
        metadata_path=run_dir / "run_metadata.json",
        metrics_path=run_dir / "metrics.json",
        predictions_path=run_dir / "predictions.csv",
        interpretability_path=run_dir / "interpretability.json",
        split_diagnostics_path=run_dir / "split_diagnostics.json",
        model_path=run_dir / "model.joblib",
    )


def guardar_json(payload: dict[str, Any], path: str | Path) -> None:
    """
    Guarda un diccionario como JSON legible.
    """

    salida = Path(path)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def guardar_metadata(metadata: RunMetadata, path: str | Path) -> None:
    """
    Persiste la metadata de una corrida.
    """

    guardar_json(asdict(metadata), path)


def _leer_json(path: Path) -> dict[str, Any]:
    """
    Carga un JSON si existe; si no, regresa un diccionario vacío.
    """

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def construir_registro_resumen(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Aplana metadata y métricas para el resumen tabular.
    """

    matriz = metrics.get("confusion_matrix")
    if not isinstance(matriz, dict):
        matriz = {}

    return {
        "run_id": metadata.get("run_id"),
        "status": metadata.get("status"),
        "experiment_name": metadata.get("experiment_name"),
        "classifier_name": metadata.get("classifier_name"),
        "feature_config_name": metadata.get("feature_config_name"),
        "started_at": metadata.get("started_at"),
        "completed_at": metadata.get("completed_at"),
        "duration_seconds": metadata.get("duration_seconds"),
        "random_state": metadata.get("random_state"),
        "validation_size": metadata.get("validation_size"),
        "roc_auc": metrics.get("roc_auc"),
        "protocol_auc": metrics.get("protocol_auc"),
        "true_positive_rate": metrics.get("true_positive_rate"),
        "false_positive_rate": metrics.get("false_positive_rate"),
        "recall": metrics.get("recall"),
        "precision": metrics.get("precision"),
        "f1": metrics.get("f1"),
        "true_negative": matriz.get("true_negative"),
        "false_positive": matriz.get("false_positive"),
        "false_negative": matriz.get("false_negative"),
        "true_positive": matriz.get("true_positive"),
    }


def actualizar_resumen(output_dir: str | Path) -> list[dict[str, Any]]:
    """
    Reconstruye summary.csv y summary.json a partir de corridas guardadas.
    """

    base_dir = Path(output_dir)
    registros = []
    for metadata_path in sorted(base_dir.glob("*/run_metadata.json")):
        metadata = _leer_json(metadata_path)
        metrics = _leer_json(metadata_path.parent / "metrics.json")
        registros.append(construir_registro_resumen(metadata, metrics))

    summary_csv = base_dir / "summary.csv"
    summary_json = base_dir / "summary.json"
    base_dir.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(registros)
    guardar_json({"runs": registros}, summary_json)
    return registros


__all__ = [
    "RunMetadata",
    "RunPaths",
    "actualizar_resumen",
    "construir_registro_resumen",
    "construir_rutas_corrida",
    "construir_run_id",
    "guardar_json",
    "guardar_metadata",
    "slugify",
    "timestamp_utc",
]
