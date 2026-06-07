"""
Este módulo centraliza la configuración de los experimentos de Fase 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reddit_mental_health.config import PROJECT_ROOT


@dataclass(frozen=True)
class Phase3Config:
    """
    Define rutas y modelos locales reproducibles para Fase 3.
    """

    project_root: Path = PROJECT_ROOT
    train_input: Path = PROJECT_ROOT / "data" / "raw" / "data_train.csv"
    test_input: Path = PROJECT_ROOT / "data" / "raw" / "data_test_fold2.csv"
    output_dir: Path = PROJECT_ROOT / "data" / "processed" / "phase3"

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "qwen2.5:3b-instruct"
    embedding_max_chars: int = 6_000

    embeddings_train_cache: Path = (
        PROJECT_ROOT / "data" / "processed" / "phase3" / "embeddings_train.json"
    )
    embeddings_test_cache: Path = (
        PROJECT_ROOT / "data" / "processed" / "phase3" / "embeddings_test_fold2.json"
    )
    embeddings_predictions_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "phase3" / "embeddings_predictions.csv"
    )
    embeddings_metrics_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "phase3" / "embeddings_metrics.json"
    )
    embeddings_metadata_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "phase3" / "embeddings_metadata.json"
    )

    llm_raw_responses_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "phase3" / "llm_zero_shot_responses.jsonl"
    )
    llm_predictions_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "phase3" / "llm_zero_shot_predictions.csv"
    )
    llm_metrics_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "phase3" / "llm_zero_shot_metrics.json"
    )
    llm_metadata_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "phase3" / "llm_zero_shot_metadata.json"
    )

    comparison_csv_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "phase3" / "phase3_comparison.csv"
    )
    comparison_json_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "phase3" / "phase3_comparison.json"
    )


__all__ = ["Phase3Config"]
