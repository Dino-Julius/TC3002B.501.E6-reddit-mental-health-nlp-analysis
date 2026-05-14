"""Configuración central del baseline de PLN para Reddit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BaselineConfig:
    """Parámetros reproducibles para el pipeline conceptual.

    La configuración concentra rutas, nombres de columnas y valores del
    experimento para que los módulos de datos, texto, modelo y evaluación no
    dependan de constantes dispersas.
    """

    project_root: Path = PROJECT_ROOT
    input_path: Path = PROJECT_ROOT / "data" / "raw" / "data_train.csv"
    model_path: Path = PROJECT_ROOT / "data" / "processed" / "baseline_model.joblib"
    predictions_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "baseline_validation_predictions.csv"
    )
    metrics_path: Path = PROJECT_ROOT / "data" / "processed" / "baseline_metrics.json"
    interpretability_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "baseline_interpretability.json"
    )

    user_column: str = "user_id"
    text_id_column: str = "text_id"
    title_column: str = "title"
    text_column: str = "text"
    label_column: str = "is_suicide"
    target_column: str = "y"
    split_column: str = "split"

    positive_label: str = "yes"
    negative_label: str = "no"
    positive_value: int = 1
    negative_value: int = 0

    validation_size: float = 0.2
    random_state: int = 42

    classifier_name: str = "logistic_regression"
    feature_config_name: str = "word_unigram_bigram"

    analyzer: str = "word"
    max_features: int = 20_000
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: tuple[int, int] = (1, 2)

    logistic_c: float = 1.0
    logistic_max_iter: int = 1_000
    linear_svm_c: float = 1.0
    linear_svm_max_iter: int = 5_000
    sgd_alpha: float = 0.0001
    sgd_max_iter: int = 1_000
    naive_bayes_alpha: float = 1.0
    class_weight: str | None = "balanced"


def ensure_parent_dir(path: Path) -> None:
    """Crea el directorio padre de una salida si todavía no existe."""

    path.parent.mkdir(parents=True, exist_ok=True)


__all__ = ["BaselineConfig", "PROJECT_ROOT", "ensure_parent_dir"]
