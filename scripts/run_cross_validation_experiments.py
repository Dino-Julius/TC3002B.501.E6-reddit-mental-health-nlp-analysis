"""
Este script ejecuta validación cruzada para selección de modelo Phase 2B.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.cross_validation import (
    ejecutar_matriz_cv,
    resolver_matriz_cv,
    resumir_resultados_cv,
    seleccionar_mejor_modelo_cv,
)
from reddit_mental_health.data import cargar_publicaciones_csv
from reddit_mental_health.experiments import (
    listar_clasificadores,
    listar_configuraciones_features,
)


def _choices_con_all(valores: tuple[str, ...]) -> tuple[str, ...]:
    """
    Agrega la opción all a una lista estable de valores CLI.
    """

    return (*valores, "all")


def parse_args() -> argparse.Namespace:
    """
    Lee argumentos CLI para la matriz de validación cruzada.
    """

    config = BaselineConfig()
    default_output = (
        config.project_root
        / "data"
        / "processed"
        / "cross_validation"
        / "phase-2b-feedback"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.input_path)
    parser.add_argument(
        "--classifier-name",
        choices=_choices_con_all(listar_clasificadores()),
        default="all",
    )
    parser.add_argument(
        "--feature-config-name",
        choices=_choices_con_all(listar_configuraciones_features()),
        default="all",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def _guardar_json(data: object, path: Path) -> None:
    """
    Guarda un objeto JSON legible.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ejecutar_validacion_cruzada(args: argparse.Namespace) -> dict[str, Path | dict[str, object]]:
    """
    Ejecuta CV y persiste resultados tabulares y resumen del mejor modelo.
    """

    base_config = BaselineConfig(input_path=args.input, random_state=args.random_state)
    datos = cargar_publicaciones_csv(args.input, base_config, require_label=True)
    matriz = resolver_matriz_cv(args.classifier_name, args.feature_config_name)

    fold_results = ejecutar_matriz_cv(
        datos,
        base_config,
        matriz,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    summary_cv = resumir_resultados_cv(fold_results, n_splits=args.n_splits)
    best_model = seleccionar_mejor_modelo_cv(summary_cv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_results_path = args.output_dir / "fold_results.csv"
    summary_csv_path = args.output_dir / "summary_cv.csv"
    summary_json_path = args.output_dir / "summary_cv.json"
    best_model_path = args.output_dir / "best_model_cv.json"

    fold_results.to_csv(fold_results_path, index=False)
    summary_cv.to_csv(summary_csv_path, index=False)
    _guardar_json(summary_cv.to_dict(orient="records"), summary_json_path)
    _guardar_json(best_model, best_model_path)

    return {
        "fold_results": fold_results_path,
        "summary_csv": summary_csv_path,
        "summary_json": summary_json_path,
        "best_model_path": best_model_path,
        "best_model": best_model,
    }


def main() -> None:
    """
    Ejecuta validación cruzada y reporta los artefactos generados.
    """

    args = parse_args()
    resultado = ejecutar_validacion_cruzada(args)
    best_model = resultado["best_model"]

    print(f"Fold results: {resultado['fold_results']}")
    print(f"Summary CSV: {resultado['summary_csv']}")
    print(f"Summary JSON: {resultado['summary_json']}")
    print(f"Best model CV: {resultado['best_model_path']}")
    print(
        "Mejor combinación CV: "
        f"{best_model['classifier_name']} + {best_model['feature_config_name']} "
        f"(mean_protocol_auc={best_model['mean_protocol_auc']:.4f})",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Ejecución interrumpida por el usuario.")
