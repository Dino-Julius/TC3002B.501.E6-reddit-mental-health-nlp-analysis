"""
Este módulo evalúa un baseline entrenado sobre datos nuevos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reddit_mental_health.config import BaselineConfig, ensure_parent_dir
from reddit_mental_health.data import (
    cargar_publicaciones_csv,
    construir_salida_predicciones,
)
from reddit_mental_health.evaluation import calcular_metricas, guardar_metricas
from reddit_mental_health.interpretability import (
    construir_reporte_interpretabilidad,
    guardar_reporte_interpretabilidad,
)
from reddit_mental_health.model import cargar_modelo, predecir_baseline
from reddit_mental_health.preprocessing import (
    combinar_title_text,
    preprocesar_publicaciones,
)


def parse_args() -> argparse.Namespace:
    """
    Lee argumentos CLI para evaluar un modelo persistido.
    """

    config = BaselineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=config.model_path)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=config.project_root / "data" / "processed" / "baseline_predictions.csv",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=config.project_root / "data" / "processed" / "baseline_eval_metrics.json",
    )
    parser.add_argument(
        "--interpretability-out",
        type=Path,
        default=config.project_root
        / "data"
        / "processed"
        / "baseline_eval_interpretability.json",
    )
    return parser.parse_args()


def main() -> None:
    """
    Genera predicciones y métricas si el CSV incluye etiquetas reales.
    """

    args = parse_args()
    config = BaselineConfig(
        input_path=args.input,
        model_path=args.model,
        predictions_path=args.predictions_out,
        metrics_path=args.metrics_out,
        interpretability_path=args.interpretability_out,
    )

    datos = cargar_publicaciones_csv(
        config.input_path,
        config,
        require_label=False,
    )
    modelo = cargar_modelo(config.model_path)
    textos = preprocesar_publicaciones(datos, config)
    y_pred, score = predecir_baseline(modelo, textos)

    tiene_etiquetas = config.target_column in datos.columns
    predicciones = construir_salida_predicciones(
        datos,
        y_pred,
        score,
        config,
        split="evaluacion",
        include_y_true=tiene_etiquetas,
    )
    ensure_parent_dir(config.predictions_path)
    predicciones.to_csv(config.predictions_path, index=False)

    if tiene_etiquetas:
        metricas = calcular_metricas(
            predicciones["y_true"],
            predicciones["y_pred"],
            predicciones["score"],
            config,
        )
        guardar_metricas(metricas, config.metrics_path)
        print(f"Métricas guardadas en: {config.metrics_path}")
    else:
        print("El CSV no incluye etiquetas; solo se guardaron predicciones.")

    reporte = construir_reporte_interpretabilidad(
        modelo,
        predicciones,
        combinar_title_text(datos, config),
        config,
    )
    guardar_reporte_interpretabilidad(reporte, config.interpretability_path)

    print(f"Predicciones guardadas en: {config.predictions_path}")
    print(f"Reporte interpretativo guardado en: {config.interpretability_path}")


if __name__ == "__main__":
    main()
