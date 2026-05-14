"""
Este módulo entrena el baseline principal de Fase 2B.
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
from reddit_mental_health.model import (
    entrenar_baseline,
    guardar_modelo,
    predecir_baseline,
)
from reddit_mental_health.preprocessing import (
    combinar_title_text,
    preprocesar_publicaciones,
)
from reddit_mental_health.splitting import separar_por_usuario


def parse_args() -> argparse.Namespace:
    """
    Lee argumentos CLI manteniendo defaults reproducibles.
    """

    config = BaselineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.input_path)
    parser.add_argument("--model-out", type=Path, default=config.model_path)
    parser.add_argument("--predictions-out", type=Path, default=config.predictions_path)
    parser.add_argument("--metrics-out", type=Path, default=config.metrics_path)
    parser.add_argument(
        "--interpretability-out",
        type=Path,
        default=config.interpretability_path,
    )
    parser.add_argument("--validation-size", type=float, default=config.validation_size)
    parser.add_argument("--random-state", type=int, default=config.random_state)
    return parser.parse_args()


def main() -> None:
    """
    Ejecuta el entrenamiento, validación y guardado de artefactos.
    """

    args = parse_args()
    config = BaselineConfig(
        input_path=args.input,
        model_path=args.model_out,
        predictions_path=args.predictions_out,
        metrics_path=args.metrics_out,
        interpretability_path=args.interpretability_out,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    datos = cargar_publicaciones_csv(config.input_path, config, require_label=True)
    entrenamiento, validacion = separar_por_usuario(datos, config)

    textos_entrenamiento = preprocesar_publicaciones(entrenamiento, config)
    textos_validacion = preprocesar_publicaciones(validacion, config)

    modelo = entrenar_baseline(
        textos_entrenamiento,
        entrenamiento[config.target_column],
        config,
    )
    y_pred, score = predecir_baseline(modelo, textos_validacion)

    predicciones = construir_salida_predicciones(
        validacion,
        y_pred,
        score,
        config,
        split="validacion",
        include_y_true=True,
    )
    metricas = calcular_metricas(
        predicciones["y_true"],
        predicciones["y_pred"],
        predicciones["score"],
        config,
    )
    reporte = construir_reporte_interpretabilidad(
        modelo,
        predicciones,
        combinar_title_text(validacion, config),
        config,
    )

    guardar_modelo(modelo, config.model_path)
    ensure_parent_dir(config.predictions_path)
    predicciones.to_csv(config.predictions_path, index=False)
    guardar_metricas(metricas, config.metrics_path)
    guardar_reporte_interpretabilidad(reporte, config.interpretability_path)

    print(f"Modelo guardado en: {config.model_path}")
    print(f"Predicciones guardadas en: {config.predictions_path}")
    print(f"Métricas guardadas en: {config.metrics_path}")


if __name__ == "__main__":
    main()
