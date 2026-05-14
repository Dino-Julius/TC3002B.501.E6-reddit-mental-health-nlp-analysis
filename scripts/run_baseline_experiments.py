"""Ejecuta una corrida baseline con tracking local de artefactos."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.data import (
    cargar_publicaciones_csv,
    construir_salida_predicciones,
)
from reddit_mental_health.evaluation import calcular_metricas
from reddit_mental_health.interpretability import construir_reporte_interpretabilidad
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
from reddit_mental_health.tracking import (
    RunMetadata,
    actualizar_resumen,
    construir_run_id,
    construir_rutas_corrida,
    guardar_json,
    guardar_metadata,
    timestamp_utc,
)


def parse_args() -> argparse.Namespace:
    """Lee argumentos CLI para una corrida con tracking."""

    config = BaselineConfig()
    default_output = (
        config.project_root
        / "data"
        / "processed"
        / "experiments"
        / "phase-2b-implementation"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.input_path)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiment-name", default="phase-2b-implementation")
    parser.add_argument("--classifier-name", default="logistic_regression")
    parser.add_argument("--feature-config-name", default="word_unigram_bigram")
    parser.add_argument("--validation-size", type=float, default=config.validation_size)
    parser.add_argument("--random-state", type=int, default=config.random_state)
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Omite guardar model.joblib para corridas rápidas.",
    )
    return parser.parse_args()


def _crear_metadata(
    args: argparse.Namespace,
    config: BaselineConfig,
    run_id: str,
    started_at: str,
    status: str,
    duration_seconds: float | None,
    error: str | None = None,
) -> RunMetadata:
    """Construye metadata consistente para éxito o falla."""

    paths = construir_rutas_corrida(args.output_dir, run_id)
    completed_at = timestamp_utc() if status != "running" else None
    return RunMetadata(
        run_id=run_id,
        experiment_name=args.experiment_name,
        classifier_name=args.classifier_name,
        feature_config_name=args.feature_config_name,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        input_path=str(config.input_path),
        random_state=config.random_state,
        validation_size=config.validation_size,
        metrics_path=str(paths.metrics_path),
        predictions_path=str(paths.predictions_path),
        interpretability_path=str(paths.interpretability_path),
        model_path=None if args.no_model else str(paths.model_path),
        error=error,
    )


def ejecutar_corrida(args: argparse.Namespace) -> RunMetadata:
    """Entrena el baseline actual y guarda artefactos versionados por run_id."""

    started_at = timestamp_utc()
    run_id = args.run_id or construir_run_id(
        args.experiment_name,
        args.classifier_name,
        args.feature_config_name,
        started_at=started_at,
    )
    paths = construir_rutas_corrida(args.output_dir, run_id)
    config = BaselineConfig(
        input_path=args.input,
        model_path=paths.model_path,
        predictions_path=paths.predictions_path,
        metrics_path=paths.metrics_path,
        interpretability_path=paths.interpretability_path,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    inicio = time.perf_counter()
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    try:
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

        predicciones.to_csv(paths.predictions_path, index=False)
        guardar_json(metricas, paths.metrics_path)
        guardar_json(reporte, paths.interpretability_path)
        if not args.no_model:
            guardar_modelo(modelo, paths.model_path)

        metadata = _crear_metadata(
            args,
            config,
            run_id,
            started_at,
            status="completed",
            duration_seconds=round(time.perf_counter() - inicio, 4),
        )
        guardar_metadata(metadata, paths.metadata_path)
        actualizar_resumen(args.output_dir)
        return metadata
    except Exception as exc:
        metadata = _crear_metadata(
            args,
            config,
            run_id,
            started_at,
            status="failed",
            duration_seconds=round(time.perf_counter() - inicio, 4),
            error=str(exc),
        )
        guardar_metadata(metadata, paths.metadata_path)
        actualizar_resumen(args.output_dir)
        raise


def main() -> None:
    """Ejecuta una corrida y reporta sus rutas principales."""

    args = parse_args()
    metadata = ejecutar_corrida(args)
    print(f"Run guardado: {metadata.run_id}")
    print(f"Métricas: {metadata.metrics_path}")
    print(f"Predicciones: {metadata.predictions_path}")
    print(f"Resumen: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
