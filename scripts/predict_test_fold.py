"""
Este módulo genera predicciones reproducibles para folds oficiales de prueba.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reddit_mental_health.config import BaselineConfig, PROJECT_ROOT, ensure_parent_dir
from reddit_mental_health.data import cargar_publicaciones_csv
from reddit_mental_health.evaluation import calcular_metricas, guardar_metricas
from reddit_mental_health.experiments import (
    construir_config_experimento,
    listar_clasificadores,
    listar_configuraciones_features,
)
from reddit_mental_health.model import (
    entrenar_baseline,
    guardar_modelo,
    predecir_baseline,
)
from reddit_mental_health.preprocessing import preprocesar_publicaciones


DEFAULT_TEST_INPUT = PROJECT_ROOT / "data" / "raw" / "data_test_fold1.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "test_folds"
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "data_test_fold1_predictions.csv"
DEFAULT_METRICS = DEFAULT_OUTPUT_DIR / "data_test_fold1_metrics.json"
DEFAULT_METADATA = DEFAULT_OUTPUT_DIR / "data_test_fold1_metadata.json"
DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "data_test_fold1_model.joblib"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "phase-2b-implementation" / "assets"
DEFAULT_DASHBOARD = (
    PROJECT_ROOT / "reports" / "phase-2b-implementation" / "test_fold1_dashboard.html"
)
ETHICAL_NOTE = (
    "Este flujo genera resultados experimentales del reto académico y no "
    "constituye una herramienta de diagnóstico clínico."
)
TEST_LABEL_NOTE = (
    "Las etiquetas del fold de prueba no se usan para entrenamiento, selección "
    "de modelo ni ajuste de hiperparámetros."
)
LABELS = {0: "no", 1: "yes"}
FIGURE_DPI = 160


def parse_args() -> argparse.Namespace:
    """
    Lee argumentos CLI para predicción y evaluación opcional del fold.
    """

    config = BaselineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=config.input_path)
    parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    parser.add_argument(
        "--classifier-name",
        choices=listar_clasificadores(),
        default="complement_nb",
    )
    parser.add_argument(
        "--feature-config-name",
        choices=listar_configuraciones_features(),
        default="char_wb_3_5",
    )
    parser.add_argument("--predictions-out", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--dashboard-out", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Omite guardar el modelo entrenado sobre construcción completa.",
    )
    parser.add_argument(
        "--evaluate-if-labeled",
        action="store_true",
        help="Calcula métricas solo si el fold incluye etiquetas.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Omite figuras y dashboard HTML del fold de prueba.",
    )
    return parser.parse_args()


def _timestamp_utc() -> str:
    """
    Genera un timestamp UTC serializable para metadata.
    """

    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _guardar_json(payload: dict[str, Any], path: str | Path) -> None:
    """
    Guarda un diccionario como JSON legible.
    """

    salida = Path(path)
    ensure_parent_dir(salida)
    salida.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _leer_columnas_crudas(path: Path, config: BaselineConfig) -> pd.DataFrame:
    """
    Carga columnas crudas para diagnosticar faltantes antes de fillna.
    """

    columnas = [config.user_column, config.title_column, config.text_column]
    if config.label_column:
        columnas.append(config.label_column)
    return pd.read_csv(path, usecols=lambda columna: columna in columnas)


def construir_predicciones_fold(
    test_data: pd.DataFrame,
    y_pred: list[int],
    score: list[float],
    config: BaselineConfig,
    include_y_true: bool,
) -> pd.DataFrame:
    """
    Construye la salida oficial de predicciones para el fold de prueba.
    """

    salida = test_data[[config.text_id_column, config.user_column]].copy()
    salida["y_pred"] = y_pred
    salida["label_pred"] = salida["y_pred"].map(LABELS)
    salida["score"] = score
    if include_y_true:
        salida["y_true"] = test_data[config.target_column].to_numpy()
    return salida


def diagnosticar_test_fold(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    raw_test_data: pd.DataFrame,
    labels_present: bool,
    config: BaselineConfig,
) -> dict[str, Any]:
    """
    Resume tamaño, faltantes, usuarios y etiquetas del fold de prueba.
    """

    usuarios_train = set(train_data[config.user_column])
    usuarios_test = set(test_data[config.user_column])
    diagnostico: dict[str, Any] = {
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "missing_title_values": int(raw_test_data[config.title_column].isna().sum()),
        "missing_text_values": int(raw_test_data[config.text_column].isna().sum()),
        "unique_train_users": int(train_data[config.user_column].nunique()),
        "unique_test_users": int(test_data[config.user_column].nunique()),
        "train_test_user_overlap_count": int(len(usuarios_train.intersection(usuarios_test))),
    }

    if labels_present and config.target_column in test_data.columns:
        etiquetas = test_data[config.target_column]
        diagnostico["test_label_distribution"] = {
            config.negative_label: int((etiquetas == config.negative_value).sum()),
            config.positive_label: int((etiquetas == config.positive_value).sum()),
        }
    return diagnostico


def construir_metadata(
    args: argparse.Namespace,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    raw_test_data: pd.DataFrame,
    labels_present: bool,
    labels_used_for_evaluation: bool,
    config: BaselineConfig,
) -> dict[str, Any]:
    """
    Construye metadata reproducible del flujo de predicción del fold.
    """

    metadata = {
        "timestamp": _timestamp_utc(),
        "classifier_name": config.classifier_name,
        "feature_config_name": config.feature_config_name,
        "train_input": str(args.train_input),
        "test_input": str(args.test_input),
        "predictions_out": str(args.predictions_out),
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "labels_present_in_test_fold": labels_present,
        "labels_used_for_evaluation": labels_used_for_evaluation,
        "ethical_note": ETHICAL_NOTE,
        "test_label_usage_note": TEST_LABEL_NOTE,
        "diagnostics": diagnosticar_test_fold(
            train_data,
            test_data,
            raw_test_data,
            labels_present,
            config,
        ),
    }
    if labels_used_for_evaluation:
        metadata["metrics_out"] = str(args.metrics_out)
    if not args.no_model:
        metadata["model_out"] = str(args.model_out)
    return metadata


def _cargar_pyplot() -> Any:
    """
    Carga matplotlib con backend no interactivo.
    """

    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "No se encontró matplotlib. Instala las dependencias del proyecto "
            "antes de generar el dashboard."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _guardar_figura(fig: Any, path: Path, plt: Any) -> Path:
    """
    Guarda una figura PNG para el dashboard del fold.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura generada: {path}")
    return path


def _formatear_metrica(valor: object) -> str:
    """
    Da formato compacto a métricas numéricas.
    """

    if valor is None or pd.isna(valor):
        return "N/D"
    try:
        return f"{float(valor):.4f}"
    except (TypeError, ValueError):
        return "N/D"


def _ruta_relativa(path: Path, dashboard_out: Path) -> str:
    """
    Convierte rutas locales a rutas relativas para HTML.
    """

    return html.escape(os.path.relpath(path, start=dashboard_out.parent))


def generar_figuras_fold(
    predicciones: pd.DataFrame,
    metricas: dict[str, Any] | None,
    figures_dir: Path,
) -> dict[str, Path]:
    """
    Genera figuras del fold de prueba según etiquetas disponibles.
    """

    plt = _cargar_pyplot()
    figuras: dict[str, Path] = {}

    conteos = predicciones["y_pred"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    barras = ax.bar([LABELS[0], LABELS[1]], conteos.to_numpy())
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Número de publicaciones")
    ax.set_title("Distribución de predicciones del fold")
    ax.bar_label(barras, labels=[str(valor) for valor in conteos.to_numpy()])
    ax.grid(True, axis="y", alpha=0.3)
    figuras["test_fold1_prediction_distribution.png"] = _guardar_figura(
        fig,
        figures_dir / "test_fold1_prediction_distribution.png",
        plt,
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.hist(pd.to_numeric(predicciones["score"], errors="coerce").dropna(), bins=20)
    ax.set_xlabel("Puntaje estimado para la clase yes")
    ax.set_ylabel("Número de publicaciones")
    ax.set_title("Distribución de puntajes del fold")
    ax.grid(True, alpha=0.3)
    figuras["test_fold1_score_distribution.png"] = _guardar_figura(
        fig,
        figures_dir / "test_fold1_score_distribution.png",
        plt,
    )

    if metricas and "y_true" in predicciones.columns:
        matriz = metricas.get("confusion_matrix", {})
        if isinstance(matriz, dict):
            valores = [
                [int(matriz["true_negative"]), int(matriz["false_positive"])],
                [int(matriz["false_negative"]), int(matriz["true_positive"])],
            ]
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            imagen = ax.imshow(valores)
            fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([0, 1], labels=[LABELS[0], LABELS[1]])
            ax.set_yticks([0, 1], labels=[LABELS[0], LABELS[1]])
            ax.set_xlabel("Etiqueta predicha")
            ax.set_ylabel("Etiqueta real")
            ax.set_title("Matriz de confusión del fold")
            umbral = max(max(fila) for fila in valores) / 2
            for fila in range(2):
                for columna in range(2):
                    color = "white" if valores[fila][columna] > umbral else "black"
                    ax.text(
                        columna,
                        fila,
                        str(valores[fila][columna]),
                        ha="center",
                        va="center",
                        color=color,
                    )
            figuras["test_fold1_confusion_matrix.png"] = _guardar_figura(
                fig,
                figures_dir / "test_fold1_confusion_matrix.png",
                plt,
            )

        if predicciones["y_true"].nunique() == 2:
            fpr, tpr, _ = roc_curve(predicciones["y_true"], predicciones["score"])
            fig, ax = plt.subplots(figsize=(5.8, 4.4))
            ax.plot(fpr, tpr, label=f"ROC AUC = {_formatear_metrica(metricas.get('roc_auc'))}")
            ax.plot([0, 1], [0, 1], linestyle="--", label="Referencia aleatoria")
            ax.set_xlabel("Tasa de falsos positivos")
            ax.set_ylabel("Tasa de verdaderos positivos")
            ax.set_title("Curva ROC del fold")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            figuras["test_fold1_roc_curve.png"] = _guardar_figura(
                fig,
                figures_dir / "test_fold1_roc_curve.png",
                plt,
            )

    return figuras


def generar_dashboard_fold(
    metadata: dict[str, Any],
    metricas: dict[str, Any] | None,
    figuras: dict[str, Path],
    dashboard_out: Path,
) -> Path:
    """
    Genera un dashboard HTML estático del fold de prueba.
    """

    dashboard_out.parent.mkdir(parents=True, exist_ok=True)
    diagnostico = metadata["diagnostics"]
    matriz = metricas.get("confusion_matrix") if metricas else None
    matriz_html = ""
    if isinstance(matriz, dict):
        matriz_html = f"""
        <section class="card">
          <h2>Matriz de confusión</h2>
          <div class="metrics">
            <article><span>TN</span><strong>{matriz.get("true_negative")}</strong></article>
            <article><span>FP</span><strong>{matriz.get("false_positive")}</strong></article>
            <article><span>FN</span><strong>{matriz.get("false_negative")}</strong></article>
            <article><span>TP</span><strong>{matriz.get("true_positive")}</strong></article>
          </div>
        </section>
        """

    metric_cards = [
        ("AUC protocolo", metricas.get("protocol_auc") if metricas else None),
        ("ROC AUC", metricas.get("roc_auc") if metricas else None),
        ("Recall", metricas.get("recall") if metricas else None),
        ("Precision", metricas.get("precision") if metricas else None),
        ("F1", metricas.get("f1") if metricas else None),
    ]
    metricas_html = "\n".join(
        f"<article><span>{html.escape(nombre)}</span>"
        f"<strong>{_formatear_metrica(valor)}</strong></article>"
        for nombre, valor in metric_cards
    )
    figuras_html = "\n".join(
        f"""
        <article class="plot-card">
          <h3>{html.escape(nombre.replace("_", " ").replace(".png", ""))}</h3>
          <img src="{_ruta_relativa(path, dashboard_out)}" alt="{html.escape(nombre)}">
        </article>
        """
        for nombre, path in figuras.items()
    )

    contenido = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard fold de prueba Fase 2B</title>
  <style>
    body {{
      margin: 0;
      padding: 28px;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #24272f;
      background: #f5f6f8;
      line-height: 1.5;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
    }}
    .card, .plot-card {{
      background: #fff;
      border: 1px solid #dfe3e8;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }}
    .metrics article {{
      background: #fff;
      border: 1px solid #dfe3e8;
      border-radius: 8px;
      padding: 14px;
    }}
    .metrics span {{
      display: block;
      color: #5c6470;
      font-size: 0.9rem;
    }}
    .metrics strong {{
      font-size: 1.45rem;
    }}
    .plots {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    img {{
      width: 100%;
      height: auto;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Dashboard fold de prueba Fase 2B</h1>
    <section class="card">
      <p>
        El modelo fue seleccionado previamente mediante validación interna.
        Las etiquetas del fold de prueba no se usan para entrenamiento,
        selección de modelo ni ajuste de hiperparámetros.
      </p>
    </section>
    <section class="card">
      <h2>Configuración</h2>
      <p><strong>Modelo:</strong> {html.escape(str(metadata["classifier_name"]))}</p>
      <p><strong>Features:</strong> {html.escape(str(metadata["feature_config_name"]))}</p>
      <p><strong>Filas train:</strong> {diagnostico["train_rows"]}</p>
      <p><strong>Filas test:</strong> {diagnostico["test_rows"]}</p>
      <p><strong>Etiquetas presentes:</strong> {metadata["labels_present_in_test_fold"]}</p>
      <p><strong>Etiquetas usadas para evaluación:</strong> {metadata["labels_used_for_evaluation"]}</p>
      <p><strong>Usuarios traslapados train/test:</strong> {diagnostico["train_test_user_overlap_count"]}</p>
    </section>
    <section class="card">
      <h2>Métricas</h2>
      <div class="metrics">{metricas_html}</div>
    </section>
    {matriz_html}
    <section>
      <h2>Visualizaciones</h2>
      <div class="plots">{figuras_html}</div>
    </section>
  </main>
</body>
</html>
"""
    dashboard_out.write_text(contenido, encoding="utf-8")
    print(f"Dashboard generado: {dashboard_out}")
    return dashboard_out


def ejecutar_prediccion(args: argparse.Namespace) -> dict[str, Any]:
    """
    Entrena con construcción completa y predice el fold oficial.
    """

    base_config = BaselineConfig(
        input_path=args.train_input,
        model_path=args.model_out,
        predictions_path=args.predictions_out,
        metrics_path=args.metrics_out,
    )
    config = construir_config_experimento(
        base_config,
        classifier_name=args.classifier_name,
        feature_config_name=args.feature_config_name,
    )

    train_data = cargar_publicaciones_csv(args.train_input, config, require_label=True)
    raw_test_data = _leer_columnas_crudas(args.test_input, config)
    labels_present = config.label_column in raw_test_data.columns
    if labels_present:
        print(
            "Advertencia: el fold de prueba contiene etiquetas. "
            "No se usarán para entrenamiento, selección ni predicción.",
            file=sys.stderr,
        )
    if args.evaluate_if_labeled and not labels_present:
        raise SystemExit(
            "--evaluate-if-labeled fue solicitado, pero el fold no contiene etiquetas.",
        )

    test_data = cargar_publicaciones_csv(args.test_input, config, require_label=False)
    textos_train = preprocesar_publicaciones(train_data, config)
    textos_test = preprocesar_publicaciones(test_data, config)
    modelo = entrenar_baseline(textos_train, train_data[config.target_column], config)
    y_pred, score = predecir_baseline(modelo, textos_test)

    usar_etiquetas_eval = bool(args.evaluate_if_labeled and labels_present)
    predicciones = construir_predicciones_fold(
        test_data,
        y_pred,
        score,
        config,
        include_y_true=usar_etiquetas_eval,
    )

    metricas = None
    if usar_etiquetas_eval:
        metricas = calcular_metricas(
            predicciones["y_true"],
            predicciones["y_pred"],
            predicciones["score"],
            config,
        )
        guardar_metricas(metricas, args.metrics_out)

    ensure_parent_dir(args.predictions_out)
    predicciones.to_csv(args.predictions_out, index=False)
    if not args.no_model:
        guardar_modelo(modelo, args.model_out)

    metadata = construir_metadata(
        args,
        train_data,
        test_data,
        raw_test_data,
        labels_present,
        usar_etiquetas_eval,
        config,
    )
    _guardar_json(metadata, args.metadata_out)

    figuras = {}
    if not args.no_dashboard:
        figuras = generar_figuras_fold(predicciones, metricas, args.figures_dir)
        generar_dashboard_fold(metadata, metricas, figuras, args.dashboard_out)

    print(f"Predicciones guardadas en: {args.predictions_out}")
    if metricas is not None:
        print(f"Métricas guardadas en: {args.metrics_out}")
    print(f"Metadata guardada en: {args.metadata_out}")
    return {"metadata": metadata, "metrics": metricas, "figures": figuras}


def main() -> None:
    """
    Ejecuta el flujo CLI de predicción del fold oficial.
    """

    args = parse_args()
    ejecutar_prediccion(args)


if __name__ == "__main__":
    main()
