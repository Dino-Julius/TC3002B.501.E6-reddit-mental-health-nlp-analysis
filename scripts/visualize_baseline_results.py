"""Genera visualizaciones y dashboard estático del baseline Fase 2B."""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = PROJECT_ROOT / "data" / "processed" / "baseline_metrics.json"
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "data" / "processed" / "baseline_validation_predictions.csv"
)
DEFAULT_INTERPRETABILITY = (
    PROJECT_ROOT / "data" / "processed" / "baseline_interpretability.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase-2b-baseline" / "assets"
DEFAULT_DASHBOARD = PROJECT_ROOT / "reports" / "phase-2b-baseline" / "dashboard.html"

LABELS = {0: "no", 1: "yes"}
FIGURE_DPI = 160
ETHICAL_NOTE = (
    "Este dashboard resume resultados experimentales del reto académico y no "
    "constituye una herramienta de diagnóstico clínico."
)


def parse_args() -> argparse.Namespace:
    """Lee argumentos CLI para construir figuras y dashboard local."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--interpretability", type=Path, default=DEFAULT_INTERPRETABILITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dashboard-out", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Genera solo las figuras PNG y omite el dashboard HTML.",
    )
    return parser.parse_args()


def _validar_archivo_requerido(path: Path, descripcion: str) -> None:
    """Detiene la ejecución si falta un insumo obligatorio."""

    if not path.exists():
        raise SystemExit(f"No se encontró {descripcion}: {path}")


def _leer_json(path: Path) -> dict[str, object]:
    """Carga un archivo JSON como diccionario."""

    return json.loads(path.read_text(encoding="utf-8"))


def _leer_interpretabilidad(path: Path) -> dict[str, object] | None:
    """Carga la interpretabilidad si existe; si no, continúa sin fallar."""

    if not path.exists():
        print(f"Reporte interpretativo opcional no encontrado; se omite: {path}")
        return None
    return _leer_json(path)


def _cargar_pyplot() -> Any:
    """Carga matplotlib con backend no interactivo solo cuando se generan figuras."""

    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "No se encontró matplotlib. Instala las dependencias del proyecto "
            "antes de generar las visualizaciones."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _guardar_figura(fig: Any, path: Path, plt: Any) -> Path:
    """Guarda una figura con parámetros consistentes para reporte."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura generada: {path}")
    return path


def _serie_binaria(serie: pd.Series) -> pd.Series:
    """Normaliza etiquetas binarias numéricas o textuales a 0/1."""

    numerica = pd.to_numeric(serie, errors="coerce")
    textual = serie.map(
        lambda valor: str(valor).strip().lower() if pd.notna(valor) else valor
    )
    mapeada = textual.map({"no": 0, "yes": 1, "0": 0, "1": 1})
    return numerica.where(numerica.notna(), mapeada)


def _formatear_metrica(valor: object) -> str:
    """Da formato corto a métricas numéricas del dashboard."""

    if valor is None:
        return "N/D"
    try:
        return f"{float(valor):.4f}"
    except (TypeError, ValueError):
        return "N/D"


def _obtener_matriz_confusion(metricas: dict[str, object]) -> np.ndarray:
    """Extrae la matriz de confusión en orden no/sí desde las métricas."""

    matriz = metricas.get("confusion_matrix")
    if not isinstance(matriz, dict):
        raise SystemExit("El JSON de métricas no contiene 'confusion_matrix'.")

    claves = [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]
    faltantes = [clave for clave in claves if clave not in matriz]
    if faltantes:
        raise SystemExit(
            "La matriz de confusión no contiene las claves requeridas: "
            + ", ".join(faltantes)
        )

    return np.array(
        [
            [int(matriz["true_negative"]), int(matriz["false_positive"])],
            [int(matriz["false_negative"]), int(matriz["true_positive"])],
        ]
    )


def graficar_matriz_confusion(
    metricas: dict[str, object],
    output_dir: Path,
    plt: Any,
) -> Path:
    """Genera la figura de matriz de confusión del baseline."""

    matriz = _obtener_matriz_confusion(metricas)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    imagen = ax.imshow(matriz)
    fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1], labels=[LABELS[0], LABELS[1]])
    ax.set_yticks([0, 1], labels=[LABELS[0], LABELS[1]])
    ax.set_xlabel("Etiqueta predicha")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusión del baseline")

    umbral = matriz.max() / 2
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            color_texto = "white" if matriz[fila, columna] > umbral else "black"
            ax.text(
                columna,
                fila,
                str(matriz[fila, columna]),
                ha="center",
                va="center",
                color=color_texto,
                fontsize=12,
            )

    return _guardar_figura(fig, output_dir / "confusion_matrix.png", plt)


def graficar_curva_roc(
    predicciones: pd.DataFrame,
    metricas: dict[str, object],
    output_dir: Path,
    plt: Any,
) -> Path | None:
    """Genera la curva ROC cuando existen etiquetas y puntajes suficientes."""

    if "y_true" not in predicciones.columns or "score" not in predicciones.columns:
        print("Se omite roc_curve.png: faltan columnas y_true o score.")
        return None

    datos = pd.DataFrame(
        {
            "y_true": _serie_binaria(predicciones["y_true"]),
            "score": pd.to_numeric(predicciones["score"], errors="coerce"),
        }
    ).dropna()

    if datos["y_true"].nunique() < 2:
        print("Se omite roc_curve.png: las etiquetas no contienen ambas clases.")
        return None

    fpr, tpr, _ = roc_curve(datos["y_true"].astype(int), datos["score"])
    roc_auc = metricas.get("roc_auc")
    etiqueta = f"ROC AUC = {_formatear_metrica(roc_auc)}"

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.plot(fpr, tpr, label=etiqueta)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Referencia aleatoria")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC del baseline")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return _guardar_figura(fig, output_dir / "roc_curve.png", plt)


def graficar_distribucion_scores(
    predicciones: pd.DataFrame,
    output_dir: Path,
    plt: Any,
) -> Path | None:
    """Genera la distribución de puntajes, agrupada por etiqueta si existe."""

    if "score" not in predicciones.columns:
        print("Se omite score_distribution.png: falta la columna score.")
        return None

    score = pd.to_numeric(predicciones["score"], errors="coerce")
    fig, ax = plt.subplots(figsize=(5.8, 4.4))

    if "y_true" in predicciones.columns:
        datos = pd.DataFrame(
            {"score": score, "y_true": _serie_binaria(predicciones["y_true"])}
        ).dropna()
        if not datos.empty:
            for valor, etiqueta in LABELS.items():
                grupo = datos.loc[datos["y_true"] == valor, "score"]
                if not grupo.empty:
                    ax.hist(grupo, bins=20, alpha=0.65, label=etiqueta)
            ax.legend(title="Etiqueta real")
            ax.set_title("Distribución de puntajes por etiqueta real")
        else:
            ax.hist(score.dropna(), bins=20)
            ax.set_title("Distribución general de puntajes")
    else:
        ax.hist(score.dropna(), bins=20)
        ax.set_title("Distribución general de puntajes")

    ax.set_xlabel("Puntaje estimado para la clase yes")
    ax.set_ylabel("Número de publicaciones")
    ax.grid(True, alpha=0.3)

    return _guardar_figura(fig, output_dir / "score_distribution.png", plt)


def graficar_distribucion_predicciones(
    predicciones: pd.DataFrame,
    output_dir: Path,
    plt: Any,
) -> Path | None:
    """Genera el conteo de clases predichas si y_pred está disponible."""

    if "y_pred" not in predicciones.columns:
        print("Se omite prediction_distribution.png: falta la columna y_pred.")
        return None

    y_pred = _serie_binaria(predicciones["y_pred"]).dropna().astype(int)
    conteos = y_pred.value_counts().reindex([0, 1], fill_value=0)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    barras = ax.bar([LABELS[0], LABELS[1]], conteos.to_numpy())
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Número de publicaciones")
    ax.set_title("Distribución de predicciones del baseline")
    ax.bar_label(barras, labels=[str(valor) for valor in conteos.to_numpy()])
    ax.grid(True, axis="y", alpha=0.3)

    return _guardar_figura(fig, output_dir / "prediction_distribution.png", plt)


def generar_figuras(
    metricas: dict[str, object],
    predicciones: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Construye todas las visualizaciones disponibles del baseline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _cargar_pyplot()
    figuras: dict[str, Path] = {}

    figuras["confusion_matrix.png"] = graficar_matriz_confusion(
        metricas,
        output_dir,
        plt,
    )

    roc_path = graficar_curva_roc(predicciones, metricas, output_dir, plt)
    if roc_path is not None:
        figuras["roc_curve.png"] = roc_path

    score_path = graficar_distribucion_scores(predicciones, output_dir, plt)
    if score_path is not None:
        figuras["score_distribution.png"] = score_path

    pred_path = graficar_distribucion_predicciones(predicciones, output_dir, plt)
    if pred_path is not None:
        figuras["prediction_distribution.png"] = pred_path

    return figuras


def _ruta_relativa(path: Path, dashboard_out: Path) -> str:
    """Convierte una ruta de figura a una ruta relativa para HTML local."""

    relativa = os.path.relpath(path, start=dashboard_out.parent)
    return Path(relativa).as_posix()


def _resumen_matriz_html(metricas: dict[str, object]) -> str:
    """Construye el resumen HTML de TN, FP, FN y TP."""

    matriz = metricas.get("confusion_matrix")
    if not isinstance(matriz, dict):
        return "<p>No se encontró resumen de matriz de confusión.</p>"

    elementos = [
        ("TN", matriz.get("true_negative")),
        ("FP", matriz.get("false_positive")),
        ("FN", matriz.get("false_negative")),
        ("TP", matriz.get("true_positive")),
    ]
    return "\n".join(
        f"<div class=\"mini-card\"><span>{nombre}</span><strong>{valor}</strong></div>"
        for nombre, valor in elementos
    )


def _terminos_html(interpretabilidad: dict[str, object] | None) -> str:
    """Genera una sección breve con términos positivos y negativos."""

    if not interpretabilidad:
        return ""

    pesos = interpretabilidad.get("pesos_caracteristicas")
    if not isinstance(pesos, dict):
        return ""

    positivos = pesos.get("terminos_asociados_a_yes")
    negativos = pesos.get("terminos_asociados_a_no")
    if not isinstance(positivos, list) and not isinstance(negativos, list):
        return ""

    def lista_terminos(items: object) -> str:
        if not isinstance(items, list):
            return "<li>No disponible</li>"
        filas = []
        for item in items[:10]:
            if not isinstance(item, dict):
                continue
            termino = html.escape(str(item.get("termino", "")))
            peso = _formatear_metrica(item.get("peso"))
            filas.append(f"<li><span>{termino}</span><small>{peso}</small></li>")
        return "\n".join(filas) or "<li>No disponible</li>"

    return f"""
    <section>
      <h2>Interpretabilidad inicial</h2>
      <p>Los términos se derivan de los pesos del clasificador lineal y deben
      leerse como evidencia exploratoria, no como explicación clínica.</p>
      <div class="terms-grid">
        <article>
          <h3>Términos asociados a yes</h3>
          <ol>{lista_terminos(positivos)}</ol>
        </article>
        <article>
          <h3>Términos asociados a no</h3>
          <ol>{lista_terminos(negativos)}</ol>
        </article>
      </div>
    </section>
    """


def _figura_html(
    nombre: str,
    titulo: str,
    explicacion: str,
    figuras: dict[str, Path],
    dashboard_out: Path,
) -> str:
    """Renderiza una tarjeta de figura o un aviso si fue omitida."""

    path = figuras.get(nombre)
    if path is None:
        imagen = "<p class=\"missing\">Visualización no generada para este insumo.</p>"
    else:
        src = html.escape(_ruta_relativa(path, dashboard_out))
        alt = html.escape(titulo)
        imagen = f"<img src=\"{src}\" alt=\"{alt}\">"

    return f"""
    <article class="plot-card">
      <h3>{html.escape(titulo)}</h3>
      {imagen}
      <p>{html.escape(explicacion)}</p>
    </article>
    """


def generar_dashboard(
    metricas: dict[str, object],
    figuras: dict[str, Path],
    interpretabilidad: dict[str, object] | None,
    dashboard_out: Path,
) -> Path:
    """Escribe un dashboard HTML estático con métricas y figuras."""

    dashboard_out.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    metric_cards = [
        ("ROC AUC", metricas.get("roc_auc")),
        ("Recall", metricas.get("recall")),
        ("Precision", metricas.get("precision")),
        ("F1", metricas.get("f1")),
    ]
    metricas_html = "\n".join(
        f"<article class=\"metric-card\"><span>{nombre}</span>"
        f"<strong>{_formatear_metrica(valor)}</strong></article>"
        for nombre, valor in metric_cards
    )

    plots_html = "\n".join(
        [
            _figura_html(
                "confusion_matrix.png",
                "Matriz de confusión",
                "Resume aciertos y errores separando verdaderos negativos, "
                "falsos positivos, falsos negativos y verdaderos positivos.",
                figuras,
                dashboard_out,
            ),
            _figura_html(
                "roc_curve.png",
                "Curva ROC",
                "Muestra la relación entre sensibilidad y falsos positivos al "
                "variar el umbral de decisión.",
                figuras,
                dashboard_out,
            ),
            _figura_html(
                "score_distribution.png",
                "Distribución de puntajes",
                "Permite revisar cómo se distribuyen las probabilidades estimadas "
                "para la clase yes.",
                figuras,
                dashboard_out,
            ),
            _figura_html(
                "prediction_distribution.png",
                "Distribución de predicciones",
                "Indica cuántas publicaciones fueron clasificadas como no o yes "
                "por el baseline.",
                figuras,
                dashboard_out,
            ),
        ]
    )

    contenido = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard del baseline Fase 2B</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
      color: #24272f;
      background: #f5f6f8;
    }}
    body {{
      margin: 0;
      padding: 32px;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
    }}
    header {{
      margin-bottom: 28px;
    }}
    h1, h2, h3, p {{
      margin-top: 0;
    }}
    h1 {{
      font-size: 2rem;
      margin-bottom: 8px;
    }}
    h2 {{
      margin-top: 32px;
      margin-bottom: 12px;
    }}
    .timestamp {{
      color: #5c6470;
      font-size: 0.95rem;
    }}
    .metrics-grid, .matrix-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .metric-card, .mini-card, .plot-card, .terms-grid article, .note {{
      background: #ffffff;
      border: 1px solid #dfe3e8;
      border-radius: 8px;
      padding: 16px;
    }}
    .metric-card span, .mini-card span {{
      display: block;
      color: #5c6470;
      font-size: 0.9rem;
      margin-bottom: 4px;
    }}
    .metric-card strong, .mini-card strong {{
      font-size: 1.55rem;
    }}
    .plots-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .plot-card img {{
      width: 100%;
      height: auto;
      display: block;
      margin: 8px 0 12px;
    }}
    .missing {{
      border: 1px dashed #a7afb9;
      border-radius: 6px;
      padding: 24px;
      color: #5c6470;
    }}
    .terms-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}
    ol {{
      padding-left: 20px;
      margin-bottom: 0;
    }}
    li {{
      margin-bottom: 6px;
    }}
    li small {{
      color: #5c6470;
      margin-left: 8px;
    }}
    .note {{
      margin-top: 24px;
      border-left: 4px solid #4f5967;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Dashboard del baseline Fase 2B</h1>
      <p class="timestamp">Generado: {html.escape(timestamp)}</p>
    </header>

    <section>
      <h2>Métricas de validación</h2>
      <div class="metrics-grid">
        {metricas_html}
      </div>
    </section>

    <section>
      <h2>Resumen de matriz de confusión</h2>
      <div class="matrix-grid">
        {_resumen_matriz_html(metricas)}
      </div>
    </section>

    <section>
      <h2>Visualizaciones</h2>
      <div class="plots-grid">
        {plots_html}
      </div>
    </section>

    {_terminos_html(interpretabilidad)}

    <section class="note">
      <h2>Nota ética</h2>
      <p>{ETHICAL_NOTE}</p>
    </section>
  </main>
</body>
</html>
"""
    dashboard_out.write_text(contenido, encoding="utf-8")
    print(f"Dashboard generado: {dashboard_out}")
    return dashboard_out


def main() -> None:
    """Coordina la lectura de resultados y la generación de reportes."""

    args = parse_args()
    _validar_archivo_requerido(args.metrics, "el archivo de métricas")
    _validar_archivo_requerido(args.predictions, "el archivo de predicciones")

    metricas = _leer_json(args.metrics)
    predicciones = pd.read_csv(args.predictions)
    interpretabilidad = _leer_interpretabilidad(args.interpretability)

    figuras = generar_figuras(metricas, predicciones, args.output_dir)
    if not args.no_dashboard:
        generar_dashboard(metricas, figuras, interpretabilidad, args.dashboard_out)


if __name__ == "__main__":
    main()
