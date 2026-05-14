"""Genera un dashboard comparativo para experimentos Fase 2B."""

from __future__ import annotations

import argparse
import html
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experiments"
    / "phase-2b-implementation"
    / "summary.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase-2b-implementation" / "assets"
DEFAULT_DASHBOARD = (
    PROJECT_ROOT / "reports" / "phase-2b-implementation" / "comparison.html"
)
FIGURE_DPI = 160
REQUIRED_COLUMNS = {
    "run_id",
    "status",
    "classifier_name",
    "feature_config_name",
    "started_at",
    "roc_auc",
    "protocol_auc",
    "recall",
    "precision",
    "f1",
}
METRIC_COLUMNS = [
    "protocol_auc",
    "roc_auc",
    "recall",
    "precision",
    "f1",
]


def parse_args() -> argparse.Namespace:
    """Lee argumentos CLI para el dashboard comparativo."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dashboard-out", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Número máximo de corridas mostradas en rankings y tablas.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Genera solo las figuras PNG y omite el dashboard HTML.",
    )
    return parser.parse_args()


def _cargar_pyplot() -> Any:
    """Carga matplotlib con backend no interactivo."""

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
    """Guarda una figura de reporte y cierra el objeto."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura generada: {path}")
    return path


def _formatear_metrica(valor: object) -> str:
    """Da formato compacto a métricas numéricas."""

    if valor is None or pd.isna(valor):
        return "N/D"
    try:
        return f"{float(valor):.4f}"
    except (TypeError, ValueError):
        return "N/D"


def _ruta_relativa(path: Path, dashboard_out: Path) -> str:
    """Convierte una ruta local a ruta relativa para HTML estático."""

    return html.escape(os.path.relpath(path, start=dashboard_out.parent))


def cargar_resumen(summary_path: Path) -> pd.DataFrame:
    """Carga y valida el resumen agregado de experimentos."""

    if not summary_path.exists():
        raise SystemExit(f"No se encontró el resumen de experimentos: {summary_path}")

    resumen = pd.read_csv(summary_path)
    faltantes = sorted(REQUIRED_COLUMNS.difference(resumen.columns))
    if faltantes:
        raise SystemExit(
            "El resumen no contiene las columnas requeridas: " + ", ".join(faltantes),
        )

    for columna in METRIC_COLUMNS:
        resumen[columna] = pd.to_numeric(resumen[columna], errors="coerce")
    resumen["started_at_sort"] = pd.to_datetime(
        resumen["started_at"],
        errors="coerce",
        utc=True,
    )
    return resumen


def preparar_corridas_comparables(resumen: pd.DataFrame) -> pd.DataFrame:
    """Filtra corridas completas y conserva la última por combinación."""

    completas = resumen.loc[resumen["status"] == "completed"].copy()
    completas = completas.dropna(subset=["protocol_auc"])
    if completas.empty:
        raise SystemExit("No hay corridas completadas con protocol_auc para comparar.")

    completas = completas.sort_values(["started_at_sort", "run_id"])
    comparables = completas.drop_duplicates(
        subset=["classifier_name", "feature_config_name"],
        keep="last",
    )
    return comparables.sort_values("protocol_auc", ascending=False).reset_index(
        drop=True,
    )


def _etiqueta_corrida(row: pd.Series) -> str:
    """Construye una etiqueta corta para rankings."""

    return f"{row['classifier_name']} | {row['feature_config_name']}"


def graficar_ranking_protocol_auc(
    corridas: pd.DataFrame,
    output_dir: Path,
    plt: Any,
    top_n: int,
) -> Path:
    """Grafica el ranking de corridas por AUC protocolo."""

    datos = corridas.head(top_n).iloc[::-1]
    etiquetas = [_etiqueta_corrida(row) for _, row in datos.iterrows()]

    fig, ax = plt.subplots(figsize=(9.5, max(4.8, len(datos) * 0.34)))
    barras = ax.barh(etiquetas, datos["protocol_auc"])
    ax.set_xlabel("AUC protocolo")
    ax.set_title("Ranking experimental por AUC protocolo")
    ax.set_xlim(0, max(1.0, float(datos["protocol_auc"].max()) + 0.05))
    ax.grid(True, axis="x", alpha=0.3)
    ax.bar_label(
        barras,
        labels=[_formatear_metrica(valor) for valor in datos["protocol_auc"]],
        padding=3,
    )

    return _guardar_figura(
        fig,
        output_dir / "experiment_protocol_auc_ranking.png",
        plt,
    )


def graficar_heatmap_protocol_auc(
    corridas: pd.DataFrame,
    output_dir: Path,
    plt: Any,
) -> Path:
    """Grafica una matriz clasificador/features con AUC protocolo."""

    tabla = corridas.pivot_table(
        index="classifier_name",
        columns="feature_config_name",
        values="protocol_auc",
        aggfunc="max",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    imagen = ax.imshow(tabla.to_numpy(), aspect="auto")
    fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04, label="AUC protocolo")

    ax.set_xticks(range(len(tabla.columns)), labels=tabla.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(tabla.index)), labels=tabla.index)
    ax.set_xlabel("Configuración de features")
    ax.set_ylabel("Clasificador")
    ax.set_title("AUC protocolo por clasificador y features")

    for fila, classifier_name in enumerate(tabla.index):
        for columna, feature_name in enumerate(tabla.columns):
            valor = tabla.loc[classifier_name, feature_name]
            if pd.notna(valor):
                ax.text(
                    columna,
                    fila,
                    _formatear_metrica(valor),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    return _guardar_figura(
        fig,
        output_dir / "experiment_protocol_auc_heatmap.png",
        plt,
    )


def graficar_metricas_top(
    corridas: pd.DataFrame,
    output_dir: Path,
    plt: Any,
    top_n: int,
) -> Path:
    """Compara métricas principales para las mejores corridas."""

    datos = corridas.head(min(top_n, 10)).copy()
    etiquetas = [_etiqueta_corrida(row) for _, row in datos.iterrows()]
    x = range(len(datos))
    ancho = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(
        [valor - ancho for valor in x],
        datos["protocol_auc"],
        width=ancho,
        label="AUC protocolo",
    )
    ax.bar(x, datos["roc_auc"], width=ancho, label="ROC AUC")
    ax.bar(
        [valor + ancho for valor in x],
        datos["f1"],
        width=ancho,
        label="F1",
    )
    ax.set_xticks(list(x), labels=etiquetas, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor de métrica")
    ax.set_title("Comparación de métricas en mejores corridas")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    return _guardar_figura(fig, output_dir / "experiment_top_metrics.png", plt)


def graficar_roc_vs_protocol_auc(
    corridas: pd.DataFrame,
    output_dir: Path,
    plt: Any,
) -> Path:
    """Compara ROC AUC de scores contra AUC protocolo de predicciones finales."""

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for classifier_name, grupo in corridas.groupby("classifier_name"):
        ax.scatter(
            grupo["roc_auc"],
            grupo["protocol_auc"],
            label=classifier_name,
            alpha=0.85,
        )
    ax.set_xlabel("ROC AUC (scores)")
    ax.set_ylabel("AUC protocolo")
    ax.set_title("ROC AUC vs AUC protocolo")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    return _guardar_figura(fig, output_dir / "experiment_roc_vs_protocol_auc.png", plt)


def generar_figuras(
    corridas: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> dict[str, Path]:
    """Genera todas las figuras comparativas disponibles."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _cargar_pyplot()
    figuras = {
        "experiment_protocol_auc_ranking.png": graficar_ranking_protocol_auc(
            corridas,
            output_dir,
            plt,
            top_n,
        ),
        "experiment_protocol_auc_heatmap.png": graficar_heatmap_protocol_auc(
            corridas,
            output_dir,
            plt,
        ),
        "experiment_top_metrics.png": graficar_metricas_top(
            corridas,
            output_dir,
            plt,
            top_n,
        ),
        "experiment_roc_vs_protocol_auc.png": graficar_roc_vs_protocol_auc(
            corridas,
            output_dir,
            plt,
        ),
    }
    return figuras


def _render_metric_card(label: str, value: object) -> str:
    """Renderiza una tarjeta de métrica para el dashboard."""

    return f"""
      <article class="card metric-card">
        <span>{html.escape(label)}</span>
        <strong>{html.escape(str(value))}</strong>
      </article>
    """


def _render_top_table(corridas: pd.DataFrame, top_n: int) -> str:
    """Renderiza la tabla de mejores corridas."""

    filas = []
    for indice, (_, row) in enumerate(corridas.head(top_n).iterrows(), start=1):
        filas.append(
            "<tr>"
            f"<td>{indice}</td>"
            f"<td>{html.escape(str(row['classifier_name']))}</td>"
            f"<td>{html.escape(str(row['feature_config_name']))}</td>"
            f"<td>{_formatear_metrica(row['protocol_auc'])}</td>"
            f"<td>{_formatear_metrica(row['roc_auc'])}</td>"
            f"<td>{_formatear_metrica(row['recall'])}</td>"
            f"<td>{_formatear_metrica(row['precision'])}</td>"
            f"<td>{_formatear_metrica(row['f1'])}</td>"
            "</tr>"
        )
    return "\n".join(filas)


def generar_dashboard(
    resumen: pd.DataFrame,
    corridas: pd.DataFrame,
    figuras: dict[str, Path],
    dashboard_out: Path,
    summary_path: Path,
    top_n: int,
) -> Path:
    """Genera el HTML estático del comparativo experimental."""

    dashboard_out.parent.mkdir(parents=True, exist_ok=True)
    mejor = corridas.iloc[0]
    total_corridas = len(resumen)
    corridas_completas = int((resumen["status"] == "completed").sum())
    combinaciones = len(corridas)
    generacion = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    cards = "\n".join(
        [
            _render_metric_card("Corridas registradas", total_corridas),
            _render_metric_card("Corridas completas", corridas_completas),
            _render_metric_card("Combinaciones comparadas", combinaciones),
            _render_metric_card("Mejor AUC protocolo", _formatear_metrica(mejor["protocol_auc"])),
            _render_metric_card("ROC AUC del mejor", _formatear_metrica(mejor["roc_auc"])),
        ]
    )

    image_sections = []
    explicaciones = {
        "experiment_protocol_auc_ranking.png": (
            "Ordena las combinaciones por AUC protocolo, la métrica alineada "
            "a la fórmula del curso basada en TPR y FPR."
        ),
        "experiment_protocol_auc_heatmap.png": (
            "Muestra qué combinaciones de clasificador y features concentran "
            "mejor desempeño bajo AUC protocolo."
        ),
        "experiment_top_metrics.png": (
            "Compara AUC protocolo, ROC AUC y F1 en las mejores corridas para "
            "detectar posibles trade-offs."
        ),
        "experiment_roc_vs_protocol_auc.png": (
            "Contrasta ROC AUC basado en scores continuos con AUC protocolo "
            "calculado sobre predicciones finales."
        ),
    }
    for nombre, path in figuras.items():
        image_sections.append(
            f"""
      <section class="plot-card">
        <h2>{html.escape(nombre.replace("_", " ").replace(".png", ""))}</h2>
        <p>{html.escape(explicaciones.get(nombre, ""))}</p>
        <img src="{_ruta_relativa(path, dashboard_out)}" alt="{html.escape(nombre)}" />
      </section>
            """
        )

    try:
        fuente = str(summary_path.relative_to(PROJECT_ROOT))
    except ValueError:
        fuente = str(summary_path)

    html_text = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Comparativo de experimentos Fase 2B</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #202124;
      background: #f6f7f9;
    }}
    body {{
      margin: 0;
      line-height: 1.5;
    }}
    header, main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px;
    }}
    header {{
      padding-bottom: 12px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    p {{
      margin: 0 0 14px;
    }}
    .subtle {{
      color: #5f6368;
    }}
    .grid {{
      display: grid;
      gap: 14px;
    }}
    .metrics {{
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      margin: 18px 0 22px;
    }}
    .card, .plot-card, .table-card {{
      background: white;
      border: 1px solid #dfe3e8;
      border-radius: 8px;
      padding: 16px;
    }}
    .metric-card span {{
      display: block;
      color: #5f6368;
      font-size: 13px;
    }}
    .metric-card strong {{
      display: block;
      margin-top: 4px;
      font-size: 24px;
    }}
    .plot-grid {{
      grid-template-columns: repeat(auto-fit, minmax(430px, 1fr));
      align-items: start;
    }}
    .plot-card img {{
      display: block;
      width: 100%;
      height: auto;
      margin-top: 10px;
      border: 1px solid #eef0f2;
      border-radius: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 9px 8px;
      border-bottom: 1px solid #e8eaed;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: #3c4043;
      background: #f8fafd;
      font-weight: 650;
    }}
    footer {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 0 28px 32px;
      color: #5f6368;
      font-size: 13px;
    }}
    @media (max-width: 760px) {{
      header, main, footer {{
        padding-left: 16px;
        padding-right: 16px;
      }}
      .plot-grid {{
        grid-template-columns: 1fr;
      }}
      table {{
        display: block;
        overflow-x: auto;
        white-space: nowrap;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Comparativo de experimentos Fase 2B</h1>
    <p class="subtle">
      Dashboard local generado desde el resumen de corridas experimentales.
      Fecha de generación: {html.escape(generacion)}.
    </p>
  </header>
  <main>
    <section class="grid metrics">
      {cards}
    </section>

    <section class="card">
      <h2>Mejor combinación actual</h2>
      <p>
        La mejor combinación por AUC protocolo es
        <strong>{html.escape(str(mejor["classifier_name"]))}</strong> con
        <strong>{html.escape(str(mejor["feature_config_name"]))}</strong>.
        Se usa la corrida más reciente por combinación clasificador/features
        para evitar duplicados cuando la matriz se ejecuta más de una vez.
      </p>
    </section>

    <section class="grid plot-grid" style="margin-top: 14px;">
      {"".join(image_sections)}
    </section>

    <section class="table-card" style="margin-top: 14px;">
      <h2>Top corridas por AUC protocolo</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Clasificador</th>
            <th>Features</th>
            <th>AUC protocolo</th>
            <th>ROC AUC</th>
            <th>Recall</th>
            <th>Precision</th>
            <th>F1</th>
          </tr>
        </thead>
        <tbody>
          {_render_top_table(corridas, top_n)}
        </tbody>
      </table>
    </section>
  </main>
  <footer>
    Fuente: {html.escape(fuente)}
  </footer>
</body>
</html>
"""
    dashboard_out.write_text(html_text, encoding="utf-8")
    print(f"Dashboard generado: {dashboard_out}")
    return dashboard_out


def main() -> None:
    """Genera figuras y dashboard comparativo de experimentos."""

    args = parse_args()
    resumen = cargar_resumen(args.summary)
    corridas = preparar_corridas_comparables(resumen)
    figuras = generar_figuras(corridas, args.output_dir, args.top_n)
    if not args.no_dashboard:
        generar_dashboard(
            resumen,
            corridas,
            figuras,
            args.dashboard_out,
            args.summary,
            args.top_n,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Ejecución interrumpida por el usuario.")
