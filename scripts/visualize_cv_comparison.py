"""
Este script genera visualizaciones de validación cruzada Phase 2B.
"""

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
    / "cross_validation"
    / "phase-2b-feedback"
    / "summary_cv.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase-2b-implementation" / "assets"
DEFAULT_DASHBOARD = (
    PROJECT_ROOT / "reports" / "phase-2b-implementation" / "cross_validation.html"
)
FIGURE_DPI = 160
REQUIRED_COLUMNS = {
    "classifier_name",
    "feature_config_name",
    "n_splits",
    "mean_protocol_auc",
    "std_protocol_auc",
    "mean_roc_auc",
    "mean_recall",
    "mean_precision",
    "mean_f1",
    "folds_completed",
}


def parse_args() -> argparse.Namespace:
    """
    Lee argumentos CLI para figuras y dashboard de CV.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dashboard-out", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Genera solo las figuras PNG y omite el dashboard HTML.",
    )
    return parser.parse_args()


def _cargar_pyplot() -> Any:
    """
    Carga matplotlib con backend no interactivo.
    """

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
    """
    Guarda una figura y cierra el objeto.
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
    Convierte una ruta local a ruta relativa para HTML estático.
    """

    return html.escape(os.path.relpath(path, start=dashboard_out.parent))


def cargar_resumen(summary_path: Path) -> pd.DataFrame:
    """
    Carga y valida el resumen agregado de validación cruzada.
    """

    if not summary_path.exists():
        raise SystemExit(f"No se encontró el resumen CV: {summary_path}")

    resumen = pd.read_csv(summary_path)
    faltantes = sorted(REQUIRED_COLUMNS.difference(resumen.columns))
    if faltantes:
        raise SystemExit(
            "El resumen CV no contiene las columnas requeridas: "
            + ", ".join(faltantes),
        )

    for columna in resumen.columns:
        if columna.startswith(("mean_", "std_")) or columna in {
            "n_splits",
            "folds_completed",
        }:
            resumen[columna] = pd.to_numeric(resumen[columna], errors="coerce")

    resumen = resumen.dropna(subset=["mean_protocol_auc"])
    if resumen.empty:
        raise SystemExit("No hay combinaciones CV con mean_protocol_auc para graficar.")

    return resumen.sort_values(
        ["mean_protocol_auc", "std_protocol_auc", "classifier_name", "feature_config_name"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _etiqueta(row: pd.Series) -> str:
    """
    Construye una etiqueta corta de combinación.
    """

    return f"{row['classifier_name']} | {row['feature_config_name']}"


def graficar_ranking_protocol_auc(
    resumen: pd.DataFrame,
    output_dir: Path,
    plt: Any,
) -> Path:
    """
    Grafica el ranking CV por mean_protocol_auc.
    """

    datos = resumen.head(20).iloc[::-1]
    etiquetas = [_etiqueta(row) for _, row in datos.iterrows()]

    fig, ax = plt.subplots(figsize=(9.5, max(4.8, len(datos) * 0.34)))
    barras = ax.barh(etiquetas, datos["mean_protocol_auc"])
    ax.set_xlabel("Mean AUC protocolo")
    ax.set_title("Ranking CV por AUC protocolo promedio")
    ax.set_xlim(0, max(1.0, float(datos["mean_protocol_auc"].max()) + 0.05))
    ax.grid(True, axis="x", alpha=0.3)
    ax.bar_label(
        barras,
        labels=[_formatear_metrica(valor) for valor in datos["mean_protocol_auc"]],
        padding=3,
    )

    return _guardar_figura(fig, output_dir / "cv_protocol_auc_ranking.png", plt)


def graficar_heatmap_protocol_auc(
    resumen: pd.DataFrame,
    output_dir: Path,
    plt: Any,
) -> Path:
    """
    Grafica mean_protocol_auc por clasificador y configuración de features.
    """

    tabla = resumen.pivot_table(
        index="classifier_name",
        columns="feature_config_name",
        values="mean_protocol_auc",
        aggfunc="max",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    imagen = ax.imshow(tabla.to_numpy(), aspect="auto")
    fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04, label="Mean AUC protocolo")

    ax.set_xticks(range(len(tabla.columns)), labels=tabla.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(tabla.index)), labels=tabla.index)
    ax.set_xlabel("Configuración de features")
    ax.set_ylabel("Clasificador")
    ax.set_title("CV mean_protocol_auc por combinación")

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

    return _guardar_figura(fig, output_dir / "cv_protocol_auc_heatmap.png", plt)


def graficar_metricas_error_bars(
    resumen: pd.DataFrame,
    output_dir: Path,
    plt: Any,
) -> Path:
    """
    Grafica métricas promedio con desviación estándar para el top 10.
    """

    datos = resumen.head(10).copy()
    etiquetas = [_etiqueta(row) for _, row in datos.iterrows()]
    x = list(range(len(datos)))

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.errorbar(
        x,
        datos["mean_protocol_auc"],
        yerr=datos["std_protocol_auc"].fillna(0),
        marker="o",
        linewidth=1.8,
        capsize=4,
        label="AUC protocolo",
    )
    ax.errorbar(
        x,
        datos["mean_roc_auc"],
        yerr=datos.get("std_roc_auc", pd.Series([0] * len(datos))).fillna(0),
        marker="s",
        linewidth=1.6,
        capsize=4,
        label="ROC AUC",
    )
    ax.errorbar(
        x,
        datos["mean_f1"],
        yerr=datos.get("std_f1", pd.Series([0] * len(datos))).fillna(0),
        marker="^",
        linewidth=1.6,
        capsize=4,
        label="F1",
    )
    ax.set_xticks(x, labels=etiquetas, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Media por fold")
    ax.set_title("Métricas CV con desviación estándar")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    return _guardar_figura(fig, output_dir / "cv_metric_error_bars.png", plt)


def generar_figuras(resumen: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """
    Genera todas las figuras comparativas de CV.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _cargar_pyplot()
    return {
        "cv_protocol_auc_ranking.png": graficar_ranking_protocol_auc(
            resumen,
            output_dir,
            plt,
        ),
        "cv_protocol_auc_heatmap.png": graficar_heatmap_protocol_auc(
            resumen,
            output_dir,
            plt,
        ),
        "cv_metric_error_bars.png": graficar_metricas_error_bars(
            resumen,
            output_dir,
            plt,
        ),
    }


def _render_metric_card(label: str, value: object) -> str:
    """
    Renderiza una tarjeta de métrica para el dashboard.
    """

    return f"""
      <article class="card metric-card">
        <span>{html.escape(label)}</span>
        <strong>{html.escape(str(value))}</strong>
      </article>
    """


def _render_top_table(resumen: pd.DataFrame) -> str:
    """
    Renderiza la tabla top 10 por mean_protocol_auc.
    """

    filas = []
    for indice, (_, row) in enumerate(resumen.head(10).iterrows(), start=1):
        filas.append(
            "<tr>"
            f"<td>{indice}</td>"
            f"<td>{html.escape(str(row['classifier_name']))}</td>"
            f"<td>{html.escape(str(row['feature_config_name']))}</td>"
            f"<td>{_formatear_metrica(row['mean_protocol_auc'])}</td>"
            f"<td>{_formatear_metrica(row['std_protocol_auc'])}</td>"
            f"<td>{_formatear_metrica(row['mean_roc_auc'])}</td>"
            f"<td>{_formatear_metrica(row['mean_recall'])}</td>"
            f"<td>{_formatear_metrica(row['mean_precision'])}</td>"
            f"<td>{_formatear_metrica(row['mean_f1'])}</td>"
            f"<td>{int(row['folds_completed'])}</td>"
            "</tr>"
        )
    return "\n".join(filas)


def generar_dashboard(
    resumen: pd.DataFrame,
    figuras: dict[str, Path],
    dashboard_out: Path,
    summary_path: Path,
) -> Path:
    """
    Genera el HTML estático del comparativo CV.
    """

    dashboard_out.parent.mkdir(parents=True, exist_ok=True)
    mejor = resumen.iloc[0]
    generacion = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    cards = "\n".join(
        [
            _render_metric_card("Combinaciones CV", len(resumen)),
            _render_metric_card("Folds configurados", int(mejor["n_splits"])),
            _render_metric_card(
                "Mejor mean AUC protocolo",
                _formatear_metrica(mejor["mean_protocol_auc"]),
            ),
            _render_metric_card("Std del mejor", _formatear_metrica(mejor["std_protocol_auc"])),
            _render_metric_card("ROC AUC del mejor", _formatear_metrica(mejor["mean_roc_auc"])),
        ]
    )

    explicaciones = {
        "cv_protocol_auc_ranking.png": (
            "Ordena las combinaciones por mean_protocol_auc, la métrica usada "
            "para seleccionar el modelo."
        ),
        "cv_protocol_auc_heatmap.png": (
            "Resume el desempeño promedio por clasificador y configuración TF-IDF."
        ),
        "cv_metric_error_bars.png": (
            "Muestra medias y desviaciones estándar para revisar estabilidad "
            "entre folds."
        ),
    }
    image_sections = []
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
  <title>Validación cruzada Phase 2B</title>
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
    <h1>Validación cruzada Phase 2B</h1>
    <p class="subtle">
      Dashboard local generado desde el resumen CV. Fecha de generación:
      {html.escape(generacion)}.
    </p>
  </header>
  <main>
    <section class="grid metrics">
      {cards}
    </section>

    <section class="card">
      <h2>Protocolo de selección</h2>
      <p>
        La validación cruzada usa únicamente <strong>data_train.csv</strong>.
        <strong>data_test_fold1.csv</strong> no se usa para selección de modelo.
      </p>
      <p>
        StratifiedGroupKFold preserva el balance de clases mientras separa por
        <strong>user_id</strong>, evitando fuga de publicaciones del mismo usuario
        entre entrenamiento y validación.
      </p>
      <p>
        La mejor combinación se selecciona por <strong>mean_protocol_auc</strong>.
        Mejor actual: <strong>{html.escape(str(mejor["classifier_name"]))}</strong>
        con <strong>{html.escape(str(mejor["feature_config_name"]))}</strong>.
      </p>
    </section>

    <section class="grid plot-grid" style="margin-top: 14px;">
      {"".join(image_sections)}
    </section>

    <section class="table-card" style="margin-top: 14px;">
      <h2>Top 10 combinaciones por mean_protocol_auc</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Clasificador</th>
            <th>Features</th>
            <th>Mean AUC protocolo</th>
            <th>Std AUC protocolo</th>
            <th>Mean ROC AUC</th>
            <th>Mean Recall</th>
            <th>Mean Precision</th>
            <th>Mean F1</th>
            <th>Folds</th>
          </tr>
        </thead>
        <tbody>
          {_render_top_table(resumen)}
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
    """
    Genera figuras y dashboard de resultados CV.
    """

    args = parse_args()
    resumen = cargar_resumen(args.summary)
    figuras = generar_figuras(resumen, args.output_dir)
    if not args.no_dashboard:
        generar_dashboard(resumen, figuras, args.dashboard_out, args.summary)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Ejecución interrumpida por el usuario.")
