# Wiki de comandos Phase 2B

Esta guía concentra los comandos principales para ejecutar, evaluar y validar la implementación de Phase 2B con `uv run`.

## Estado actual

La implementación actual incluye:

- Pipeline baseline TF-IDF + Regresión Logística.
- Métricas alineadas al protocolo del curso: `protocol_auc`, `true_positive_rate` y `false_positive_rate`.
- Visualizaciones PNG y dashboard HTML estático.
- Tracking local ligero de corridas experimentales.
- Matriz experimental de 5 clasificadores x 4 configuraciones de características.
- Validación cruzada con `StratifiedGroupKFold` para selección de modelo sin fuga por `user_id`.
- Pruebas unitarias para métricas, tracking, catálogo experimental y modelos.

## Reglas de artefactos

Los datos y resultados generados deben permanecer fuera de git.

Artefactos ignorados esperados:

- `data/raw/data_train.csv`
- `data/processed/baseline_*.json`
- `data/processed/baseline_*.csv`
- `data/processed/*.joblib`
- `data/processed/experiments/`
- `data/processed/cross_validation/`
- `reports/phase-2b-implementation/assets/*.png`
- `reports/phase-2b-implementation/assets/cv_*.png`
- `reports/phase-2b-implementation/dashboard.html`
- `reports/phase-2b-implementation/comparison.html`
- `reports/phase-2b-implementation/cross_validation.html`

## Entrenamiento baseline

Entrena el baseline principal y genera modelo, predicciones, métricas e interpretabilidad.

```zsh
uv run python scripts/train_baseline.py
```

Salidas esperadas:

- `data/processed/baseline_model.joblib`
- `data/processed/baseline_validation_predictions.csv`
- `data/processed/baseline_metrics.json`
- `data/processed/baseline_interpretability.json`

## Evaluación de un modelo entrenado

Evalúa un modelo persistido sobre un CSV. Si el CSV incluye etiquetas, también guarda métricas.

```zsh
uv run python scripts/evaluate_baseline.py --input data/raw/data_train.csv
```

Salidas esperadas:

- `data/processed/baseline_predictions.csv`
- `data/processed/baseline_eval_metrics.json`
- `data/processed/baseline_eval_interpretability.json`

## Visualizaciones y dashboard

Genera PNGs y dashboard HTML a partir de los artefactos baseline.

```zsh
uv run python scripts/visualize_baseline_results.py
```

Genera solo las visualizaciones PNG, sin dashboard.

```zsh
uv run python scripts/visualize_baseline_results.py --no-dashboard
```

Salidas esperadas:

- `reports/phase-2b-implementation/assets/confusion_matrix.png`
- `reports/phase-2b-implementation/assets/roc_curve.png`
- `reports/phase-2b-implementation/assets/score_distribution.png`
- `reports/phase-2b-implementation/assets/prediction_distribution.png`
- `reports/phase-2b-implementation/dashboard.html`

## Dashboard comparativo de experimentos

Genera visualizaciones comparativas desde el resumen agregado de corridas.

```zsh
uv run python scripts/visualize_experiment_comparison.py
```

Genera solo las figuras comparativas, sin dashboard HTML.

```zsh
uv run python scripts/visualize_experiment_comparison.py --no-dashboard
```

Salidas esperadas:

- `reports/phase-2b-implementation/assets/experiment_protocol_auc_ranking.png`
- `reports/phase-2b-implementation/assets/experiment_protocol_auc_heatmap.png`
- `reports/phase-2b-implementation/assets/experiment_top_metrics.png`
- `reports/phase-2b-implementation/assets/experiment_roc_vs_protocol_auc.png`
- `reports/phase-2b-implementation/comparison.html`

## Validación cruzada para selección de modelo

La validación cruzada es ahora el método preferido para selección de modelo.
Usa únicamente `data_train.csv`: el fold oficial `data_test_fold1.csv` no se
usa para elegir clasificador, configuración TF-IDF ni hiperparámetros. Las
corridas single-split se conservan como baseline exploratorio e histórico.

Corre la matriz completa de 20 combinaciones con 5 folds:

```zsh
UV_NO_SYNC=1 uv run python scripts/run_cross_validation_experiments.py --classifier-name all --feature-config-name all --n-splits 5
```

Visualiza los resultados CV:

```zsh
UV_NO_SYNC=1 uv run python scripts/visualize_cv_comparison.py
```

Corre una sola combinación CV:

```zsh
UV_NO_SYNC=1 uv run python scripts/run_cross_validation_experiments.py --classifier-name logistic_regression --feature-config-name char_wb_3_5 --n-splits 5
```

Salidas esperadas:

- `data/processed/cross_validation/phase-2b-feedback/fold_results.csv`
- `data/processed/cross_validation/phase-2b-feedback/summary_cv.csv`
- `data/processed/cross_validation/phase-2b-feedback/summary_cv.json`
- `data/processed/cross_validation/phase-2b-feedback/best_model_cv.json`
- `reports/phase-2b-implementation/assets/cv_protocol_auc_ranking.png`
- `reports/phase-2b-implementation/assets/cv_protocol_auc_heatmap.png`
- `reports/phase-2b-implementation/assets/cv_metric_error_bars.png`
- `reports/phase-2b-implementation/cross_validation.html`

Después de elegir el modelo con CV, el fold oficial de prueba se evalúa solo
como medición final del modelo ya seleccionado.

## Predicción del fold oficial de prueba

El archivo `data_test_fold1.csv` debe colocarse en `data/raw/`. Aunque el fold
incluya la columna `is_suicide`, esas etiquetas no se usan para entrenamiento,
selección de modelo ni ajuste de hiperparámetros.

Genera predicciones sin evaluar etiquetas:

```zsh
uv run python scripts/predict_test_fold.py --no-model
```

Genera predicciones y evalúa explícitamente porque este fold incluye etiquetas:

```zsh
uv run python scripts/predict_test_fold.py --no-model --evaluate-if-labeled
```

Salidas esperadas:

- `data/processed/test_folds/data_test_fold1_predictions.csv`
- `data/processed/test_folds/data_test_fold1_metrics.json`
- `data/processed/test_folds/data_test_fold1_metadata.json`
- `reports/phase-2b-implementation/test_fold1_dashboard.html`
- `reports/phase-2b-implementation/assets/test_fold1_confusion_matrix.png`
- `reports/phase-2b-implementation/assets/test_fold1_score_distribution.png`
- `reports/phase-2b-implementation/assets/test_fold1_prediction_distribution.png`
- `reports/phase-2b-implementation/assets/test_fold1_roc_curve.png`

Las predicciones, métricas, modelos y dashboards generados quedan ignorados por
git. La evaluación del fold solo debe consultarse después de fijar el workflow
de predicción.

## Tracking de una corrida

Ejecuta una corrida experimental versionada por `run_id`.

```zsh
uv run python scripts/run_baseline_experiments.py --no-model
```

El flag `--no-model` evita guardar `model.joblib` y deja la corrida más ligera.

Salidas esperadas:

- `data/processed/experiments/phase-2b-implementation/<run_id>/metrics.json`
- `data/processed/experiments/phase-2b-implementation/<run_id>/predictions.csv`
- `data/processed/experiments/phase-2b-implementation/<run_id>/interpretability.json`
- `data/processed/experiments/phase-2b-implementation/<run_id>/run_metadata.json`
- `data/processed/experiments/phase-2b-implementation/summary.csv`
- `data/processed/experiments/phase-2b-implementation/summary.json`

## Catálogo experimental

Lista los clasificadores y configuraciones disponibles.

```zsh
uv run python scripts/run_baseline_experiments.py --list-experiments
```

Clasificadores disponibles:

- `logistic_regression`
- `linear_svm`
- `sgd_logistic`
- `multinomial_nb`
- `complement_nb`

Configuraciones de características disponibles:

- `word_unigram`
- `word_unigram_bigram`
- `word_unigram_trigram`
- `char_wb_3_5`

## Correr una combinación específica

Ejecuta un clasificador con una configuración de features.

```zsh
uv run python scripts/run_baseline_experiments.py \
  --classifier-name linear_svm \
  --feature-config-name char_wb_3_5 \
  --no-model
```

## Correr la matriz completa

Ejecuta las 20 combinaciones: 5 clasificadores x 4 configuraciones.

```zsh
uv run python scripts/run_baseline_experiments.py \
  --classifier-name all \
  --feature-config-name all \
  --no-model
```

Si se quiere que el runner continúe aunque una corrida falle:

```zsh
uv run python scripts/run_baseline_experiments.py \
  --classifier-name all \
  --feature-config-name all \
  --no-model \
  --continue-on-error
```

## Validación local

Comandos recomendados antes de cerrar un commit:

```zsh
uv run ruff check src scripts tests
uv run python -m compileall -q src scripts tests
uv run pytest
```

Comandos útiles para revisar estado de artefactos ignorados:

```zsh
git status --short --ignored data/processed reports/phase-2b-implementation data/raw
```

## Flujo recomendado de trabajo

1. Validar que el ambiente responde:

```zsh
uv run python scripts/run_baseline_experiments.py --help
```

2. Entrenar el baseline principal:

```zsh
uv run python scripts/train_baseline.py
```

3. Generar visualizaciones y dashboard:

```zsh
uv run python scripts/visualize_baseline_results.py
```

4. Correr la matriz experimental ligera:

```zsh
uv run python scripts/run_baseline_experiments.py \
  --classifier-name all \
  --feature-config-name all \
  --no-model
```

5. Correr validación cruzada para seleccionar modelo:

```zsh
UV_NO_SYNC=1 uv run python scripts/run_cross_validation_experiments.py \
  --classifier-name all \
  --feature-config-name all \
  --n-splits 5
```

6. Generar el dashboard de validación cruzada:

```zsh
UV_NO_SYNC=1 uv run python scripts/visualize_cv_comparison.py
```

7. Validar calidad antes de commit:

```zsh
uv run ruff check src scripts tests
uv run python -m compileall -q src scripts tests
uv run pytest
```

8. Generar el comparativo final de experimentos single-split:

```zsh
uv run python scripts/visualize_experiment_comparison.py
```
