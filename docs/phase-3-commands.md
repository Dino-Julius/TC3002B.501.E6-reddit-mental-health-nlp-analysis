# Wiki de comandos Phase 3

Esta guía concentra los comandos para reproducir la Fase 3 con Ollama local.
Los datos, cachés y resultados generados permanecen fuera de git.

## Modelos locales requeridos

Verifica que Ollama esté disponible y que los modelos estén instalados.

```zsh
ollama list
```

Modelos usados:

- `nomic-embed-text`: embeddings transformer locales.
- `qwen2.5:3b-instruct`: LLM local instruct.
- `llama3.2:3b`: LLM local instruct ligero.
- `gemma3:4b`: LLM local instruct alternativo.

Si hace falta instalarlos:

```zsh
ollama pull nomic-embed-text
ollama pull qwen2.5:3b-instruct
ollama pull llama3.2:3b
ollama pull gemma3:4b
```

Verifica que la API local responda:

```zsh
curl http://localhost:11434/api/tags
```

## Baseline Fase 2B sobre fold2

Ejecuta el baseline seleccionado en Fase 2B contra `data_test_fold.csv` / `data_test_fold2.csv`.

```zsh
uv run python scripts/predict_test_fold.py \
  --test-input data/raw/data_test_fold2.csv \
  --predictions-out data/processed/phase3/baseline_fold2_predictions.csv \
  --metrics-out data/processed/phase3/baseline_fold2_metrics.json \
  --metadata-out data/processed/phase3/baseline_fold2_metadata.json \
  --model-out data/processed/phase3/baseline_fold2_model.joblib \
  --dashboard-out reports/phase-2b-implementation/test_fold2_dashboard.html \
  --no-model \
  --evaluate-if-labeled
```

## Embeddings transformer con Ollama

El flujo base genera embeddings con `nomic-embed-text`, entrena una regresión
logística sobre `data_train.csv` y evalúa `data_test_fold2.csv`.

```zsh
uv run python scripts/run_phase3_embeddings.py
```

Salidas principales:

- `data/processed/phase3/embeddings_train.json`
- `data/processed/phase3/embeddings_test_fold2.json`
- `data/processed/phase3/embeddings_predictions.csv`
- `data/processed/phase3/embeddings_metrics.json`
- `data/processed/phase3/embeddings_metadata.json`

## Matriz de clasificadores con embeddings

Para cumplir la retroalimentación de Fase 3, se entrenan tres clasificadores
sobre los mismos embeddings locales:

- `logistic_regression`
- `linear_svm`
- `sgd_logistic`

```zsh
uv run python scripts/run_phase3_embedding_classifiers.py
```

Salidas principales:

- `data/processed/phase3/embedding_classifiers/logistic_regression_metrics.json`
- `data/processed/phase3/embedding_classifiers/linear_svm_metrics.json`
- `data/processed/phase3/embedding_classifiers/sgd_logistic_metrics.json`
- `data/processed/phase3/embedding_classifiers/summary_embedding_classifiers.csv`
- `data/processed/phase3/embedding_classifiers/summary_embedding_classifiers.json`

## LLM zero-shot base con Ollama

Clasifica `data_test_fold2.csv` con `qwen2.5:3b-instruct` y salida JSON
estructurada.

```zsh
uv run python scripts/run_phase3_llm.py
```

Salidas principales:

- `data/processed/phase3/llm_zero_shot_responses.jsonl`
- `data/processed/phase3/llm_zero_shot_predictions.csv`
- `data/processed/phase3/llm_zero_shot_metrics.json`
- `data/processed/phase3/llm_zero_shot_metadata.json`

Si alguna respuesta no cumple el esquema JSON, el script guarda el error en el
CSV/JSONL y omite métricas hasta revisar esas filas.

Si las respuestas crudas son semánticamente válidas pero falló la validación
local, recalcula predicciones y métricas sin volver a llamar a Ollama:

```zsh
uv run python scripts/evaluate_phase3_llm_raw.py
```

## Matriz de LLMs zero-shot y few-shot

Ejecuta tres LLMs locales en dos modos de prompt:

- `zero_shot`
- `few_shot`

El modo few-shot selecciona de forma reproducible 3 ejemplos positivos y 3
negativos desde `data_train.csv`.

```zsh
uv run python scripts/run_phase3_llm_matrix.py
```

Salidas principales:

- `data/processed/phase3/llm_matrix/few_shot_examples.json`
- `data/processed/phase3/llm_matrix/summary_llm_matrix.csv`
- `data/processed/phase3/llm_matrix/summary_llm_matrix.json`
- `data/processed/phase3/llm_matrix/*_predictions.csv`
- `data/processed/phase3/llm_matrix/*_responses.jsonl`
- `data/processed/phase3/llm_matrix/*_metrics.json`

En la corrida final, `gemma3:4b` en modo zero-shot devolvió 53 respuestas `{}` y
por tanto no generó métricas válidas. El reintento aislado con más intentos
mantuvo los mismos 53 errores:

```zsh
uv run python scripts/run_phase3_llm_matrix.py \
  --llm-model gemma3:4b \
  --prompt-mode zero_shot \
  --max-attempts 5 \
  --summary-csv-out /tmp/gemma3_zero_retry_summary.csv \
  --summary-json-out /tmp/gemma3_zero_retry_summary.json
```

Ese resultado se documenta como una limitación de robustez del contrato JSON en
zero-shot para ese modelo. La variante few-shot de `gemma3:4b` sí terminó sin
errores y con métricas válidas.

## Comparación final

Consolida las métricas de baseline, embeddings, matriz de clasificadores y
matriz de LLMs en una tabla lista para documentación.

```zsh
uv run python scripts/compare_phase3_results.py
```

Salidas:

- `data/processed/phase3/phase3_comparison.csv`
- `data/processed/phase3/phase3_comparison.json`

## Validación

Corre las pruebas unitarias.

```zsh
uv run pytest
```