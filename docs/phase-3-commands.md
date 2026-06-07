# Wiki de comandos Phase 3

Esta guía concentra los comandos para ejecutar la Fase 3 con Ollama local.
Los datos, cachés y resultados generados permanecen fuera de git.

## Modelos locales requeridos

Verifica que Ollama esté disponible y que los modelos estén instalados.

```zsh
ollama list
```

Modelos usados:

- `nomic-embed-text`: embeddings transformer locales.
- `qwen2.5:3b-instruct`: LLM local para clasificación zero-shot.

Si hace falta instalarlos:

```zsh
ollama pull nomic-embed-text
ollama pull qwen2.5:3b-instruct
```

Verifica que la API local responda:

```zsh
curl http://localhost:11434/api/tags
```

## Baseline Fase 2B sobre fold2

Ejecuta el baseline seleccionado en Fase 2B contra `data_test_fold2.csv`.

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

Genera embeddings con `nomic-embed-text`, entrena regresión logística sobre
`data_train.csv` y evalúa `data_test_fold2.csv`.

```zsh
uv run python scripts/run_phase3_embeddings.py
```

Salidas principales:

- `data/processed/phase3/embeddings_train.json`
- `data/processed/phase3/embeddings_test_fold2.json`
- `data/processed/phase3/embeddings_predictions.csv`
- `data/processed/phase3/embeddings_metrics.json`
- `data/processed/phase3/embeddings_metadata.json`

## LLM zero-shot con Ollama

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

## Comparación final

Consolida las métricas de baseline, embeddings y LLM en una tabla lista para
documentación.

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

## Limpieza local al terminar

Después de exportar PDFs y resultados finales, se puede liberar espacio local.

```zsh
ollama rm qwen2.5:3b-instruct
ollama rm nomic-embed-text
rm -rf data/processed/phase3
```
