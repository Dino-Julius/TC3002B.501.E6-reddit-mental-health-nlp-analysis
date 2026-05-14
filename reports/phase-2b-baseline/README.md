# Reporte del baseline Fase 2B

Este directorio documenta las visualizaciones locales del baseline de Fase 2B.
El objetivo es resumir resultados experimentales del modelo TF-IDF + Regresión
Logística sin modificar el entrenamiento, la evaluación ni las métricas ya
calculadas.

## Objetivo

El baseline de Fase 2B establece una referencia reproducible para clasificar
publicaciones de Reddit con etiquetas binarias `no` y `yes`. Las visualizaciones
permiten revisar desempeño global, tipos de error y distribución de puntajes
antes de comparar con técnicas más avanzadas.

## Entrenamiento

Desde la raíz del repositorio, el entrenamiento se ejecuta con:

```zsh
uv run python scripts/train_baseline.py
```

El script genera los artefactos procesados en `data/processed/`, incluyendo
métricas, predicciones de validación, reporte interpretativo y modelo
serializado. Estos archivos son salidas locales y no deben versionarse.

## Visualizaciones

Para generar las figuras PNG y el dashboard HTML:

```zsh
uv run python scripts/visualize_baseline_results.py
```

Para generar solo las figuras PNG, sin dashboard:

```zsh
uv run python scripts/visualize_baseline_results.py --no-dashboard
```

El script usa por defecto:

- `data/processed/baseline_metrics.json`
- `data/processed/baseline_validation_predictions.csv`
- `data/processed/baseline_interpretability.json`
- `reports/phase-2b-baseline/assets/`
- `reports/phase-2b-baseline/dashboard.html`

## Archivos esperados

Las salidas generadas son:

- `reports/phase-2b-baseline/assets/confusion_matrix.png`
- `reports/phase-2b-baseline/assets/roc_curve.png`
- `reports/phase-2b-baseline/assets/score_distribution.png`
- `reports/phase-2b-baseline/assets/prediction_distribution.png`
- `reports/phase-2b-baseline/dashboard.html`

Estos archivos se generan localmente y permanecen ignorados por Git. El único
archivo rastreable dentro de `assets/` es `.gitkeep`.

## Explicación de gráficas

- Matriz de confusión: muestra verdaderos negativos, falsos positivos, falsos
  negativos y verdaderos positivos para revisar tipos de acierto y error.
- Curva ROC: resume la relación entre verdaderos positivos y falsos positivos
  al variar el umbral de clasificación.
- Distribución de puntajes: muestra cómo se distribuyen los puntajes estimados
  para la clase `yes`, agrupados por etiqueta real cuando está disponible.
- Distribución de predicciones: cuenta cuántas publicaciones fueron asignadas a
  cada clase predicha (`no` o `yes`).

## Interpretación actual del baseline

En la primera corrida de validación, el baseline obtuvo ROC AUC 0.7683, recall
0.6747, precision 0.7467 y F1 0.7089. La matriz de confusión fue TN 97, FP 38,
FN 54 y TP 112.

Estos valores deben interpretarse como resultados experimentales de referencia
para el reto académico. No representan desempeño clínico final ni validación
para uso en decisiones reales de salud mental.

## Nota ética

Esta herramienta es experimental y no constituye un sistema de diagnóstico
clínico. Sus resultados deben revisarse únicamente como apoyo académico para
análisis de modelos de PLN y no como evaluación individual de riesgo.
