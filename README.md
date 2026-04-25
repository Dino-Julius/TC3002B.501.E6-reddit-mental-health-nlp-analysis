# TC3002B.501.E6 – Reddit Mental Health NLP Analysis

Repositorio del `Equipo 6` para el reto de la unidad de formación **TC3002B.501 – Desarrollo de aplicaciones avanzadas de ciencias computacionales**.

El proyecto se centra en el análisis computacional de publicaciones de Reddit relacionadas con salud mental, utilizando técnicas de procesamiento de lenguaje natural, aprendizaje automático y modelos modernos de inteligencia artificial para detectar indicadores asociados con suicidalidad.

---

## Información del equipo

| Campo | Información |
|---|---|
| Unidad de formación | TC3002B.501 – Desarrollo de aplicaciones avanzadas de ciencias computacionales |
| Campus | Tecnológico de Monterrey Campus Estado de México |
| Proyecto | Reddit Mental Health NLP Analysis |
| Equipo | 6 |
| Integrante | Ulises Jaramillo Portilla – A01798380 |
| Integrante | Fernanda Ponce Maciel – A01799293 |
| Integrante | Julio Cesar Vivas Medina – A01749879 |

---

## Profesores y módulos

| Módulo | Área | Profesor |
|---|---|---|
| Módulo 1 | Metodología de la investigación | Raúl Monroy Borja |
| Módulo 2 | Inteligencia Artificial | Jorge Adolfo Ramírez Uresti |
| Módulo 3 | Compiladores | Ariel Ortíz Ramírez |
| Módulo 4 | Métodos Cuantitativos y Simulación | Miguel González Mendoza |

---

## Resumen del reto

El reto consiste en diseñar y desarrollar una herramienta computacional para detectar indicadores de suicidalidad en publicaciones de Reddit. El problema se ubica dentro del análisis de salud mental en redes sociales, donde los usuarios suelen compartir experiencias, emociones y señales lingüísticas relacionadas con depresión, ideación suicida u otros estados de vulnerabilidad psicológica.

El objetivo general del proyecto es construir una solución efectiva y evaluable que permita clasificar publicaciones con base en su contenido textual. Para ello, se consideran técnicas de procesamiento de lenguaje natural, aprendizaje automático, aprendizaje profundo, modelos basados en transformers y modelos de lenguaje de gran escala.

Además de la detección binaria, el reto requiere analizar cómo las distintas aproximaciones se relacionan con dimensiones como:

- efectividad predictiva;
- interpretabilidad clínica;
- calidad del preprocesamiento textual;
- selección de atributos lingüísticos;
- modularidad de la solución;
- desempeño cuantitativo;
- reproducibilidad;
- y comparación entre modelos tradicionales y modelos modernos de IA.

---

## Fases del proyecto

El desarrollo del reto se organiza en tres fases principales.

### Fase 1 – Estado del arte y trabajo relacionado

La primera fase consiste en identificar y analizar literatura científica actual sobre detección de salud mental, depresión e ideación suicida en redes sociales.

Esta fase busca responder preguntas como:

- ¿Qué propuestas han abordado problemas similares?
- ¿Qué técnicas de PLN, ML, DL o LLMs han utilizado?
- ¿Qué fuentes de datos emplean?
- ¿Cómo representan el texto?
- ¿Qué métricas utilizan para evaluar sus modelos?
- ¿Qué limitaciones presentan?
- ¿Qué oportunidades existen para una solución propia?

El entregable principal de esta fase es un reporte en formato de artículo, desarrollado con LaTeX y plantilla Elsevier, ubicado en:

[`reports/state-of-the-art`](reports/state-of-the-art)

---

### Fase 2A – Diseño conceptual de la herramienta

La segunda fase, parte A, corresponde al diseño de la solución propuesta.

En esta etapa se define la arquitectura conceptual de la herramienta, incluyendo:

- flujo general de procesamiento;
- módulos principales;
- dependencias entre módulos;
- datos de entrada;
- datos de salida;
- técnicas de preprocesamiento;
- extracción de características;
- algoritmo de clasificación;
- estrategia de evaluación;
- y justificación técnica de la solución.

Este diseño debe preparar el camino para una implementación modular, verificable y evaluable.

---

### Fase 2B – Implementación, pruebas y evaluación

La segunda fase, parte B, consiste en implementar la herramienta y evaluarla formalmente.

Esta fase incluye:

- código fuente documentado;
- pruebas unitarias;
- preprocesamiento reproducible;
- entrenamiento de modelos;
- evaluación con el protocolo definido;
- cálculo de métricas como ROC AUC;
- análisis de errores;
- optimización del código;
- y justificación del algoritmo utilizado.

La implementación se desarrollará principalmente en:

[`src/reddit_mh_nlp`](src/reddit_mh_nlp)

Las pruebas estarán en:

[`tests`](tests)

---

### Fase 3 – Técnicas avanzadas de IA y comparación

La tercera fase busca comparar la solución desarrollada previamente contra una aproximación basada en técnicas modernas de inteligencia artificial.

Esta fase puede incluir:

- transformers;
- embeddings contextuales;
- modelos generativos;
- LLMs;
- estrategias zero-shot o few-shot;
- extracción de evidencia textual;
- generación de explicaciones;
- y comparación con el baseline tradicional.

El objetivo es determinar si una solución basada en modelos modernos mejora alguna dimensión relevante, como desempeño predictivo, interpretabilidad, robustez o facilidad de uso.

---

## Datos y evaluación

El conjunto de construcción analizado para el proyecto contiene:

| Característica | Valor |
|---|---:|
| Publicaciones totales | 1,516 |
| Etiquetas `yes` | 784 |
| Etiquetas `no` | 732 |
| Usuarios únicos | 1,408 |
| Usuarios con más de una publicación | 92 |

---