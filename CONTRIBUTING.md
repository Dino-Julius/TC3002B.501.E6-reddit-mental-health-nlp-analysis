# Contributing

## Objetivo

Este repositorio utiliza un flujo de trabajo basado en GitHub Pull Requests para mantener orden entre desarrollo de código, elaboración de reportes y versiones entregables del proyecto.

## Branches principales

### `master`

Es la rama oficial del proyecto.

Su objetivo es almacenar únicamente versiones estables, revisadas y listas para entrega o referencia final.

### `develop`

Es la rama de integración de código.

Su objetivo es concentrar el avance técnico del proyecto, incluyendo implementación, pruebas, ajustes y mejoras relacionadas con la solución computacional.

### `reports`

Es la rama de integración de reportes.

Su objetivo es concentrar el trabajo documental del proyecto, incluyendo reportes en LaTeX, revisiones de redacción, bibliografía, tablas, figuras y entregables escritos.

## Ramas de trabajo

Las contribuciones no deben hacerse directamente sobre las ramas principales.

Se deben crear ramas temporales a partir de la rama base correspondiente:

- `feat/*` para trabajo de código que parte de `develop`
- `fix/*` para correcciones puntuales
- `report/*` para trabajo documental que parte de `reports`
- `release/*` para preparar una entrega que combine código y reportes

## Flujo de trabajo

### Trabajo de código

1. Crear una rama de trabajo a partir de `develop`.
2. Realizar cambios y confirmarlos con commits claros.
3. Abrir un Pull Request hacia `develop`.
4. Integrar el cambio en `develop` una vez revisado y aprobado.

### Trabajo de reportes

1. Crear una rama de trabajo a partir de `reports`.
2. Realizar cambios y confirmarlos con commits claros.
3. Abrir un Pull Request hacia `reports`.
4. Integrar el cambio en `reports` una vez revisado y aprobado.

### Entregas

Cuando la entrega corresponda únicamente a código:

1. Abrir un Pull Request de `develop` hacia `master`.
2. Revisar y aprobar antes de integrar.

Cuando la entrega corresponda únicamente a reportes:

1. Abrir un Pull Request de `reports` hacia `master`.
2. Revisar y aprobar antes de integrar.

Cuando una entrega combine código y reportes:

1. Crear una rama `release/*` a partir de `master`.
2. Integrar en esa rama los cambios necesarios desde `develop` y `reports`.
3. Validar el contenido final de la entrega.
4. Abrir un Pull Request de `release/*` hacia `master`.

## Pull Requests

Todo cambio debe integrarse mediante Pull Request en GitHub.

Cada Pull Request debe:

- tener un título claro;
- describir brevemente el cambio realizado;
- indicar la rama base correcta;
- y ser revisado por mínimo un integrante del equipo antes de fusionarse.

## Resumen del modelo

- `master`: versión oficial y entregable
- `develop`: integración de código
- `reports`: integración de reportes
- ramas temporales: trabajo específico por tarea
- integración final: siempre por Pull Request en GitHub
