# Contributing

## Objetivo

Este repositorio utiliza un flujo de trabajo basado en ramas temporales y Pull Requests para mantener orden entre desarrollo de código, elaboración de reportes, documentación y versiones entregables del proyecto.

El flujo general del repositorio es:

```text
feat/*, fix/*, report/*, docs/*, chore/*
        ↓ Pull Request
develop
        ↓ Pull Request
master
````

## Branches principales

### `master`

`master` es la rama oficial y estable del proyecto.

Su objetivo es almacenar únicamente versiones revisadas, aprobadas y listas para entrega o referencia final. No se debe trabajar directamente sobre esta rama.

`master` solo debe recibir cambios mediante Pull Request desde `develop`.

### `develop`

`develop` es la rama de integración activa del proyecto.

Su objetivo es concentrar el avance validado de código, reportes, documentación, experimentos y ajustes relacionados con el reto.

Todo cambio debe integrarse primero a `develop` mediante Pull Request desde una rama temporal.

No se debe trabajar directamente sobre `develop`, salvo tareas excepcionales de mantenimiento del flujo de ramas.

## Ramas de trabajo

Las contribuciones deben realizarse en ramas temporales creadas a partir de `develop` actualizado.

Convenciones de nombres:

* `feat/*` para nuevas funcionalidades o componentes de código.
* `fix/*` para correcciones puntuales.
* `report/*` para reportes, entregables escritos, figuras, tablas o documentos.
* `docs/*` para documentación general del repositorio.
* `chore/*` para mantenimiento, configuración, dependencias o estructura del proyecto.
* `release/*` solo si se necesita preparar una entrega especial antes de integrarla a `master`.

Ejemplos:

```text
feat/phase-2b-baseline-pipeline
report/phase-2b-results
fix/user-split-leakage
docs/update-contributing-workflow
chore/update-python-dependencies
```

## Flujo de trabajo

### 1. Actualizar `develop`

Antes de crear una rama nueva, se debe partir desde `develop` actualizado:

```bash
git fetch origin
git switch develop
git pull --ff-only origin develop
```

### 2. Crear una rama temporal

Crear una rama de trabajo según el tipo de cambio:

```bash
git switch -c feat/nombre-de-la-tarea
```

Otros ejemplos:

```bash
git switch -c report/nombre-del-reporte
git switch -c fix/nombre-del-arreglo
git switch -c docs/nombre-del-cambio
git switch -c chore/nombre-del-cambio
```

### 3. Trabajar y confirmar cambios

Realizar cambios en la rama temporal y crear commits claros:

```bash
git status
git add <archivos>
git commit -m "Mensaje claro del cambio"
```

### 4. Mantener la rama actualizada

Antes de abrir o actualizar un Pull Request, sincronizar la rama con `develop`:

```bash
git fetch origin
git pull --rebase origin develop
```

Si la rama ya existe en remoto:

```bash
git push origin nombre-de-la-rama
```

Si es la primera vez que se sube:

```bash
git push -u origin nombre-de-la-rama
```

### 5. Abrir Pull Request hacia `develop`

Todo trabajo de código, reportes, documentación o configuración debe abrirse primero hacia:

```text
develop
```

No se deben abrir Pull Requests de ramas temporales directamente hacia `master`.

### 6. Eliminar la rama temporal

Después de que el Pull Request sea aprobado y fusionado a `develop`, la rama temporal debe eliminarse para mantener limpio el repositorio.

## Entregas y releases

Cuando `develop` contenga una versión estable y lista para entrega, se debe abrir un Pull Request:

```text
develop → master
```

Este Pull Request representa una entrega final, release o versión estable del proyecto.

Antes de fusionar hacia `master`, se debe verificar que:

* el contenido de la entrega esté completo;
* los reportes finales estén incluidos cuando aplique;
* el código relevante ejecute correctamente;
* no se hayan incluido datasets privados, archivos generados innecesarios o artefactos locales;
* el Pull Request haya sido revisado y aprobado por al menos un integrante del equipo.

## Pull Requests

Todo cambio debe integrarse mediante Pull Request en GitHub.

Cada Pull Request debe:

* tener un título claro;
* describir brevemente el cambio realizado;
* indicar la rama base correcta;
* incluir capturas, resultados o evidencia cuando aplique;
* evitar mezclar cambios no relacionados;
* ser revisado por mínimo un integrante del equipo antes de fusionarse.

La rama base debe ser:

```text
develop
```

excepto para releases finales, donde la rama base será:

```text
master
```

y la rama origen deberá ser:

```text
develop
```

## Commits

Los commits deben ser claros, específicos y describir una unidad lógica de trabajo.

Ejemplos recomendados:

```text
Implement Phase 2B baseline core pipeline
Add baseline training and evaluation scripts
Update Phase 2A conceptual design report
Fix user-based validation split
Document baseline execution workflow
```

Evitar mensajes genéricos como:

```text
changes
update
fix stuff
final
```

## Archivos que no deben versionarse

No se deben subir al repositorio:

* entornos virtuales como `.venv/`;
* cachés como `.pytest_cache/`, `.ruff_cache/` o `__pycache__/`;
* archivos del sistema como `.DS_Store`;
* datasets privados o locales en `data/raw/`;
* artefactos generados en `data/processed/`, salvo archivos `.gitkeep`;
* modelos entrenados como `.joblib`, salvo que el equipo acuerde explícitamente versionarlos;
* archivos temporales o de prueba local.

## Reglas de sincronización

Para evitar historiales duplicados o ramas divergentes:

1. Toda rama temporal debe crearse desde `develop` actualizado.
2. Todo cambio entra primero a `develop`.
3. `master` solo recibe Pull Requests desde `develop`.
4. No se deben mergear ramas temporales directamente a `master`.
5. No se debe mantener una rama permanente separada para reportes.
6. Si una rama se fusiona mediante squash, no debe fusionarse de nuevo por otro camino.
7. Después de fusionar una rama temporal, debe eliminarse.
8. Si `master` y `develop` divergen, se debe resolver antes de iniciar nuevas ramas de trabajo.