# Phase 1 State of the Art Report

Este directorio contiene el entregable de **Fase 1: Estado del arte / related work** del reto `TC3002B.501`, preparado en LaTeX con la plantilla `elsarticle` de Elsevier.

## Compilación

Para compilar el PDF desde este directorio:

```zsh
make all
```

## Salida generada

El PDF compilado se genera en:

```text
build/main.pdf
```

## Limpieza

Para eliminar los artefactos de compilación:

```zsh
make clean
```

## Empaquetado para entrega

Para generar el archivo comprimido listo para envío:

```zsh
make zip
```

El comando crea:

```text
TC3002B-StateOfTheArt-LaTeX.zip
```

El archivo `.zip` incluye `main.tex`, `references.bib`, `sections/`, `tables/`, `figures/`, `Makefile`, `.latexmkrc` y `README.md`, sin incluir `build/`.
