#!/usr/bin/env bash
# ===========================================================================
# notebook.sh — Abre el notebook principal en Jupyter.
# Uso:  bash scripts/notebook.sh
# ===========================================================================
set -e
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "❌ No existe el entorno (.venv). Ejecuta primero:  bash scripts/instalar.sh"
  exit 1
fi

echo "==> Abriendo el notebook en Jupyter ..."
./.venv/bin/jupyter notebook notebooks/proyecto_indicadores_economicos.ipynb
