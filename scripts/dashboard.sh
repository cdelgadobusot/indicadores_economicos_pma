#!/usr/bin/env bash
# ===========================================================================
# dashboard.sh — Abre el dashboard interactivo (Streamlit).
# Uso:  bash scripts/dashboard.sh        (Ctrl+C para detener)
# ===========================================================================
set -e
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "❌ No existe el entorno (.venv). Ejecuta primero:  bash scripts/instalar.sh"
  exit 1
fi

echo "==> Abriendo el dashboard en http://localhost:8501  (Ctrl+C para detener)"
./.venv/bin/streamlit run dashboard/app.py
