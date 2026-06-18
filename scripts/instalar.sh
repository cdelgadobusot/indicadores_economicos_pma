#!/usr/bin/env bash
# ===========================================================================
# instalar.sh — Crea el entorno virtual e instala todas las dependencias.
# Uso:  bash scripts/instalar.sh
# ===========================================================================
set -e
cd "$(dirname "$0")/.."   # Ir a la raíz del proyecto

echo "==> 1/3 Creando el entorno virtual (.venv) ..."
python3 -m venv .venv

echo "==> 2/3 Actualizando pip ..."
./.venv/bin/python -m pip install --upgrade pip --quiet

echo "==> 3/3 Instalando dependencias (puede tardar unos minutos) ..."
./.venv/bin/python -m pip install -r requirements.txt

echo ""
echo "✅ Listo. Ahora puedes ejecutar:"
echo "   bash scripts/dashboard.sh     # abre el dashboard"
echo "   bash scripts/notebook.sh      # abre el notebook"
echo "   bash scripts/chatbot.sh       # prueba el chatbot en la terminal"
