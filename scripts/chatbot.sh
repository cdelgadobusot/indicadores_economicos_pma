#!/usr/bin/env bash
# ===========================================================================
# chatbot.sh — Prueba el chatbot RAG en la terminal (sin abrir el dashboard).
# Muestra qué motor usa (ollama / claude / extractivo) y responde ejemplos.
# Uso:  bash scripts/chatbot.sh
# ===========================================================================
set -e
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "❌ No existe el entorno (.venv). Ejecuta primero:  bash scripts/instalar.sh"
  exit 1
fi

./.venv/bin/python -m src.rag.chatbot
