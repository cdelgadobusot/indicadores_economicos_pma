#!/usr/bin/env bash
# ===========================================================================
# preparar_ollama.sh — Descarga el modelo de IA local (gratis) para el chatbot.
# Requiere tener Ollama instalado: https://ollama.com/download
# Uso:  bash scripts/preparar_ollama.sh
# ===========================================================================
set -e

if ! command -v ollama >/dev/null 2>&1; then
  echo "❌ Ollama no está instalado."
  echo "   Descárgalo de:  https://ollama.com/download"
  exit 1
fi

MODELO="${OLLAMA_MODELO:-llama3.2}"
echo "==> Descargando el modelo '$MODELO' (puede tardar unos minutos la 1.ª vez)..."
ollama pull "$MODELO"

echo ""
echo "✅ Modelo listo. El chatbot usará Ollama automáticamente (gratis, local)."
echo "   Verifícalo con:  bash scripts/chatbot.sh   (debe decir 'activo: ollama')"
