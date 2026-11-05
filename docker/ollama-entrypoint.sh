#!/bin/sh
set -e

MODEL=${OLLAMA_MODEL:-llama3.2}

# Arrancar el servidor Ollama en background
ollama serve &
OLLAMA_PID=$!

# Esperar hasta que el servidor esté listo (sin curl ni wget)
echo "[KikoLens] Waiting for Ollama server..."
until ollama list > /dev/null 2>&1; do
  sleep 2
done
echo "[KikoLens] Ollama ready."

# Descargar el modelo solo si no está ya en el volumen
if ollama list | grep -q "^${MODEL}"; then
  echo "[KikoLens] Model '${MODEL}' already present, skipping pull."
else
  echo "[KikoLens] Pulling model '${MODEL}'..."
  ollama pull "$MODEL"
  echo "[KikoLens] Model '${MODEL}' ready."
fi

# Mantener el servidor corriendo
wait $OLLAMA_PID
