FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para algunas librerías Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo lo necesario para instalar dependencias primero (mejor cache)
COPY pyproject.toml README.md ./
COPY kikolens/ ./kikolens/
COPY .streamlit/ ./.streamlit/

# Instalar el proyecto y todas sus dependencias
RUN pip install --no-cache-dir -e .

# Puerto de Streamlit
EXPOSE 8501

# URL de Ollama (sobreescribible via docker-compose o docker run -e)
ENV OLLAMA_HOST=http://ollama:11434
ENV OLLAMA_MODEL=llama3.2

# Comprobación de salud del servidor de Streamlit
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

# Arrancar el dashboard directamente
CMD ["streamlit", "run", "kikolens/dashboard/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.fileWatcherType=none", \
     "--server.headless=true"]
