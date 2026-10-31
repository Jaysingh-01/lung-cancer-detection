# Serves the Streamlit app. Works on Render, Hugging Face Spaces (Docker),
# Railway, Fly.io, Google Cloud Run and Azure Web App for Containers.
# Python matches the version the model bundle was trained under.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501

WORKDIR /app

COPY requirements.txt .
# Wheels only: never compile numpy/scikit-learn from source during a build.
RUN pip install --only-binary=:all: -r requirements.txt

COPY .streamlit ./.streamlit
COPY src ./src
COPY models ./models
COPY app.py .

# Non-root user; uid 1000 is also what Hugging Face Spaces expects.
RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)"

# Most hosts inject $PORT; default 8501 otherwise.
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
