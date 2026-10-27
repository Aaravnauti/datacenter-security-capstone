FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached until requirements change.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config ./config
COPY src ./src
COPY app.py ./
COPY .streamlit ./.streamlit

# Run as an unprivileged user. These folders are meant to be mounted as volumes.
RUN useradd --create-home appuser \
    && mkdir -p artifacts outputs data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Default: run the full pipeline (train -> evaluate -> explain).
# Dashboard: docker run -p 8501:8501 <image> streamlit run app.py --server.address 0.0.0.0
CMD ["python", "-m", "src.pipeline"]
