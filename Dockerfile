# Backend image for Render's free web service (D-084). NumPy-only inference, no torch (D-085).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# Render's free tier is 0.1 CPU; OpenBLAS would start one thread per host core and thrash the quota.
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install -r /app/server/requirements.txt

COPY indian_rummy /app/indian_rummy
COPY server /app/server
COPY models /app/models

RUN useradd -m -u 1000 app && chown -R app /app
USER app

# Render injects PORT (default 10000). One worker: game sessions live in memory.
CMD ["sh", "-c", "uvicorn server.server:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
