FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /app/server/requirements.txt

COPY indian_rummy /app/indian_rummy
COPY server /app/server
COPY models /app/models

# Hugging Face Spaces run containers as uid 1000
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
