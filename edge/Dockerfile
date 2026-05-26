FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY scripts /app/scripts
COPY config.yaml /app/edge/config.yaml
COPY models /app/edge/models

RUN mkdir -p /app/edge/data/crops /app/edge/demo_videos

EXPOSE 9101

CMD ["python", "-m", "app.main", "--config", "/app/edge/config.yaml", "--no-kafka"]
