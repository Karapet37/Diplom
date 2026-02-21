# syntax=docker/dockerfile:1

#############################################
#              BUILD STAGE                  #
#############################################

# React build stage
FROM node:20-bookworm-slim AS web-builder
WORKDIR /webapp
COPY webapp/package*.json ./
RUN npm ci
COPY webapp/ ./
RUN npm run build

# Базовый образ Python
FROM python:3.12-slim AS builder

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Устанавливаем зависимости ОС
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Собираем зависимости в виртуальный кеш
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

#############################################
#               FINAL STAGE                 #
#############################################

# Финальный мини-образ
FROM python:3.12-slim

# Runtime environment defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root runtime user
RUN addgroup --system app && adduser --system --ingroup app --uid 10001 app

# Рабочая директория
WORKDIR /app

# Копируем зависимости из билдера
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем весь проект
COPY . .
COPY --from=web-builder /webapp/dist ./webapp/dist

# Writable paths for runtime state.
RUN mkdir -p /app/data /app/runtime \
 && chown -R app:app /app

# Run as unprivileged user.
USER app

# Graph workspace API port
EXPOSE 8008

# Default run: graph-first web API
CMD ["python", "start.py", "--web-api", "--host", "0.0.0.0", "--port", "8008"]
