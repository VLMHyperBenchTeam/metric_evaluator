FROM python:3.10-slim

WORKDIR /workspace

# Установка системных зависимостей для компиляции пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry версии 1.8.3
RUN pip install --no-cache-dir poetry==1.8.3

# Отключаем создание виртуальных окружений
RUN poetry config virtualenvs.create false

# Копируем файлы зависимостей
COPY pyproject.toml poetry.lock ./

# Устанавливаем зависимости проекта
RUN poetry install --no-interaction --no-ansi --no-root

# Запускаем терминал
CMD ["/bin/bash"]