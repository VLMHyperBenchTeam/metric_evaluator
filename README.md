# metric_evaluator

**metric_evaluator** — репозиторий, предназначенный для вычисления и проверки метрик качества распознавания текста в задачах Visual Question Answering (VQA). 

## Описание метрик

- **WER (Word Error Rate)** — метрика, которая оценивает различия между предсказанным и эталонным текстом на уровне слов. Она измеряет частоту ошибок в словах, вычисляя отношения замен, вставок и удалений слов.
  
- **CER (Character Error Rate)** — метрика, похожая на WER, но работает на уровне символов. Она помогает оценить точность распознавания текста с точки зрения отдельных символов, что полезно при анализе моделей, ориентированных на распознавание текста.

## Запуск

### Запуск контейнера из опубликованного Docker-образа

Для запуска оценки метрик выполните:

```bash
docker run \
    -it \
    --rm \
    -v .:/workspace \
    ghcr.io/vlmhyperbenchteam/metric-evaluator:python3.10-slim_v0.1.0 \
    python3 example2.py
```

### Сборка Docker-образа локально

Для сборки Docker-образа выполните следующую команду:

```bash
docker build -t metric-evaluator:python3.10-slim_v0.1.0 -f docker/Dockerfile .
```

### Запуск собранного Docker-контейнера

Для запуска оценки метрик выполните:

```bash
docker run \
    -it \
    --rm \
    -v .:/workspace \
    metric-evaluator:python3.10-slim_v0.1.0 \
    python3 example2.py
```