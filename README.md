# MV Review Classification

Небольшой проект по классификации русскоязычных отзывов на 3 класса:
- `Доставка`
- `Магазин`
- `Товар`

## Что есть в репозитории

- `data.csv` — исходные отзывы с метками классов
- `solution.ipynb` — исследование данных, предобработка, модели BM25/LogisticRegression, CatBoost и эксперименты с fine-tune на BERT
- `model.py` — простой скрипт для инференса на сохраненной модели
- `final_model/` — готовая модель Hugging Face (`config.json`, `model.safetensors`)
- `pyproject.toml` / `requirements.txt` — зависимости проекта

## Как запустить

1. Установите окружение:

```bash
python -m pip install -r requirements.txt
```

2. Запустите ноутбук для изучения и тренировки:

```bash
jupyter notebook solution.ipynb
```

3. Запустите инференс на примерах:

```bash
python model.py
```

## Использование

Скрипт `model.py` загружает модель из `./final_model` и предсказывает класс для списка отзывов. Если нужно, можно заменить `MODEL_PATH` на другую папку с моделью.

## Замечания

- Основной подход в `solution.ipynb`:
  - TF-IDF / BM25 + логистическая регрессия
  - CatBoost на текстовых признаках
  - Fine-tune RuBERT с K-fold обучением
- Для быстрого применения подойдет `model.py`, а полный экспериментальный пайплайн описан в `solution.ipynb`.
