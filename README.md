<div align="center">
  <h1>
      AI animals registration
  </h1>
</div>

## Описание системы

1. Распознавание
2. Классификация
3. Регистрация

## Распознавание

Используется обычная модель yolov8n, которая детектирует животных на картинке,  на основе этих данных мы берём количество животных в регистрации

## Классификация

Вырезаем Bounding Box, который вадала нам yolo, нормализуем, изменяем размер, отдаём в модель, которая выдаёт нам класс

## Регистрация

1. Проходим по отсортированным картинкам

    Отслеживаем животных на них и сохраняем время первого появления

    1.1. Получаем разницу между текущим временем и последним появлением

    1.2. Если время > 30 минут, то записываем регистрацию и перестаем отслеживать объект

    1.3. Обновляем текущее отслеживание

    Время, максимальное количество объектов и т.д.

2. Дополняем регистрации неучтенными отслеживаниями

## Описания файлов

- [Веса моделей](https://drive.google.com/drive/folders/14YmiZmSwurXOYQR2_nHMRViu8XVVdHZ_?usp=sharing)
- [Веб интерфейс](https://github.com/llitone/ai-animals-tracking/blob/main/site)
- [Веб интерфейс на Flask](https://github.com/llitone/ai-animals-tracking/blob/main/site_flask)
- [Реализация регистрации животного](https://github.com/llitone/ai-animals-tracking/blob/main/site_flask/frames_tracker.py)
- [Ноутбук с генерацией датасета](https://github.com/llitone/ai-animals-tracking/blob/main/dataset_generation.ipynb)
- [Ноутбук с обучением классификатора (VGG19)](https://github.com/llitone/ai-animals-tracking/blob/main/classifier_fitting.ipynb)
