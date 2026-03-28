# Tiny-Tale
# TinyLoRA на малых моделях: зависимость эффективности от числа параметров

Воспроизведение и расширение метода из статьи  
**"Learning to Reason in 13 Parameters"** (Morris et al., FAIR at Meta, 2026)  
[arxiv 2602.04118](https://arxiv.org/abs/2602.04118)

## Идея

Оригинальная статья показала: модель Qwen2.5-7B можно улучшить на задачах математики,
обучив всего **13 параметров** через RL (GRPO). Мы проверяем — работает ли это на
моделях меньшего размера, и как эффективность зависит от числа параметров модели.

Дополнительно сравним два поколения — Qwen3 и Qwen3.5 — при схожих размерах.

## Модели в эксперименте

| Модель | Параметров | Поколение | Размер bf16 | Где скачать |
|--------|-----------|-----------|-------------|-------------|
| Qwen3-0.6B  | 0.6B | Qwen3   | ~1.2 GB | [HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main) |
| Qwen3.5-0.8B | 0.8B | Qwen3.5 | ~1.8 GB | [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/main) |
| Qwen3-1.7B  | 1.7B | Qwen3   | ~3.5 GB | [HuggingFace](https://huggingface.co/Qwen/Qwen3-1.7B/tree/main) |
| Qwen3.5-2B | 2B | Qwen3.5 | ~4.5 GB | [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-2B/tree/main) |

Все четыре модели помещаются целиком в 6GB VRAM без квантизации.

## Структура репозитория

```
tiny-lora-scale-study/
├── tinylora.py        # ядро метода: TinyLoRA слой, SVD, weight tying
├── train.py           # обучение через GRPO на GSM8K
├── evaluate.py        # оценка accuracy на тестовой выборке
├── requirements.txt   # зависимости
├── README.md
├── .gitignore
├── configs/           # параметры для каждой модели
│   ├── qwen3-0.6b.json
│   ├── qwen3.5-0.8b.json
│   ├── qwen3-1.7b.json
│   └── qwen3.5-2b.json
├── models/            # сюда кладём скачанные модели (в .gitignore)
└── results/           # сюда сохраняются результаты (в .gitignore)
```

## Установка необходимых библиотек для воспроизведения эксперимента

```bash
pip install -r requirements.txt
```

## Запуск

### 1. Скачать модель вручную
Зайти на HuggingFace → Files and versions → скачать все файлы в соответствующую папку,
например `models/Qwen3-0.6B/`

### 2. Тестовый прогон (~20-40 минут)
Проверяет что всё работает корректно, не оценивает итоговое качество:
```bash
python train.py --config configs/qwen3-0.6b.json --test_run
```

### 3. Полное обучение
```bash
python train.py --config configs/qwen3-0.6b.json
```

### 4. Оценка
```bash
python evaluate.py --checkpoint ./results/Qwen3-0.6B/tinylora_v.pt
```

## Аппаратное обеспечение

Разработка и эксперименты ведутся на:
- GPU: NVIDIA RTX 4050 laptop edition 6GB VRAM
- RAM: 64GB
- CPU: Intel Core i5-12500H

## Ожидаемые результаты

Основная гипотеза: чем меньше модель, тем более будет заметным улучшение от дообучения TinyLoRa ввиду низкого baseline, и, как следствие, большего количества потенциальных задач для обучения.

Дополнительная гипотеза: Qwen3.5 при схожем размере покажет отличный результат от Qwen3 ввиду гибридной архитектуры, на которой метод не тестировался.

Результаты будут добавлены по мере завершения экспериментов в виде готовых pt-файлов.

