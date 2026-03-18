"""
Скрипт обучения TinyLoRA на датасете GSM8K через GRPO (RL).
Воспроизводит эксперимент из статьи "Learning to Reason in 13 Parameters".

Запуск:
    python train.py

Или с кастомными параметрами:
    python train.py --proj_dim 13 --rank 1 --lr 1e-5 --epochs 3
"""

import argparse
import re
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from tinylora import apply_tinylora_to_model, count_trainable_params


# ─────────────────────────────────────────────
# 1. Аргументы командной строки
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="TinyLoRA GRPO Training")
    parser.add_argument("--config", type=str, default=None,
                        help="Путь к JSON конфигу (например configs/qwen3-0.6b.json). "
                             "Флаги командной строки перекрывают значения из конфига.")
    parser.add_argument("--model_name", type=str, default="./models/Qwen3-0.6B",
                        help="Путь к модели (локальная папка или HuggingFace id)")
    parser.add_argument("--proj_dim", type=int, default=13,
                        help="Размер обучаемого вектора v (= число параметров в модели)")
    parser.add_argument("--rank", type=int, default=1,
                        help="Ранг SVD разложения")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (статья тестирует 1e-7 до 2e-4)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Число эпох обучения")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (уменьшить при OOM)")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Число сэмплов на задачу (статья использует 4)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Макс. длина ответа модели")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Папка для сохранения результатов")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="4-bit квантизация для экономии VRAM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Ограничить обучающую выборку (None = весь датасет). "
                             "Для быстрого теста используй --max_train_samples 50")
    parser.add_argument("--max_eval_samples", type=int, default=200,
                        help="Ограничить eval выборку")
    parser.add_argument("--test_run", action="store_true", default=False,
                        help="Быстрый тест: 50 train, 20 eval, 1 эпоха, 128 токенов. "
                             "Займёт ~15-30 минут вместо нескольких дней.")
    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. Функция награды (Reward Function)
# ─────────────────────────────────────────────
def extract_answer(text: str) -> str | None:
    """
    Извлекает финальный ответ из текста модели.
    Ищет паттерны типа: "#### 42", "The answer is 42", "= 42"
    """
    # GSM8K стандартный формат ответа: #### <число>
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: последнее число в тексте
    numbers = re.findall(r"[\-\d]+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        return numbers[-1]

    return None


def reward_correct_answer(completions: list[str], **kwargs) -> list[float]:
    """
    Функция награды для GRPO:
      +1.0 если ответ совпадает с правильным
       0.0 иначе

    В trl 0.29 все поля датасета приходят через **kwargs.
    Поле 'answer' мы кладём в датасет в prepare_dataset().

    completions : список ответов модели (строки)
    kwargs      : остальные поля батча, в т.ч. 'answer'
    """
    answers = kwargs.get("answer", [])
    rewards = []
    for completion, gt_answer in zip(completions, answers):
        predicted = extract_answer(completion)
        gt_clean = str(gt_answer).replace(",", "").strip()

        if predicted is not None and predicted == gt_clean:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


# ─────────────────────────────────────────────
# 3. Подготовка датасета GSM8K
# ─────────────────────────────────────────────
def prepare_dataset(tokenizer, split: str = "train", max_samples: int = None):
    """
    Загружает GSM8K и форматирует в chat-формат для Qwen3.

    GSM8K содержит ~7500 задач в train, ~1319 в test.
    Формат: {"question": "...", "answer": "... #### 42"}
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def format_example(example):
        # Извлекаем числовой ответ из строки вида "... #### 42"
        answer_text = example["answer"]
        gt_answer = extract_answer(answer_text)

        # Формируем промпт в chat-формате
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful math tutor. "
                    "Solve the problem step by step. "
                    "At the end, write your final answer after '####'."
                ),
            },
            {
                "role": "user",
                "content": example["question"],
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {
            "prompt": prompt,
            "answer": gt_answer,  # только число для reward function
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    return dataset


# ─────────────────────────────────────────────
# 4. Загрузка модели
# ─────────────────────────────────────────────
def load_model_and_tokenizer(model_name: str, load_in_4bit: bool = False):
    """
    Загружает модель на GPU явно через device_map="cuda:0".
    Для маленьких моделей (0.6B, 1.7B, 2.5B) это гарантирует
    что все вычисления идут на GPU, а не на CPU.
    """
    print(f"📥 Загружаем модель: {model_name}")
    print(f"   4-bit квантизация: {load_in_4bit}")

    # Определяем устройство
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"   Устройство: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,          # явно на GPU, не "auto"
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    # Проверяем что модель реально на GPU
    vram_used = round(torch.cuda.memory_allocated(0) / 1e9, 2)
    print(f"✅ Модель загружена | VRAM занято: {vram_used} GB")
    if vram_used == 0.0:
        print("⚠️  ВНИМАНИЕ: VRAM = 0, модель может быть на CPU!")

    return model, tokenizer


# ─────────────────────────────────────────────
# 5. Главная функция
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # --- Загрузка JSON конфига (если указан) ---
    if args.config is not None:
        import json
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key, value in cfg.items():
            # Пропускаем служебные поля с "_" (комментарии, usage и т.д.)
            if key.startswith("_"):
                continue
            # Перезаписываем все значения из конфига
            # Явные флаги командной строки приоритетнее — их нужно передавать явно
            setattr(args, key, value)
        print(f"📋 Конфиг загружен: {args.config}")

    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Режим быстрого теста ---
    if args.test_run:
        print("⚡ ТЕСТОВЫЙ ПРОГОН — урезанные параметры для проверки работоспособности")
        print("   Цель: убедиться что всё запускается, loss считается, градиент течёт")
        print("   Не ожидай улучшения качества — слишком мало данных!\n")
        args.max_train_samples = args.max_train_samples or 20  # 20 вместо 50
        args.max_eval_samples  = 10
        args.epochs            = 1
        args.max_new_tokens    = 64   # 64 вместо 128 — вдвое быстрее генерация
        args.num_generations   = 2
        args.batch_size        = 1

    # --- Загрузка модели ---
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
    )

    # --- Применяем TinyLoRA ---
    print(f"\n🔧 Применяем TinyLoRA (proj_dim={args.proj_dim}, rank={args.rank})")
    model, shared_v = apply_tinylora_to_model(
        model,
        rank=args.rank,
        proj_dim=args.proj_dim,
        seed=args.seed,
    )

    # Диагностика
    stats = count_trainable_params(model)
    print(f"\n📊 Статистика параметров:")
    print(f"   Обучаемых  : {stats['trainable']:,} (должно быть {args.proj_dim})")
    print(f"   shared_v   : {stats['shared_v_params']:,} параметров ← это наш вектор")
    print(f"   Заморожено : {stats['frozen']:,} параметров")
    print(f"   Доля       : {stats['ratio']:.2e}")
    if stats['trainable'] != args.proj_dim:
        print(f"⚠️  ВНИМАНИЕ: ожидалось {args.proj_dim}, получено {stats['trainable']}!")

    # --- Данные ---
    print(f"\n📚 Загружаем GSM8K...")
    train_dataset = prepare_dataset(
        tokenizer, split="train", max_samples=args.max_train_samples
    )
    eval_dataset = prepare_dataset(
        tokenizer, split="test", max_samples=args.max_eval_samples
    )
    print(f"   Train: {len(train_dataset)} примеров")
    print(f"   Eval:  {len(eval_dataset)} примеров")

    # --- Конфигурация GRPO ---
    # Параметры для trl 0.29.x
    # gradient_accumulation: для теста 1 (быстро), для полного обучения 4
    accum_steps = 1 if args.test_run else 4

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=accum_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,                # было warmup_ratio — убрали в trl v5.2
        # Автоматически выбираем точность в зависимости от GPU
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        # gradient_checkpointing отключаем — для малых моделей только замедляет
        gradient_checkpointing=False,
        # GRPO-специфичные параметры
        num_generations=args.num_generations,
        # generation_batch_size должен быть кратен num_generations
        generation_batch_size=args.num_generations,
        max_completion_length=args.max_new_tokens,
        temperature=1.0,
        # beta=0.0 — дефолт, KL penalty выключен
        # Отключаем thinking mode у Qwen3
        chat_template_kwargs={"enable_thinking": False},
        # Логирование
        logging_steps=1,               # каждый шаг — чтобы видеть прогресс
        eval_steps=50,
        save_strategy="no",      # полностью отключаем автосохранение трансформера
                                 # оно не работает с shared tensors (наш shared_v)
                                 # сохранение делаем вручную через SaveVCallback
        report_to="none",
        seed=args.seed,
    )

    # --- Инициализация GRPO Trainer ---
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_correct_answer,
    )

    # --- Колбэк для замера времени шагов ---
    import time
    from transformers import TrainerCallback

    class TimingCallback(TrainerCallback):
        def __init__(self):
            self.step_start = None
            self.times = []

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_start = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if self.step_start is not None:
                elapsed = time.time() - self.step_start
                self.times.append(elapsed)
                if len(self.times) <= 3 or len(self.times) % 5 == 0:
                    avg = sum(self.times) / len(self.times)
                    remaining = (args.max_steps - state.global_step) * avg
                    print(f"   ⏱ Шаг {state.global_step}: {elapsed:.1f} сек "
                          f"| среднее: {avg:.1f} сек "
                          f"| осталось: ~{remaining/60:.0f} мин")

    timing_cb = TimingCallback()

    # --- Колбэк для сохранения shared_v каждые 100 шагов ---
    # Стандартный save_pretrained не работает с shared tensors,
    # поэтому сохраняем только наш вектор v вручную
    class SaveVCallback(TrainerCallback):
        def __init__(self, model, output_dir, save_every=100):
            self.model = model
            self.output_dir = output_dir
            self.save_every = save_every

        def _save(self, step):
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, f"tinylora_v_step{step}.pt")
            torch.save({
                "shared_v": self.model.tinylora_shared_v.data.cpu(),
                "step": step,
            }, path)
            print(f"   💾 Сохранён checkpoint: {path}")

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.save_every == 0:
                self._save(state.global_step)

        def on_train_end(self, args, state, control, **kwargs):
            self._save(state.global_step)

    save_cb = SaveVCallback(model, args.output_dir, save_every=100)

    # --- Запуск обучения ---
    print(f"\n🚀 Начинаем обучение TinyLoRA с GRPO!")
    print(f"   Обучаемых параметров: {stats['trainable']}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Эпох: {args.epochs}")
    print(f"   Батч: {args.batch_size} x {grpo_config.gradient_accumulation_steps} = "
          f"{args.batch_size * grpo_config.gradient_accumulation_steps} эффективный")
    print("-" * 60)

    trainer.add_callback(timing_cb)
    trainer.add_callback(save_cb)
    trainer.train()

    # --- Финальное сохранение ---
    print(f"\n💾 Сохраняем финальные результаты в {args.output_dir}")
    shared_v = model.tinylora_shared_v
    torch.save(
        {
            "shared_v": shared_v.data.cpu(),
            "proj_dim": args.proj_dim,
            "rank": args.rank,
            "model_name": args.model_name,
            "seed": args.seed,
        },
        os.path.join(args.output_dir, "tinylora_v.pt"),
    )
    print(f"✅ Вектор v сохранён: {args.output_dir}/tinylora_v.pt")
    print(f"   Размер: {shared_v.numel() * 2} байт (bf16)")
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Обучение завершено!")

if __name__ == "__main__":
    main()
