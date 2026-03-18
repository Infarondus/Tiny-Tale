"""
Скрипт оценки модели с обученным TinyLoRA вектором на GSM8K.

Использование:
    python evaluate.py --checkpoint ./output/tinylora_v.pt
    python evaluate.py --checkpoint ./output/tinylora_v.pt --num_samples 500
"""

import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

from tinylora import apply_tinylora_to_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Путь к файлу tinylora_v.pt")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Сколько примеров из test-сета оценивать")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--compare_baseline", action="store_true", default=True,
                        help="Также оценить базовую модель без TinyLoRA")
    return parser.parse_args()


def extract_answer(text: str) -> str | None:
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\-\d]+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def evaluate_model(model, tokenizer, dataset, batch_size: int, max_new_tokens: int,
                   label: str = "Model") -> float:
    """Прогоняет модель по датасету и считает accuracy."""
    correct = 0
    total = 0

    print(f"\n📊 Оцениваем: {label}")
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i: i + batch_size]
        prompts = batch["prompt"]
        gt_answers = batch["answer"]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,      # greedy для детерминированной оценки
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Декодируем только новые токены
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for pred_text, gt in zip(decoded, gt_answers):
            predicted = extract_answer(pred_text)
            gt_clean = str(gt).replace(",", "").strip()
            if predicted == gt_clean:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"   ✅ Accuracy: {correct}/{total} = {acc:.1%}")
    return acc


def main():
    args = parse_args()

    # Загружаем метаданные из checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = ckpt["model_name"]
    proj_dim = ckpt["proj_dim"]
    rank = ckpt["rank"]
    seed = ckpt["seed"]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"   Модель: {model_name}")
    print(f"   proj_dim (обучаемых params): {proj_dim}")
    print(f"   SVD rank: {rank}")

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                               padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Загружаем датасет
    print(f"\n📚 Загружаем GSM8K test ({args.num_samples} примеров)...")
    raw = load_dataset("openai/gsm8k", "main", split="test")
    raw = raw.select(range(min(args.num_samples, len(raw))))

    def format_example(example):
        answer_text = example["answer"]
        gt_answer = extract_answer(answer_text)
        messages = [
            {"role": "system", "content": "Solve the math problem step by step. Write final answer after '####'."},
            {"role": "user", "content": example["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "answer": gt_answer}

    dataset = raw.map(format_example, remove_columns=raw.column_names)

    # Загружаем модель
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    results = {}

    # Оценка базовой модели (без TinyLoRA)
    if args.compare_baseline:
        baseline_acc = evaluate_model(
            model, tokenizer, dataset,
            args.batch_size, args.max_new_tokens,
            label="Baseline (без TinyLoRA)"
        )
        results["baseline"] = baseline_acc

    # Применяем TinyLoRA и загружаем обученный вектор v
    model, shared_v = apply_tinylora_to_model(
        model, rank=rank, proj_dim=proj_dim, seed=seed
    )

    # Восстанавливаем обученные параметры
    with torch.no_grad():
        shared_v.data.copy_(ckpt["shared_v"].to(shared_v.dtype))

    tinylora_acc = evaluate_model(
        model, tokenizer, dataset,
        args.batch_size, args.max_new_tokens,
        label=f"TinyLoRA ({proj_dim} params)"
    )
    results["tinylora"] = tinylora_acc

    # Итоги
    print("\n" + "="*50)
    print("📈 ИТОГИ:")
    if "baseline" in results:
        print(f"   Baseline:  {results['baseline']:.1%}")
    print(f"   TinyLoRA:  {results['tinylora']:.1%}")
    if "baseline" in results:
        delta = results["tinylora"] - results["baseline"]
        print(f"   Улучшение: {delta:+.1%}")
        recovery = delta / (0.95 - results["baseline"]) * 100  # относительно ~95% ceiling
        print(f"   Recovery:  ~{recovery:.0f}% от потенциального улучшения")
    print("="*50)


if __name__ == "__main__":
    main()