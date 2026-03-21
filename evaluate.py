"""
Быстрая оценка модели с TinyLoRA на GSM8K test split.
Запуск:
    python evaluate.py --checkpoint ./results/Qwen3-0.6B/tinylora_v.pt
    python evaluate.py --checkpoint ./results/Qwen3-0.6B/tinylora_v.pt --num_samples 50
"""
import argparse
import re
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tinylora import apply_tinylora_to_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Сколько задач оценивать (default: 100, ~10 мин)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Батч для генерации (default: 4)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def extract_answer(text: str):
    match = re.search(r"####\s*<?\s*([\-\d,\.]+)\s*>?", text)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"\\boxed\{([\-\d,\.]+)\}", text)
    if match:
        return match.group(1).replace(",", "").strip()
    tail = text[-200:].replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", tail)
    return numbers[-1] if numbers else None


def evaluate(model, tokenizer, prompts, answers, batch_size, max_new_tokens, label):
    correct = 0
    total = 0
    t0 = time.time()

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_answers = answers[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=1024,  # промпт может быть длинным
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, (out, gt) in enumerate(zip(outputs, batch_answers)):
            input_len = inputs["input_ids"].shape[1]
            generated = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            predicted = extract_answer(generated)
            gt_clean = str(gt).replace(",", "").strip()
            if predicted == gt_clean:
                correct += 1
            total += 1

        elapsed = time.time() - t0
        speed = total / elapsed
        eta = (len(prompts) - total) / speed if speed > 0 else 0
        print(f"   {label}: {total}/{len(prompts)} | "
              f"{correct/total:.1%} | ~{eta/60:.0f} мин осталось", end="\r")

    acc = correct / total if total > 0 else 0
    print(f"\n   ✅ {label}: {correct}/{total} = {acc:.1%}          ")
    return acc


def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = ckpt["model_name"]
    proj_dim   = ckpt["proj_dim"]
    rank       = ckpt["rank"]
    seed       = ckpt["seed"]

    print(f"📦 Модель: {model_name} | proj_dim: {proj_dim}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Загружаем датасет
    print(f"📚 Загружаем {args.num_samples} задач из GSM8K test...")
    raw = load_dataset("openai/gsm8k", "main", split="test")
    raw = raw.select(range(min(args.num_samples, len(raw))))

    def get_answer(text):
        m = re.search(r"####\s*([\-\d,\.]+)", text)
        return m.group(1).replace(",", "").strip() if m else None

    prompts, answers = [], []
    for ex in raw:
        msgs = [
            {"role": "system", "content": "Solve the math problem. Show your work briefly. End with: #### <number>"},
            {"role": "user",   "content": ex["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
        answers.append(get_answer(ex["answer"]))

    # Загружаем модель
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    # --- Baseline ---
    print("\n📊 Оцениваем baseline (без TinyLoRA)...")
    baseline_acc = evaluate(
        model, tokenizer, prompts, answers,
        args.batch_size, args.max_new_tokens, "Baseline"
    )

    # --- TinyLoRA ---
    print("\n📊 Применяем TinyLoRA и оцениваем...")
    model, shared_v = apply_tinylora_to_model(
        model, rank=rank, proj_dim=proj_dim, seed=seed
    )

    # После применения TinyLoRA новые буферы (U_r, S_r, V_r, P) создаются на CPU.
    # Нужно явно перенести всю модель на GPU снова.
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(target_device)

    # Загружаем обученный вектор v
    with torch.no_grad():
        model.tinylora_shared_v.data.copy_(
            ckpt["shared_v"].to(dtype=model.tinylora_shared_v.dtype,
                                device=target_device)
        )

    tinylora_acc = evaluate(
        model, tokenizer, prompts, answers,
        args.batch_size, args.max_new_tokens, "TinyLoRA"
    )

    # --- Итог ---
    print("\n" + "="*50)
    print(f"📈 РЕЗУЛЬТАТ ({model_name}):")
    print(f"   Baseline : {baseline_acc:.1%}")
    print(f"   TinyLoRA : {tinylora_acc:.1%}")
    delta = tinylora_acc - baseline_acc
    print(f"   Δ        : {delta:+.1%}")
    print("="*50)


if __name__ == "__main__":
    main()
