"""
Замеряем скорость генерации модели напрямую,
без GRPO и TinyLoRA — чистый инференс.
Это покажет реальный потолок скорости на твоём железе.
"""
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./models/Qwen_3_06B"  # поменяй если папка называется иначе

print("Загружаем модель...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()

vram = round(torch.cuda.memory_allocated(0) / 1e9, 2)
print(f"Модель загружена | VRAM: {vram} GB")

# Тестовый промпт — простая математическая задача
prompt = "Solve step by step: Janet has 3 apples. She buys 5 more. How many does she have?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
input_len = inputs["input_ids"].shape[1]

print(f"\nДлина промпта: {input_len} токенов")
print("-" * 50)

# Тест 1: одиночная генерация
print("\n📊 Тест 1: одиночная генерация (batch=1, 64 токена)")
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
torch.cuda.synchronize()
t1 = time.time()
generated = out.shape[1] - input_len
speed = generated / (t1 - t0)
print(f"   Время: {t1-t0:.2f} сек | Токенов: {generated} | Скорость: {speed:.1f} tok/s")

# Тест 2: батч из 2 (как в GRPO с num_generations=2)
print("\n📊 Тест 2: батч из 2 промптов (как GRPO num_generations=2)")
inputs2 = tokenizer([text, text], return_tensors="pt", padding=True).to("cuda:0")
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out2 = model.generate(**inputs2, max_new_tokens=64, do_sample=True,
                          temperature=1.0, pad_token_id=tokenizer.eos_token_id)
torch.cuda.synchronize()
t1 = time.time()
total_tokens = (out2.shape[1] - inputs2["input_ids"].shape[1]) * 2
speed2 = total_tokens / (t1 - t0)
print(f"   Время: {t1-t0:.2f} сек | Токенов: {total_tokens} | Скорость: {speed2:.1f} tok/s")

# Тест 3: батч из 4 (как в GRPO с num_generations=4)
print("\n📊 Тест 3: батч из 4 промптов (как GRPO num_generations=4)")
inputs4 = tokenizer([text]*4, return_tensors="pt", padding=True).to("cuda:0")
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out4 = model.generate(**inputs4, max_new_tokens=64, do_sample=True,
                          temperature=1.0, pad_token_id=tokenizer.eos_token_id)
torch.cuda.synchronize()
t1 = time.time()
total_tokens4 = (out4.shape[1] - inputs4["input_ids"].shape[1]) * 4
speed4 = total_tokens4 / (t1 - t0)
print(f"   Время: {t1-t0:.2f} сек | Токенов: {total_tokens4} | Скорость: {speed4:.1f} tok/s")

print("\n" + "="*50)
print("📈 ИТОГ:")
print(f"   Одиночная генерация:  {speed:.1f} tok/s")
print(f"   Батч 2 (GRPO test):   {speed2:.1f} tok/s")
print(f"   Батч 4 (GRPO full):   {speed4:.1f} tok/s")
expected_step_time = (64 * 2) / speed2
print(f"\n   Ожидаемое время шага GRPO (test): ~{expected_step_time:.0f} сек")
expected_full = (256 * 4) / speed4
print(f"   Ожидаемое время шага GRPO (full): ~{expected_full:.0f} сек")
EOF
