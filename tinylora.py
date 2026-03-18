"""
TinyLoRA — реализация метода из статьи "Learning to Reason in 13 Parameters"
(Morris et al., 2026, arxiv 2602.04118)

Идея:
  Для каждой весовой матрицы W делаем SVD: W = U @ diag(S) @ V^T
  Обновление веса: W' = W + U @ diag(S) @ (sum_i v_i * P_i) @ V^T
  где v — крошечный обучаемый вектор (например, размер 1),
      P_i — фиксированные случайные матрицы ранга r.
  Через weight tying (sharing) v одного вектора на ВСЕ слои
  получаем total_params = u (например, 13 при u=13 или 1 при u=1).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# Пробуем импортировать bitsandbytes — он может быть не установлен
try:
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    Linear4bit = None
    Linear8bitLt = None


def _is_linear(module: nn.Module) -> bool:
    """Проверяет является ли модуль любым видом линейного слоя."""
    if isinstance(module, nn.Linear):
        return True
    if BNB_AVAILABLE:
        if isinstance(module, (Linear4bit, Linear8bitLt)):
            return True
    return False


def _get_weight_tensor(module: nn.Module) -> torch.Tensor:
    """
    Извлекает веса из модуля в float32 на CPU.
    Обрабатывает как обычный nn.Linear, так и bitsandbytes квантованные слои.
    """
    if BNB_AVAILABLE and isinstance(module, Linear4bit):
        # Linear4bit хранит веса в упакованном виде.
        # dequantize() возвращает их в float16/bfloat16 — переводим в float32.
        # Важно: это только для SVD при инициализации, не для forward pass.
        try:
            import bitsandbytes.functional as F_bnb
            weight = F_bnb.dequantize_4bit(
                module.weight.data,
                module.weight.quant_state,
            ).float().cpu()
        except Exception:
            # Fallback: берём как есть и конвертируем
            weight = module.weight.data.float().cpu()
        return weight

    if BNB_AVAILABLE and isinstance(module, Linear8bitLt):
        weight = module.weight.data.float().cpu()
        return weight

    # Обычный nn.Linear
    return module.weight.data.float().cpu()


class TinyLoRALayer(nn.Module):
    """
    Один TinyLoRA адаптер для одной линейной матрицы W (d_out x d_in).

    Параметры:
        weight      : исходная замороженная матрица (d_out x d_in)
        rank        : ранг SVD (r), используется для аппроксимации
        proj_dim    : размер обучаемого вектора v (u в статье)
        shared_v    : если передан — используем ОБЩИЙ вектор v (weight tying)
                      именно это позволяет получить 13 параметров на всю модель
        seed        : для воспроизводимости случайных матриц P
    """

    def __init__(
        self,
        weight: torch.Tensor,
        rank: int = 1,
        proj_dim: int = 13,
        shared_v: Optional[nn.Parameter] = None,
        seed: int = 42,
    ):
        super().__init__()

        d_out, d_in = weight.shape
        self.rank = rank
        self.proj_dim = proj_dim
        self.d_out = d_out
        self.d_in = d_in

        # --- SVD разложение исходной матрицы ---
        # Делаем один раз при инициализации на CPU (float32 для точности SVD)
        W_cpu = weight.detach().float().cpu()
        U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)

        # Берём только top-r сингулярных компонент
        r = min(rank, min(d_out, d_in))
        U_r = U[:, :r]       # (d_out, r)
        S_r = S[:r]          # (r,)
        V_r = Vh[:r, :].T    # (d_in, r)

        # Регистрируем как float32 буферы — они автоматически переедут на GPU
        # вместе с моделью при model.to("cuda") или device_map="cuda:0"
        self.register_buffer("U_r", U_r.float())
        self.register_buffer("S_r", S_r.float())
        self.register_buffer("V_r", V_r.float())

        # --- Случайные проекционные матрицы P_i (фиксированные) ---
        rng = torch.Generator()
        rng.manual_seed(seed)
        P = torch.randn(proj_dim, r, r, generator=rng)
        P = P / (P.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        self.register_buffer("P", P.float())  # тоже float32, тоже поедет на GPU

        # --- Обучаемый вектор v ---
        if shared_v is not None:
            self.v = shared_v
        else:
            self.v = nn.Parameter(torch.zeros(proj_dim))

        # Запоминаем целевой dtype для финального приведения
        self._orig_dtype = weight.dtype

    def compute_delta_W(self) -> torch.Tensor:
        """
        Вычисляет ΔW = U_r @ diag(S_r) @ R @ V_r^T
        где R = sum_i v_i * P_i

        Все тензоры (U_r, S_r, V_r, P) живут на GPU как registered buffers.
        v (shared_v) тоже на GPU — зарегистрирован в корневой модели.
        Всё вычисление происходит на GPU — никаких переносов CPU↔GPU.
        """
        # Приводим v к float32 для численной стабильности
        v = self.v.float()

        # R = sum_i v_i * P_i, форма (r, r) — всё на GPU
        R = torch.einsum("u, u r k -> r k", v, self.P)

        # ΔW = U_r @ diag(S_r) @ R @ V_r^T, форма (d_out, d_in)
        US = self.U_r * self.S_r.unsqueeze(0)
        delta_W = US @ R @ self.V_r.T

        # Приводим к dtype оригинальных весов (bfloat16)
        return delta_W.to(self._orig_dtype)



class TinyLoRALinear(nn.Module):
    """
    Обёртка вокруг любого линейного слоя.
    Оригинальный слой замораживается полностью.
    Обучается только shared_v который живёт в корневом модуле.
    """

    def __init__(
        self,
        linear: nn.Module,
        weight_tensor: torch.Tensor,
        rank: int = 1,
        proj_dim: int = 13,
        shared_v: Optional[nn.Parameter] = None,
        seed: int = 42,
    ):
        super().__init__()

        # Замораживаем оригинальный слой
        self.original_layer = linear
        for p in self.original_layer.parameters():
            p.requires_grad_(False)

        self.adapter = TinyLoRALayer(
            weight=weight_tensor,
            rank=rank,
            proj_dim=proj_dim,
            shared_v=shared_v,
            seed=seed,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original_layer(x)
        # delta_W уже на том же устройстве что и x — буферы переехали с моделью
        delta_W = self.adapter.compute_delta_W()
        delta_out = nn.functional.linear(x, delta_W.to(x.dtype))
        return base_out + delta_out


def apply_tinylora_to_model(
    model: nn.Module,
    rank: int = 1,
    proj_dim: int = 13,
    target_modules: Optional[list] = None,
    seed: int = 42,
) -> tuple[nn.Module, nn.Parameter]:
    """
    Применяет TinyLoRA ко всем целевым линейным слоям модели.

    Ключевой момент: создаём ОДИН shared_v на всю модель.
    Это и есть weight tying — все слои делят одни и те же 13 параметров.

    Аргументы:
        model          : модель HuggingFace (Qwen3)
        rank           : ранг SVD (рекомендуется 1 для минимума параметров)
        proj_dim       : размер вектора v = итоговое число обучаемых параметров
        target_modules : список имён модулей для замены (None = авто-определение)
        seed           : seed для воспроизводимости

    Возвращает:
        (model, shared_v) — модель с адаптерами и сам обучаемый параметр
    """
    if target_modules is None:
        # Стандартные attention + MLP проекции в Qwen/LLaMA архитектурах
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    # Сначала замораживаем ВСЕ параметры модели
    for p in model.parameters():
        p.requires_grad_(False)

    # Создаём shared_v и регистрируем его прямо в корневой модели
    # Только так PyTorch увидит его через model.parameters()
    shared_v = nn.Parameter(torch.zeros(proj_dim))
    model.register_parameter("tinylora_shared_v", shared_v)

    replaced_count = 0
    for name, module in model.named_modules():
        # Проверяем, является ли модуль нужным линейным слоем
        module_name = name.split(".")[-1]
        if not _is_linear(module):
            continue
        if module_name not in target_modules:
            continue

        # Получаем веса (деквантизируем если нужно)
        try:
            weight_tensor = _get_weight_tensor(module)
        except Exception as e:
            print(f"⚠️  Пропускаем {name}: не удалось извлечь веса ({e})")
            continue

        if weight_tensor.dim() != 2:
            continue

        # Находим родительский модуль и заменяем слой → TinyLoRALinear
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        layer_seed = seed + replaced_count  # разные P для каждого слоя
        tinylora_linear = TinyLoRALinear(
            linear=module,
            weight_tensor=weight_tensor,
            rank=rank,
            proj_dim=proj_dim,
            shared_v=shared_v,
            seed=layer_seed,
        )
        setattr(parent, parts[-1], tinylora_linear)
        replaced_count += 1

    print(f"✅ TinyLoRA применён к {replaced_count} линейным слоям")
    print(f"✅ Обучаемых параметров: {proj_dim} (shared_v размером {proj_dim})")
    print(f"✅ Замороженных параметров: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    return model, shared_v


def count_trainable_params(model: nn.Module) -> dict:
    """Подсчёт параметров для диагностики."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    # Отдельно показываем shared_v
    shared_v_params = 0
    if hasattr(model, "tinylora_shared_v"):
        shared_v_params = model.tinylora_shared_v.numel()
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": trainable + frozen,
        "shared_v_params": shared_v_params,
        "ratio": trainable / (trainable + frozen) if (trainable + frozen) > 0 else 0,
    }
