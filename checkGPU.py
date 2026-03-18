import torch

print(f"Количество GPU видит torch: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

print(f"\nТекущее устройство: {torch.cuda.current_device()}")
print(f"CUDA доступна: {torch.cuda.is_available()}")

# Пробуем создать тензор на GPU и проверяем где он оказался
x = torch.tensor([1.0, 2.0, 3.0]).cuda()
print(f"\nТестовый тензор на: {x.device}")
print(f"VRAM после создания тензора: {round(torch.cuda.memory_allocated(0)/1e9, 4)} GB")

# Проверяем возможности GPU
props = torch.cuda.get_device_properties(0)
print(f"\nСвойства GPU 0:")
print(f"  Название: {props.name}")
print(f"  Всего VRAM: {round(props.total_memory/1e9, 2)} GB")
print(f"  Compute capability: {props.major}.{props.minor}")