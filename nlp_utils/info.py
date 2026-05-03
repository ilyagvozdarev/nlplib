import torch
from IPython.display import display, HTML


def params_stats(model):

    stats = {}
    stats['total_non_buffer_params'] = sum(p.numel() for p in model.parameters())
    stats['total_trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats['total_params'] = sum(params.numel() for _, params in model.state_dict().items())
    stats['total_buffer_params'] = stats['total_params'] - stats['total_non_buffer_params']

    stats['non_buffer_params_names'] = set(name for name, _ in model.named_parameters())
    stats['trainable_params_names'] = set(name for name, p in model.named_parameters() if p.requires_grad)
    stats['params_names'] = set(model.state_dict())
    stats['buffer_params_names'] = stats['params_names'] - stats['non_buffer_params_names']

    return stats


def named_modules(model):
    return model.named_modules()


def print_trainable_parameters(model, config):
    lora_model = get_peft_model(model, config)
    return lora_model.print_trainable_parameters()


def print_gpu_stats():

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(
        torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
    )
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    print(torch.cuda.memory_summary())




def print_gpu_memory():
    if not torch.cuda.is_available():
        print("🚫 CUDA не доступна. GPU не обнаружены.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"🔢 Обнаружено {num_gpus} GPU:\n")

    # HTML-стилизация для красивого вывода в Jupyter
    html_output = """
    <style>
        .gpu-table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        .gpu-table th, .gpu-table td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        .gpu-table th { background-color: #f2f2f2; }
        .low-mem { background-color: #ffebee; color: #c62828; font-weight: bold; }
        .med-mem { background-color: #fff8e1; color: #ff8f00; }
        .high-mem { background-color: #e8f5e8; color: #2e7d32; }
    </style>
    <table class="gpu-table">
        <tr>
            <th>GPU</th>
            <th>Название</th>
            <th>Свободно (GB)</th>
            <th>Всего (GB)</th>
            <th>Загрузка (%)</th>
        </tr>
    """

    for i in range(num_gpus):
        # Получаем свободную и общую память в байтах
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        used_gb = total_gb - free_gb
        usage_percent = (used_gb / total_gb) * 100

        # Определяем класс для подсветки
        if free_gb < 5:
            mem_class = "low-mem"
        elif free_gb < 10:
            mem_class = "med-mem"
        else:
            mem_class = "high-mem"

        # Добавляем строку таблицы
        html_output += f"""
        <tr>
            <td><strong>cuda:{i}</strong></td>
            <td>{torch.cuda.get_device_name(i)}</td>
            <td class="{mem_class}">{free_gb:.2f} GB</td>
            <td>{total_gb:.2f} GB</td>
            <td>{usage_percent:.1f}%</td>
        </tr>
        """

    html_output += "</table>"
    display(HTML(html_output))

# Запуск
print_gpu_memory()