"""
#### model params estimation ####
Считает только статическую часть — то, что лежит в памяти постоянно.
Активации сюда не входят: они зависят от длины батча, а не от модели.

stats_from_model / stats_from_config:
    Теоретическая оценка количества параметров и памяти для них (см. ParamStats): 
    - trainable, frozen (в том числе с учетом округления caching allocator)
    - buffers 
    Два режима:
        stats_from_model(model)      — по заданной модели
        stats_from_config(model_id)  — по конфигу с HF, без скачивания весов

training_footprint:
    - stats_from_model / stats_from_config
    - градиенты (trainable параметры), состояния оптимизатора (в том числе с учетом округления caching allocator)
    - mixed precision

format_report - training_footprint в виде строки


#### GPU info ####
cuda_report: срез памяти по всем GPU в виде строки: total/free, allocated, reseved, peak и т.д. (см. DeviceMemory)
gpu_memory_html: срез памяти по всем GPU в виде html
print_gpu_memory: срез памяти по всем GPU в виде html или если произошла ошибка то в виде строки 
"""

from __future__ import annotations
from IPython.display import display, HTML

from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch import nn



# AdamW держит exp_avg и exp_avg_sq, SGD с моментом — только momentum_buffer.
OPTIMIZER_STATES = {
    "adamw": 2,
    "adam": 2,
    "sgd_momentum": 1,
    "sgd": 0,
    "adafactor": 0,  # факторизованные состояния, размер O(n+m) вместо O(n*m)
}


# Кеширующий аллокатор (caching allocator) CUDA выдаёт блоки не байт-в-байт: мелкие округляются до 512 B,
# крупные (свыше 10 МБ) — до кратного 2 МБ. Отсюда систематическая разница между
# теоретическим размером и torch.cuda.memory_allocated() — например на XLM-R это ~1.6% и почти
# целиком приходится на матрицу эмбеддингов.
_MIN_BLOCK = 512
_MIN_LARGE_ALLOC = 10 * 2**20
_ROUND_LARGE = 2 * 2**20


def allocator_bytes(n_bytes: int) -> int:
    """Сколько реально займёт блок такого размера в кеширующем аллокаторе CUDA."""
    if n_bytes > _MIN_LARGE_ALLOC:
        return -(-n_bytes // _ROUND_LARGE) * _ROUND_LARGE
    return -(-n_bytes // _MIN_BLOCK) * _MIN_BLOCK


@dataclass
class ParamStats:
    """
    Разбор параметров модели по dtype и обучаемости.
    """
    # количество элементов тензоров (маппинг тип элемента -> количество). 
    # (frozen/trainable - named_parameters(), buffers - named_buffers())
    #   frozen: с requires_grad False
    #   trainable: с requires_grad True
    name: str
    trainable: dict[torch.dtype, int] = field(default_factory=lambda: defaultdict(int))
    frozen: dict[torch.dtype, int] = field(default_factory=lambda: defaultdict(int))
    buffers: dict[torch.dtype, int] = field(default_factory=lambda: defaultdict(int))

    trainable_numels: list[int] = field(default_factory=list)
    by_module: dict[str, int] = field(default_factory=dict)

    # размеры отдельных тензоров — нужны, чтобы посчитать округление аллокатора
    tensor_bytes: list[int] = field(default_factory=list)
    trainable_tensor_bytes: list[int] = field(default_factory=list)

    @property
    def n_trainable(self) -> int:
        return sum(self.trainable.values())

    @property
    def n_frozen(self) -> int:
        return sum(self.frozen.values())

    @property
    def n_params(self) -> int:
        return self.n_trainable + self.n_frozen

    @property
    def n_buffers(self) -> int:
        return sum(self.buffers.values())

    def _bytes(self, counts: dict[torch.dtype, int]) -> int:
        return sum(n * dtype.itemsize for dtype, n in counts.items())

    @property
    def weights_bytes(self) -> int:
        """Размер самих весов — то, что занимает модель сразу после .to(device)."""
        return self._bytes(self.trainable) + self._bytes(self.frozen) + self._bytes(self.buffers)

    @property
    def trainable_bytes(self) -> int:
        return self._bytes(self.trainable)

    @property
    def weights_allocated_bytes(self) -> int:
        """Веса с поправкой на гранулярность аллокатора — то, что покажет memory_allocated()."""
        return sum(allocator_bytes(n) for n in self.tensor_bytes)


def _group_key(param_name: str) -> str:
    """Схлопывает имя параметра в имя группы, чтобы одинаковые блоки слоёв слились.

    Числовые сегменты — это индексы в nn.ModuleList/nn.Sequential (номер слоя) который считаем одним модулем, 
    заменяем эти индексы на "*";
    последний сегмент — это weight/bias, он для группировки не нужен.

        encoder.layer.0.attention.self.query.weight -> encoder.layer.*.attention.self.query
        encoder.layer.7.attention.self.query.bias   -> encoder.layer.*.attention.self.query
    """
    parts = param_name.split(".")
    if len(parts) > 1:
        parts = parts[:-1]
    return ".".join("*" if p.isdigit() else p for p in parts)


def stats_from_model(model: nn.Module, top_modules: int = 0) -> ParamStats:
    """
    Разбирает живую модель. SentenceTransformer тоже подойдёт — это nn.Module.

    Args:
        top_modules: сохранять только top_modules слоев с наибольшим количеством параметров

    Notes:
    У SentenceTransformer параметры лежат в подмодулях (Transformer, Pooling, ...),
    named_parameters() их обходит рекурсивно, так что разворачивать ничего не надо.
    """
    st = ParamStats(name=type(model).__name__)

    for _, p in model.named_parameters():
        bucket = st.trainable if p.requires_grad else st.frozen
        bucket[p.dtype] += p.numel()
        st.tensor_bytes.append(p.numel() * p.dtype.itemsize)
        if p.requires_grad:
            st.trainable_numels.append(p.numel())
            st.trainable_tensor_bytes.append(p.numel() * p.dtype.itemsize)

    for _, b in model.named_buffers():
        st.buffers[b.dtype] += b.numel()
        st.tensor_bytes.append(b.numel() * b.dtype.itemsize)

    if top_modules:
        sizes: dict[str, int] = defaultdict(int)
        for name, p in model.named_parameters():
            sizes[_group_key(name)] += p.numel() * p.dtype.itemsize
        st.by_module = dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:top_modules])

    return st


def stats_from_config(model_id: str, dtype: torch.dtype = torch.float32, **kwargs) -> ParamStats:
    """Считает по конфигу с HF, не скачивая веса — модель строится на meta-device.

    Полезно, чтобы прикинуть влезет ли, до того как тянуть чекпоинт.
    """
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_id, **kwargs)
    with torch.device("meta"):
        try:
            model = AutoModel.from_config(config, dtype=dtype)
        except TypeError:  # transformers < 5 принимает только torch_dtype
            model = AutoModel.from_config(config, torch_dtype=dtype)

    st = stats_from_model(model)
    st.name = model_id
    return st


def training_footprint(
    stats: ParamStats,
    optimizer: str = "adamw",
    amp: str | None = None,
    optimizer_dtype: torch.dtype = torch.float32,
    allocator_aware: bool = False,
) -> dict[str, int]:
    """
    Статическая память под обучение, в байтах.

    optimizer: ключ из OPTIMIZER_STATES.
    amp: None | "fp16" | "bf16" — режим torch.amp (fp16=True/bf16=True в HF).

    optimizer_dtype: torch создаёт состояния через zeros_like(p), то есть в dtype
        параметра. Указывайте явно, если используете оптимизатор с fp32-состояниями
        поверх половинных весов.

    allocator_aware: учитывать округление блоков кеширующего аллокатора. Без него
        получается теоретический минимум, с ним — то, что покажет memory_allocated().
    """
    if optimizer not in OPTIMIZER_STATES:
        raise ValueError(f"Unknown optimizer {optimizer!r}, expected one of {sorted(OPTIMIZER_STATES)}.")
    if amp not in (None, "fp16", "bf16"):
        raise ValueError(f"amp must be None, 'fp16' or 'bf16', got {amp!r}.")

    n_states = OPTIMIZER_STATES[optimizer]

    if allocator_aware:
        weights = stats.weights_allocated_bytes
        # По одному тензору градиента на обучаемый параметр, того же размера.
        grads = sum(allocator_bytes(n) for n in stats.trainable_tensor_bytes)
        # И по n_states тензоров состояний, но уже в dtype оптимизатора.
        optim = n_states * sum(allocator_bytes(n * optimizer_dtype.itemsize) for n in stats.trainable_numels)
    else:
        weights = stats.weights_bytes
        grads = stats.trainable_bytes
        optim = stats.n_trainable * n_states * optimizer_dtype.itemsize

    out = {
        "weights": weights,
        "gradients": grads,
        "optimizer": optim,
    }

    if amp is not None:
        # autocast кеширует приведённые к половинной точности веса на время forward.
        # Копия транзиентная, но живёт весь шаг, поэтому в пике её надо учитывать.
        out["autocast_cache"] = stats.n_trainable * 2

    out["total"] = sum(out.values())
    return out


def gib(n_bytes: int) -> float:
    return n_bytes / 2**30


def format_report(
    stats: ParamStats,
    optimizer: str = "adamw",
    amp: str | None = None,
    allocator_aware: bool = False,
) -> str:
    fp = training_footprint(stats, optimizer=optimizer, amp=amp, allocator_aware=allocator_aware)
    dtypes = ", ".join(
        f"{str(d).removeprefix('torch.')}: {n:,}"
        for d, n in sorted({**stats.trainable, **stats.frozen}.items(), key=lambda kv: -kv[1])
    )

    lines = [
        f"{stats.name}",
        f"  параметров      {stats.n_params:>15,}  ({dtypes})",
        f"  обучаемых       {stats.n_trainable:>15,}",
        f"  замороженных    {stats.n_frozen:>15,}",
        f"  буферов         {stats.n_buffers:>15,}",
        "",
        f"  статическая память (optimizer={optimizer}, amp={amp}"
        + (", с округлением аллокатора" if allocator_aware else "")
        + "):",
    ]
    for key in ("weights", "gradients", "optimizer", "autocast_cache"):
        if key in fp:
            lines.append(f"    {key:<16} {gib(fp[key]):>8.3f} GiB")
    lines.append(f"    {'ИТОГО':<16} {gib(fp['total']):>8.3f} GiB")

    if stats.by_module:
        lines += ["", "  крупнейшие блоки:"]
        for name, size in stats.by_module.items():
            lines.append(f"    {name:<48} {gib(size):>8.3f} GiB")

    return "\n".join(lines)


@dataclass
class DeviceMemory:
    """Срез памяти одного GPU.

    total/free - всего памяти GPU / сколько свободно (приходят от драйвера и учитывают все процессы разом).
    Остальное — только текущий процесс PyTorch:

        tensors         живые тензоры (memory_allocated)
        reserved        сколько PyTorch забрал у драйвера и держит за собой
        peak_tensors    пиковый tensors
        peak_reserved   пиковый reserved
        cache           reserved - tensors: освобождённые блоки, которые аллокатор
                        оставил себе. Драйверу они не возвращаются, поэтому nvidia-smi
                        продолжает считать их занятыми. Снимается empty_cache().
        used            всего занято всеми процессами (total - free)
        outside         used - reserved: всё, что PyTorch не контролирует (занятое чужими процессами, контекстом CUDA, faiss-gpu)
    """

    index: int
    name: str
    total: int
    free: int
    tensors: int
    reserved: int
    peak_tensors: int
    peak_reserved: int

    @property
    def used(self) -> int:
        return self.total - self.free

    @property
    def cache(self) -> int:
        return self.reserved - self.tensors

    @property
    def outside(self) -> int:
        # На чужие процессы reserved не распространяется, так что разница может
        # включать и их — но чаще это контекст CUDA (~0.3-0.5 GiB).
        return max(0, self.used - self.reserved)

    @property
    def usage_percent(self) -> float:
        return self.used / self.total * 100


def device_memory(index: int) -> DeviceMemory:
    free, total = torch.cuda.mem_get_info(index)
    return DeviceMemory(
        index=index,
        name=torch.cuda.get_device_name(index),
        total=total,
        free=free,
        tensors=torch.cuda.memory_allocated(index),
        reserved=torch.cuda.memory_reserved(index),
        peak_tensors=torch.cuda.max_memory_allocated(index),
        peak_reserved=torch.cuda.max_memory_reserved(index),
    )


def collect_gpu_memory() -> list[DeviceMemory]:
    if not torch.cuda.is_available():
        return []
    return [device_memory(i) for i in range(torch.cuda.device_count())]


def cuda_report(devices: list[DeviceMemory] = None) -> str:
    """Текстовая сводка по всем GPU."""
    if devices is None:
        devices = collect_gpu_memory()
    if not devices:
        return "CUDA недоступна, GPU не обнаружены."

    header = f"{'GPU':<7}{'тензоры':>10}{'кеш':>10}{'вне torch':>11}{'свободно':>11}{'всего':>9}{'пик':>10}"
    lines = [header, "-" * len(header)]
    for d in devices:
        lines.append(
            f"cuda:{d.index:<2}"
            f"{gib(d.tensors):>9.2f}G"
            f"{gib(d.cache):>9.2f}G"
            f"{gib(d.outside):>10.2f}G"
            f"{gib(d.free):>10.2f}G"
            f"{gib(d.total):>8.2f}G"
            f"{gib(d.peak_tensors):>9.2f}G"
        )
    return "\n".join(lines)


_GPU_TABLE_CSS = """
<style>
    .gpu-table { border-collapse: collapse; width: 100%; margin-bottom: 20px;
                 color: #212121; background-color: #ffffff; }
    .gpu-table th, .gpu-table td { border: 1px solid #ccc; padding: 8px; text-align: left; }
    .gpu-table th { background-color: #f2f2f2; color: #212121; }
    .gpu-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .gpu-table .sub { color: #757575; font-size: 0.85em; font-weight: normal; }
    .low-mem { background-color: #ffebee; color: #c62828; font-weight: bold; }
    .med-mem { background-color: #fff8e1; color: #ff8f00; }
    .high-mem { background-color: #e8f5e8; color: #2e7d32; }
</style>
"""


def gpu_memory_html(
    devices: list[DeviceMemory] = None,
    low_free_gb: float = 5.0,
    med_free_gb: float = 10.0,
    cache_warn_gb: float = 1.0,
) -> str:
    """
    Собирает HTML-таблицу. 
    low_free_gb, med_free_gb: пороги подсветки в GiB для free.
    """
    rows = []
    if devices is None:
        devices = collect_gpu_memory()
    for d in devices:
        free_gb = gib(d.free)
        if free_gb < low_free_gb:
            mem_class = "low-mem"
        elif free_gb < med_free_gb:
            mem_class = "med-mem"
        else:
            mem_class = "high-mem"

        # Крупный кеш — единственная строка, с которой можно что-то сделать
        # прямо сейчас (empty_cache()), поэтому подсвечиваем её отдельно.
        cache_class = "med-mem" if gib(d.cache) >= cache_warn_gb else ""

        rows.append(f"""
        <tr>
            <td><strong>cuda:{d.index}</strong></td>
            <td>{d.name}</td>
            <td class="num">{gib(d.tensors):.2f} GB</td>
            <td class="num {cache_class}">{gib(d.cache):.2f} GB</td>
            <td class="num">{gib(d.outside):.2f} GB</td>
            <td class="num {mem_class}">{free_gb:.2f} GB</td>
            <td class="num">{gib(d.total):.2f} GB</td>
            <td class="num">{d.usage_percent:.1f}%</td>
            <td class="num">{gib(d.peak_tensors):.2f} / {gib(d.peak_reserved):.2f} GB</td>
        </tr>""")

    return _GPU_TABLE_CSS + f"""
    <table class="gpu-table">
        <tr>
            <th>GPU</th>
            <th>Название</th>
            <th>Тензоры<div class="sub">allocated</div></th>
            <th>Кеш<div class="sub">reserved − allocated</div></th>
            <th>Вне torch<div class="sub">контекст, другие процессы, faiss, др.</div></th>
            <th>Свободно</th>
            <th>Всего</th>
            <th>Загрузка</th>
            <th>Пик<div class="sub">allocated / reserved</div></th>
        </tr>{"".join(rows)}
    </table>
    """


def print_gpu_memory(
    low_free_gb: float = 5.0,
    med_free_gb: float = 10.0,
    cache_warn_gb: float = 1.0,
    reset_peak: bool = False,
) -> None:
    """
    Таблица по всем GPU: HTML в ноутбуке (gpu_memory_html()), текст в обычном скрипте (cuda_report()).

    reset_peak: нужно ли обнулять счётчики пика после вывода. Пик держится
        с момента запуска процесса или предыдущего сброса, так что для замера
        отдельного этапа его надо сбрасывать явно.
    """
    devices = collect_gpu_memory()
    if not devices:
        print("CUDA is not available. No GPUs were detected.")
        return

    print(f"detected {len(devices)} GPU:\n")

    try:
        from IPython.display import HTML, display
        display(HTML(gpu_memory_html(devices, low_free_gb, med_free_gb, cache_warn_gb)))
    except ImportError:
        print(cuda_report(devices))

    if reset_peak:
        for d in devices:
            torch.cuda.reset_peak_memory_stats(d.index)


if __name__ == "__main__":
    import sys

    model_id = sys.argv[1] if len(sys.argv) > 1 else "deepvk/USER-bge-m3"
    stats = stats_from_config(model_id)
    print(format_report(stats, optimizer="adamw", amp=None, allocator_aware=True))
    print()
    print(format_report(stats, optimizer="adamw", amp="fp16", allocator_aware=True))
    print()
    print(cuda_report())

