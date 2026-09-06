"""
Собирает SPLADE-совместимый бэкбон из deepvk/USER-bge-m3.

Проблема: USER-bge-m3 — это XLMRobertaModel (dense-энкодер), у него НЕТ MLM-головы,
а SPLADE работает именно на логитах MLM-головы. AutoModelForMaskedLM создаст её
со случайными весами -> на выходе будет шум.

Решение: USER-bge-m3 — это XLM-RoBERTa-large с урезанным до 46166 токенов словарём,
и этот словарь — строгое подмножество словаря xlm-roberta-large (проверено: 0 лишних
токенов). Поэтому MLM-голову можно перенести почти без потерь:
  * lm_head.dense / lm_head.layer_norm  — не зависят от словаря, копируются как есть;
  * lm_head.decoder.weight              — привязан (tied) к word_embeddings, т.е. уже
                                          содержит нужные строки урезанного словаря;
  * lm_head.bias                        — переносится по строкам токенов (id -> id).

Запуск:  python build_splade_backbone.py
Результат: локальная папка ./models/USER-bge-m3-mlm, которую можно грузить в SparseEncoder.
"""

from __future__ import annotations

import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

SRC = "deepvk/USER-bge-m3"
HEAD_SRC = "FacebookAI/xlm-roberta-large"
OUT = Path("models/USER-bge-m3-mlm")

HEAD_KEYS = [
    "lm_head.dense.weight",
    "lm_head.dense.bias",
    "lm_head.layer_norm.weight",
    "lm_head.layer_norm.bias",
    "lm_head.bias",
]
_NP_DTYPE = {"F32": np.float32, "F16": np.float16}


def fetch_lm_head(repo: str) -> dict[str, torch.Tensor]:
    """Тянет только тензоры головы (~5 МБ) через HTTP Range вместо 2.2 ГБ чекпоинта."""
    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"

    def rng(a: int, b: int) -> bytes:
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
        return urllib.request.urlopen(req, timeout=120).read()

    header_len = struct.unpack("<Q", rng(0, 7))[0]
    header = json.loads(rng(8, 8 + header_len - 1))
    data_start = 8 + header_len

    tensors = {}
    for key in HEAD_KEYS:
        meta = header[key]
        start, end = meta["data_offsets"]
        buf = rng(data_start + start, data_start + end - 1)
        arr = np.frombuffer(buf, dtype=_NP_DTYPE[meta["dtype"]]).reshape(meta["shape"])
        tensors[key] = torch.from_numpy(arr.copy()).float()
    return tensors


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(SRC)
    # Предупреждение "newly initialized: ['lm_head...']" здесь ожидаемо — головy чиним ниже.
    model = AutoModelForMaskedLM.from_pretrained(SRC)

    head = fetch_lm_head(HEAD_SRC)
    lm_head = model.lm_head
    with torch.no_grad():
        lm_head.dense.weight.copy_(head["lm_head.dense.weight"])
        lm_head.dense.bias.copy_(head["lm_head.dense.bias"])
        lm_head.layer_norm.weight.copy_(head["lm_head.layer_norm.weight"])
        lm_head.layer_norm.bias.copy_(head["lm_head.layer_norm.bias"])

        # bias: сопоставляем токены урезанного словаря с их id в полном словаре XLM-R
        full_vocab = AutoTokenizer.from_pretrained(HEAD_SRC).get_vocab()
        vocab = tokenizer.get_vocab()
        missing = [t for t in vocab if t not in full_vocab]
        if missing:
            raise RuntimeError(f"{len(missing)} токенов нет в словаре {HEAD_SRC}, перенос bias невозможен")
        index = torch.tensor([full_vocab[t] for t in sorted(vocab, key=vocab.__getitem__)])
        lm_head.bias.copy_(head["lm_head.bias"][index])
        lm_head.decoder.bias = lm_head.bias  # roberta связывает эти два тензора

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    print(f"saved -> {OUT.resolve()}  (vocab={model.config.vocab_size})")


if __name__ == "__main__":
    main()
