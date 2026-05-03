import copy
from typing import List, Dict
from transformers import AutoTokenizer

from torch.utils.data import Dataset


def content_to_type_and_text(messages):
    messages = copy.deepcopy(messages)
    for m in messages:
        m["content"] = [{"type": "text", "text": m["content"]}]
    return messages


class Dataset(Dataset):

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        original_records: List[Dict],
        sample_rate: float = 1.0,
        convert_content = False
    ):
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.convert_content = convert_content
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]
