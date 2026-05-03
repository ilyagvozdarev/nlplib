import copy
import random
from typing import List, Dict

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def content_to_type_and_text(messages):
    messages = copy.deepcopy(messages)
    for m in messages:
        m["content"] = [{"type": "text", "text": m["content"]}]
    return messages


class DPODataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        sample_rate: float = 1.0,
        apply_chat_template: bool = False,
        convert_content = False
    ):
        super().__init__(original_records, sample_rate, convert_content, tokenizer, max_tokens_count)

        self.records = []

        for record in tqdm(original_records):

            if random.random() > self.sample_rate:
                continue

            prompt_messages = record["prompt"]
            chosen_messages = record["chosen"]
            rejected_messages = record["rejected"]

            if convert_content:
                prompt_messages = content_to_type_and_text(prompt_messages)
                chosen_messages = content_to_type_and_text(chosen_messages)
                rejected_messages = content_to_type_and_text(rejected_messages)


            chosen_tokens = self.get_tokens(self, prompt_messages + chosen_messages)
            rejected_tokens = self.get_tokens(self, prompt_messages + rejected_messages)
            if len(chosen_tokens) > self.max_tokens_count - 5 or len(rejected_tokens) > self.max_tokens_count - 5:
                continue

            if not apply_chat_template:
                self.records.append(
                    {
                        "prompt": prompt_messages,
                        "chosen": chosen_messages,
                        "rejected": rejected_messages,
                    }
                )
            else:
                prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False)

                chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
                chosen = chosen.replace(tokenizer.bos_token, "")

                rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
                rejected = rejected.replace(tokenizer.bos_token, "")

                assert chosen.strip()
                assert rejected.strip()

                self.records.append(
                    {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                )
        

    def get_tokens(self, messages):
        tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
        )
        if isinstance(tokens[0], list):
            tokens = tokens[0]
            print('tokens list after tokenization is 2-dim')
        return tokens