from .llm_base import LLM_base

from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch


class LLM_transformers(LLM_base):
    def load_model(self):
        self.logger.info(f"llm model {self.model_name} loaded on device: {self.llm.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name=self.model_name,
            load_in_8bit=True,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.llm.eval()
        self.sampling_params = GenerationConfig.from_pretrained(self.model_name)

    def generate(self, prompts: List[int]):
        return super().generate(prompts)