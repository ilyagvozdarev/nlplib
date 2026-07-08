from typing import List

from .llm_base import LLM_base


class LLM_unsloth(LLM_base):
    def load_model(self):
        self.llm, self.tokenizer = self.flm.from_pretrained(
            model_name=self.model_name,
            load_in_4bit=True,
            device_map="cuda"
        )
        self.logger.info(f"llm model {self.model_name} loaded on device: {self.llm.device}")
        self.llm = self.flm.for_inference(self.llm)
        self.sampling_params = self.model_config['generation_config']

    def generate(self, prompts: List[int]):
        return super().generate(prompts)








