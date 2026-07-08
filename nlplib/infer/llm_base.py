from .logging import setup_logging
from typing import List


class LLM_base:
    def __init__(self, model_name, model_config):
        self.logger = setup_logging(self.__class__.__name__)
        self.model_name = model_name
        self.model_config = model_config
        self.load_model()

    def load_model(self):
        raise NotImplementedError("Subclasses must implement load_model method.")

    def generate(self, prompts: List[int]):
        prompts_ids = prompts

        # if prompt texts are passed, then we convert them into indexes
        if isinstance(prompts[0], str):
            prompts_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            assert prompts_ids["input_ids"][0][0] != prompts_ids["input_ids"][0][1]
            for prompt_ids in prompts_ids:
                prompt_ids.pop("token_type_ids", None)
            prompts_ids = prompts_ids.to(self.llm.device)

            # for older versions of transformers:
            # prompts_ids = tokenizer(prompts, return_tensors="pt", add_special_tokens=False)
            # prompts_ids = [{k: v.to(device) for k, v in input.items()} for prompt_ids in prompts_ids]

        outputs_ids = self.llm.generate(
            **prompts_ids,
            generation_config=self.sampling_params,
        )
        completions_ids = [
            output_ids[len(prompt_ids[0]):]
            for prompt_ids, output_ids 
            in zip(prompts_ids, outputs_ids)
        ]
        completions = self.tokenizer.decode(completions_ids, skip_special_tokens=True)
        return completions




        
