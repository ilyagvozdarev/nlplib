from tqdm import tqdm
import importlib

from .utils import conversations_to_inputs_prompts, gen_batch
from .logging import setup_logging


class LLM:

    def __init__(
            self, 
            model_name, 
            model_config, 
            tokenizer_params,
            engine='vllm'
        ):
        self.logger = setup_logging(self.__class__.__name__)
        self.engine = engine
        self.tokenizer_params = tokenizer_params
        self.load_model(model_name, model_config)


    def load_model(self, model_name, model_config):
        llm_module = importlib.import_module(f'.llm_{self.engine}', package=__package__)
        llm_engine = getattr(llm_module, f'LLM_{self.engine}')
        self.llm = llm_engine(model_name, model_config)
        self.logger.info(f"model_name = {model_name}")


    def generate(self, conversations, batch_size=9999, tokenizer_params={}):

        self.logger.info(f"LLM.generate")
        self.logger.info(f"batch_size = {batch_size}")

        if self.engine in ['unsloth', 'transformers'] and batch_size > 1:
            assert self.llm.tokenizer.padding_side == "left", "Batched inference for right padding side is impossible"

        inputs, prompts = conversations_to_inputs_prompts(
            self.llm.tokenizer, 
            conversations, 
            {**self.tokenizer_params, **(tokenizer_params or {})},
            device='cuda'
        )

        self.logger.info(f"generating ...")
        
        completions = []

        for batch_inputs_prompts in tqdm(gen_batch(list(zip(inputs, prompts)), batch_size)):
            batch_prompts_ids, batch_prompts = list(zip(*batch_inputs_prompts))
            if self.engine == 'unsloth':
                batch_prompts = batch_prompts_ids
            batch_completions = self.llm.generate(batch_prompts)
            completions.extend(batch_completions)
        
        return prompts, completions



        
