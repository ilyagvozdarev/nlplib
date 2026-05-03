import os

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from .llm_base import LLM_base


class LLM_tensorrtllm(LLM_base):

    def load_model(self):
        self.logger.info('\nmodel_config = \n' + repr(self.model_config['model']))

        args = {
            'model': self.model_name,
            **self.model_config['model'],
            **({'kv_cache_config': KvCacheConfig(**c)} if (c := self.model_config.get('kv_cache_config')) else {})
        }
        print('args = ', args)
        self.llm = LLM(**args)
        self.SP = SamplingParams(**self.model_config['generation_config'])
        self.tokenizer = self.llm.get_tokenizer()


    def generate(self, prompts):
        outputs = self.llm.generate(prompts, self.SP)
        # completions = [output.outputs[0].text for output in outputs]
        # completions = [[n_output.text.encode("utf-8").decode("utf-8", "ignore") for n_output in output.outputs] for output in outputs]
        completions = [[n_output.text for n_output in output.outputs] for output in outputs]
        return completions
