import os
# установка scheduling deterministic для воспроизводимости
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams
from .llm_base import LLM_base


class LLM_vllm(LLM_base):

    def load_model(self):
        self.logger.info('\nmodel_config = \n' + repr(self.model_config['model']))  
        self.llm = LLM(
            model=self.model_name,
            **self.model_config['model']
        )
        self.SP = SamplingParams(**self.model_config['generation_config'])
        self.tokenizer = self.llm.get_tokenizer()


    def generate(self, prompts):
        outputs = self.llm.generate(prompts, self.SP)
        # completions = [output.outputs[0].text for output in outputs]
        # completions = [[n_output.text.encode("utf-8").decode("utf-8", "ignore") for n_output in output.outputs] for output in outputs]
        completions = [[n_output.text for n_output in output.outputs] for output in outputs]
        return completions
