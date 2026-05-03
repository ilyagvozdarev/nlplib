import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel


def resave_model(model_name: str, output_path: str, device_map: str = "auto"):

    '''
        скачивает LM модель и токенизатор и сохраняет по заданному пути
    '''

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    base_model.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(resave)
