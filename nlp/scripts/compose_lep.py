import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig, LogitsProcessorList
import argparse
import time
import yaml
import codecs
from pathlib import Path
from huggingface_hub import snapshot_download
from ..src.utils.load_model import load_model
from .src.lep_proj_utils import list_projection_modes
import shutil
import os
from safetensors import safe_open
from safetensors.torch import save_file
import gc



def load_donor_model(model_path, out_dir, alpha_scale=1.0, not_scale_lm_head=False, device="cuda:0"):
    # отличия от load_model_base_or_adapt:
    #   адаптер применяется только для эмбеддингов (embed_path) тк для проекции используются только они у donor 
    adapters_path = None
    if check_if_adapt(model_path):
        embed_path, adapters_path = save_split_adapter(model_path, out_dir)
        model = load_adapt_and_merge(embed_path, None, alpha_scale, not_scale_lm_head, device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        print("Model loaded")
        time.sleep(2)
    else:
        model, tokenizer = load_model(model_path, device=device)
        
    return model, tokenizer, adapters_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./composed_models")
    parser.add_argument("--custom_chat_template_path", type=str, default=None)      # chat_template который устанавливается токенизатору в конце перед сохранением
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    
    config_file = Path(args.config_file)
    print(f"Output will placed at: {out_dir}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("config",
        *[f"{k.upper()}: {v}" for k, v in config.items()],
        sep='\n'
    )
    
    target_model, target_model_tokenizer = load_model(config["target_model_path"], device="cuda:0")
    source_model, source_model_tokenizer = load_model(config["source_model_path"], device="cuda:1")
    donor_model, donor_model_tokenizer, adapters_path = load_donor_model(
        config["donor_model_path"], 
        out_dir, 
        config.get('alpha_scale', 1.0), 
        config.get('not_scale_lm_head', False),
        device='cuda:2'
    )
    
    proj_modes = {
        "lm_head": config["mode"],
        "model.embed_tokens": config["mode"]
    }

    model, tokenizer = make_lep(
        target_model, source_model, donor_model,
        target_model_tokenizer, donor_model_tokenizer, 
        module_projection_modes=proj_modes,
        coocurrence_map_path=None,
        overlap_penalty=1.0
    )
    
    
    model.save_pretrained(out_dir)
    if args.custom_chat_template_path is not None:
        with codecs.open(args.custom_chat_template_path, 'r', 'utf-8') as file:
            tokenizer.chat_template = json.load(file)

    tokenizer.save_pretrained(out_dir)

    
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if adapters_path is not None:
        model = load_adapt_and_merge(adapters_path, out_dir, config.get('alpha_scale', 1.0), config.get('not_scale_lm_head', False), device='cuda:3')
        model.save_pretrained(out_dir)
