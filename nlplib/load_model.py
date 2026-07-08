import os, shutil, codecs, json, time
from pathlib import Path

import torch 
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download


def check_if_adapt(model_dir):
    """
    True if a local path exists and contains 'adapter_config.json' and 
    'adapter_model.(bin|safetensors)', or if no such local path exists, but 
    it can be downloaded via PeftConfig.from_pretrained
    """
    if_adapter = False
    is_exist = lambda file: os.path.exists(os.path.join(model_dir, file))
    if os.path.exists(model_dir):
        adapter_config_exists = is_exist('adapter_config.json')
        adapter_model_exists = is_exist('adapter_model.bin') or is_exist('adapter_model.safetensors')
        if_adapter = adapter_config_exists and adapter_model_exists
        return if_adapter
    try:
        PeftConfig.from_pretrained(model_dir)
        if_adapter  = True
    except:
        pass
    return if_adapter


def load_adapt_and_merge(
    peft_model_path, 
    base_model_path=None, 
    alpha_scale=1.0, 
    not_scale_lm_head=False, 
    device_map: str = "auto", 
    dtype=None
):
    """
    Loads an adapter and its base model and merges them into a standalone model.
    Creates a peft model from the base model, or if not given, from the adapter's 
    base model (torch_dtype equals dtype, or if not set, the same as the adapter's 
    base model's dtype)

    Parameters
    ----------
    peft_model_path:
        the adapter
    base_model_path:
        the base model
    alpha_scale:
        divides alpha by alpha_scale for modules whose alpha differs from the default 
        alpha (alpha_pattern from the adapter config)
        (except for the output embeddings, if `not_scale_lm_head` is set)

    Notes
    -----
    embedding handling (tying + alpha scaling):
    if the output embedding (lm_head) is among the adapter's fully-trainable modules 
    (modules_to_save in the adapter config), then:
    - we rescale its final adaptation - we set it equal to the base model's output 
      embedding plus the difference between the adapter's trained output embedding 
      and the base model's output embedding (computed before merging via merge_and_unload), 
      divided by alpha_scale. That is, alpha_scale presumably controls the strength of the 
      fine-tuning applied to the output embedding from the last adaptation.
    - if the base model's input and output embedding weights are tied (tie_word_embeddings 
      in the base model config), we also tie them in the resulting standalone model 
      (obtained after merge_and_unload). This also checks that the input embedding 
      (embed_tokens) is not included in modules_to_save, since that would mean both the 
      input and output embeddings were simultaneously among the adapter's fully-trainable 
      modules (modules_to_save in the adapter config), meaning they were untied during 
      adaptation, while being tied in the base model -> this could result in an incorrect model
    """
    config = PeftConfig.from_pretrained(peft_model_path)
    lm_head_alpha = config.alpha_pattern.get("lm_head", config.lora_alpha)

    config.lora_alpha /= alpha_scale
    for name in config.alpha_pattern:
        config.alpha_pattern[name] /= alpha_scale

    if not_scale_lm_head:
        config.alpha_pattern["lm_head"] = lm_head_alpha

    base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
    torch_dtype = dtype or base_model_config.torch_dtype
    base_model_path = config.base_model_name_or_path if base_model_path is None else base_model_path

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="sdpa"
    )

    model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        torch_dtype=torch_dtype,
        config=config
    )
    
    if 'lm_head' in config.modules_to_save:
        with torch.no_grad():
            lm_head = model.base_model.model.lm_head
            delta = lm_head.modules_to_save['default'].weight - lm_head.original_module.weight
            delta /= alpha_scale
            new_embeds = lm_head.original_module.weight + delta

    model = model.merge_and_unload()
    model.train(False)

    if 'lm_head' in config.modules_to_save:
        with torch.no_grad():
            model.lm_head.weight.copy_(new_embeds)
            if base_model.config.tie_word_embeddings:
                msg_assert = "'lm_head' and 'embed_tokens' were both in the adapter's " \
                             "'modules_to_save', i.e. they were untied during adaptation, but " \
                             "were tied in the base model (tie_word_embeddings == True) -> " \
                             "merge would produce an incorrect model"
                assert 'embed_tokens' not in config.modules_to_save, msg_assert
                model.model.embed_tokens.weight = model.lm_head.weight

    model.train(False)
    model.eval()
    return model



def load_prune_save_adapt(model_path, key_filter):
    tensors = {}
    with safe_open(Path(model_path)/"adapter_model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            if key_filter(key):
                tensors[key] = f.get_tensor(key)
                
    with codecs.open(Path(model_path)/'adapter_config.json', 'r', 'utf-8') as config_file:
        config = json.load(config_file)
        
    if config['modules_to_save'] is not None:
        config['modules_to_save'] = [k for k in config['modules_to_save'] if key_filter(k)]

    config['target_modules'] = [k for k in config['target_modules'] if key_filter(k)]

    if len(config['target_modules']) == 0:
        config['target_modules'] = None
        
    if config['modules_to_save']  is None or len(config['modules_to_save']) == 0:
        config['modules_to_save'] = None
        
    assert config['modules_to_save'] is not None or config['target_modules']

    with codecs.open(Path(model_path)/'adapter_config.json', 'w', 'utf-8') as config_file:
        json.dump(config, config_file, indent=4)
        
    save_file(tensors, Path(model_path)/'adapter_model.safetensors')


def save_split_adapter(model_path, out_dir):
    """
    downloads the whole adapter repository and saves it into folders:
    {out_dir}/full         the entire repo as-is
    {out_dir}/embeds       same as full, but:
                           in adapter_model.safetensors we keep only the tensors 
                           whose name contains 'lm_head' or 'embed_tokens'
                           in adapter_config.json's 'modules_to_save' and 
                           'target_modules' we keep only 'lm_head' or 'embed_tokens'
    {out_dir}/adapters     same as 'embeds' but the other way around - everything 
                           except 'lm_head' and 'embed_tokens'
    """
    if not os.path.exists(model_path):
        snapshot_download(model_path, local_dir=out_dir/'full')
        model_path = out_dir/'full'
    
    # copying the repository directory tree to embeds and adapters
    for division in ('embeds', 'adapters'):
        if os.path.exists(out_dir/division):
            shutil.rmtree(out_dir/division)
        shutil.copytree(model_path, out_dir/division)

    load_prune_save_adapt(out_dir/'embeds', lambda key: 'lm_head' in key or 'embed_tokens' in key)
    load_prune_save_adapt(out_dir/'adapters', lambda key: 'lm_head' not in key and 'embed_tokens' not in key)

    return out_dir/'embeds', out_dir/'adapters'


def load_model(model_path, device="cpu"):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=config.torch_dtype,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.train(False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model loaded")
    time.sleep(2)
    return model, tokenizer


def load_model_base_or_adapt(model_path, out_dir=None, alpha_scale=1.0, not_scale_lm_head=False, device="cuda:0"):
    adapters_path = None
    if check_if_adapt(model_path):
        if out_dir:
            save_split_adapter(model_path, out_dir)
        model = load_adapt_and_merge(
            model_path, None, alpha_scale, not_scale_lm_head, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        print("Model loaded")
        time.sleep(2)
    else:
        model, tokenizer = load_model(model_path, device=device)
        
    return model, tokenizer, adapters_path


def merge_lora(model_name: str, output_path: str, device_map: str = "auto"):
    '''
    downloads the base model of a given peft model, combines them, and saves them to a given path
    '''
    config = PeftConfig.from_pretrained(model_name)
    base_model_path = config.base_model_name_or_path

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    lora_model = PeftModel.from_pretrained(
        base_model, model_name, torch_dtype=torch.bfloat16, device_map=device_map
    )

    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    lora_model.save_pretrained(output_path)