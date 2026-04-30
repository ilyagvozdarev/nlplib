import time

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .adapt import check_if_adapt, load_adapt, save_split_adapter


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
        model = load_adapt(model_path, None, alpha_scale, not_scale_lm_head, device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        print("Model loaded")
        time.sleep(2)
    else:
        model, tokenizer = load_model(model_path, device=device)
        
    return model, tokenizer, adapters_path