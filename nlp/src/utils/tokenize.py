

def fix_tokenizator(tokenizer, config):
    # fix spec tokens
    tokenize_params = config.get('tokenize', None)
    if tokenize_params:
        for st in ["bos_token", "eos_token", "pad_token"]:
            if spec_token := tokenize_params.get(st, None):
                setattr(tokenizer, st, spec_token)
                setattr(tokenizer, f"{st}_id", tokenizer.convert_tokens_to_ids([spec_token])[0])
    
    # fix chat_template
    if custom_chat_template := config.get('chat_template', None):
        tokenizer.chat_template = custom_chat_template