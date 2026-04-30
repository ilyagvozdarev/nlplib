import os, shutil, codecs, json
from pathlib import Path


import torch 
from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftConfig, PeftModel

from safetensors import safe_open
from safetensors.torch import save_file

from huggingface_hub import snapshot_download




def check_if_adapt(model_dir):
    '''
	    если локальный путь существует и хранит 'adapter_config.json' и 'adapter_model.(bin|safetensors)' или
        если такого лок. пути нет, но можно скачать методом PeftConfig.from_pretrained
    '''
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

    '''
        загружает адаптер и ее базовую модель и объединяет их в автономную модель

        peft_model_path - адаптер
        base_model_path - базовая модель
        делим alpha на alpha_scale для модулей у которых alpha отличается от alpha по умолчанию (alpha_pattern из конфига адаптера)
        (кроме выходных эмбеддингов если задано not_scale_lm_head)

        создаем peft-модель из базовой или если не задано то из базовой модели адаптера (torch_dtype равен dtype или если не задан то такой же как у базовой модели адаптера)

        обработка эмбеддингов (связывание + масштабирование на альфа):
            если выходной эмбеддинг (lm_head) в полностью обучаемых модулях адаптера (modules_to_save конфига адаптера), 
            то: 
              - масштабируем его последнюю адаптацию - делаем равным выходному эмбеддингу базовой модели + деленная на alpha_scale разница между обученным выходным 
                эмбеддингом адаптера и выходным эмбеддингом базовой модели (посчитали до объединения merge_and_unload), то есть alpha_scale вероятно управляет силой 
                дообучения выходного эмбеддинга из последней адаптации.
              - если веса входных и выходных эмбеддингов базовой модели связаны (tie_word_embeddings в конфиге базовой модели) то связываем их и в автономной модели 
                (полученной после объединения merge_and_unload), при этом проверяется что входной эмбеддинг (embed_tokens) не содержится в modules_to_save, тк это бы
                означало что входной и выходной эмбеддинги одновременно были в полностью обучаемых модулях адаптера (modules_to_save конфига адаптера), следовательно
                они были развязанны при адаптации, но в базовой модели были связаны -> можно получить некорректную модель
    '''

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
    
    # embed_modules_to_save = {'lm_head', 'embed_tokens'} & set(config.modules_to_save)
    # cond = model.config.tie_word_embeddings and embed_modules_to_save
    if 'lm_head' in config.modules_to_save:
        with torch.no_grad():
            lm_head = model.base_model.model.lm_head
            delta = lm_head.modules_to_save['default'].weight - lm_head.original_module.weight
            delta /= alpha_scale
            new_embeds = lm_head.original_module.weight + delta

    model = model.merge_and_unload()
    model.train(False)

    print(model.model.embed_tokens.weight[0])
    print(model.lm_head.weight[0])
    print(model.config.tie_word_embeddings)
    print(config.modules_to_save)

    if 'lm_head' in config.modules_to_save:
        with torch.no_grad():
            model.lm_head.weight.copy_(new_embeds)
            if base_model.config.tie_word_embeddings:
                msg_assert = "'lm_head' и 'embed_tokens' оба были в 'modules_to_save' адаптера то есть не были связаны при адаптации, но в базовой" \
                             "были связаны (tie_word_embeddings == True) -> при merge получим некорректную модель"
                assert 'embed_tokens' not in config.modules_to_save, msg_assert
                model.model.embed_tokens.weight = model.lm_head.weight

    print(model.model.embed_tokens.weight[0])
    print(model.lm_head.weight[0])

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

    '''
	    весь репозиторий адаптера загружаем и сохраняем в папки: 
		{out_dir}/full 	       весь как есть 
		{out_dir}/embeds       так же как full но:
                               в adapter_model.safetensors оставляем только тензоры в названии которых есть 'lm_head' или 'embed_tokens'
                               в adapter_config.json в 'modules_to_save' и 'target_modules' оставляем только 'lm_head' или 'embed_tokens'          
		{out_dir}/adapters     аналогично 'embeds' но наоборот все кроме 'lm_head' и 'embed_tokens'
    '''

    if not os.path.exists(model_path):
        snapshot_download(model_path, local_dir=out_dir/'full')
        model_path = out_dir/'full'
    
    # копирование дерева каталогов репозитория в embeds и adapters

    for division in ('embeds', 'adapters'):
        if os.path.exists(out_dir/division):
            shutil.rmtree(out_dir/division)
        shutil.copytree(model_path, out_dir/division)

    load_prune_save_adapt(out_dir/'embeds', lambda key: 'lm_head' in key or 'embed_tokens' in key)
    load_prune_save_adapt(out_dir/'adapters', lambda key: 'lm_head' not in key and 'embed_tokens' not in key)

    return out_dir/'embeds', out_dir/'adapters'