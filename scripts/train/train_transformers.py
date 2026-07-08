import random, json, os, sys, argparse

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification
)
from transformers import (
    Trainer,
    TrainingArguments,
    logging,
    BitsAndBytesConfig
)

from peft import get_peft_model, LoraConfig
from unsloth.models._utils import prepare_model_for_kbit_training

from src.dataset.ConvDataset import ConvDataset
from src.utils.tokenize import fix_tokenizator

nlp_utils_path = r'C:/Users/el1ja/Desktop/repo/modules'
sys.path.extend([nlp_utils_path])

from nlp_utils.utils.io import read_jsonl
from nlp_utils.set_seed import set_seed



def train(
    config_file: str,
    train_file: str,
    val_file: str,
    output_dir: str,
    seed: int = 42,
):
    set_seed(seed)
    logging.set_verbosity_info()

    with open(config_file, "r") as r:
        config = json.load(r)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print('LOCAL RANK: ', local_rank)

    adapt_config = config.get("adaptation")


    model_name = config["model_name"]
    tokenizer_name = config.get("tokenizer_name", model_name)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # fix tokenizator (spec tokens, chat_tempalte)
    fix_tokenizator(tokenizer, config)

    if config.get("save_tokenizer", False):
        tokenizer.save_pretrained(output_dir)

    train_conv = read_jsonl(train_file)
    val_conv = read_jsonl(val_file)
    random.shuffle(train_conv)

    datasets = []

    datasets = [
        ConvDataset(
            conversations,
            tokenizer,
            max_tokens_count=config["max_tokens_count"],
            sample_rate=config["data"]["sample_rate"],
            only_target_loss=config.get("only_target_loss", True),
            add_global_bos=config.get("add_global_bos", True),
            add_global_eos=config.get("add_global_eos", True)
        )
        for conversations in (train_conv, val_conv)
    ]
    train_dataset, val_dataset = datasets

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)


    # info
    print(f'1-st sample: {train_conv[0]}')
    for input_field, info in zip(
        ["input_ids", "attention_mask", "labels"], 
        ["INPUT_IDS (COLLATOR)", "MASK (COLLATOR)", "LABELS (COLLATOR)"]
    ):
        print(info)
        print(data_collator([train_dataset[0], train_dataset[1]])[input_field][0])


    quantization_config = config.get("quantization_config", {})
    load_in_8bit = bool(quantization_config.get("load_in_8bit", False))
    load_in_4bit = bool(quantization_config.get("load_in_4bit", False))
    bnb_config = None

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=f"cuda:{local_rank}",    # auto
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    # если включен gradient checkpointing то задаем не только 'gradient_checkpointing' в TrainingArguments
    # но и вызываем методы подготовки к gradient checkpointing
    gradient_checkpointing = config.get('gradient_checkpointing', False)
    if load_in_4bit or load_in_8bit:
        print('prepare_model_for_kbit_training with use_gradient_checkpointing')
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=gradient_checkpointing   # optional: gradient_checkpointing_kwargs={"use_reentrant": True}
        )  
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()       # optional: gradient_checkpointing_kwargs={"use_reentrant": True}
        model.enable_input_require_grads()


    adapt_config = config.get("adaptation", {})
    if adapt_config.pop('method') == 'lora':
        need_tie = adapt_config.pop('tie_word_embeddings_if_been_tied', True)
        lora_config = LoraConfig(**adapt_config)
        modules_to_save = adapt_config.get("modules_to_save", [])
        tie_word_embeddings = model.config.tie_word_embeddings
        if tie_word_embeddings and need_tie:
            msg = "в 'modules_to_save' нужно задавать либо 'lm_head', либо 'embed_tokens' (не одновременно) если в базовой модели модули были связаны, иначе они развяжутся"
            assert len({'lm_head', 'embed_tokens'} & set(modules_to_save)) == 1, msg
            if "gemma3" not in config["model_name"]:
                print("Tying lm_head and embed_tokens...")
                model.base_model.model.model.embed_tokens.modules_to_save["default"].weight = \
                    model.base_model.model.lm_head.modules_to_save["default"].weight
                # другой тип связывания: model.base_model.model.model.embed_tokens.weight = model.base_model.model.lm_head.modules_to_save["default"].weight
    elif unfreeze_modules := config.get("unfreeze_modules", None):
        for param_name, param in model.model.named_parameters():
            if not any([m for m in unfreeze_modules if m in param_name]):
                param.requires_grad = False


    trainer_config = config.get("trainer")
    training_args = TrainingArguments(
        output_dir=output_dir, report_to=trainer_config.get("report_to", "wandb"), **trainer_config
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--val_file', type=str)
    parser.add_argument('--output_dir', default="output_dir", type=str)

    args = parser.parse_args()

    train(**args)
