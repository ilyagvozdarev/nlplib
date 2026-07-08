import os
import json, argparse, random

from unsloth import FastLanguageModel, UnslothTrainingArguments
from transformers import (
    DefaultDataCollator,
    BitsAndBytesConfig
)

import torch

from src.dataset.ConvDataset import ConvDataset
from nlp_utils.utils.io import read_jsonl
from nlp_utils.set_seed import set_seed
from src.utils.fix_embeddings import fix_untrained_tokens
from src.utils.tokenize import fix_tokenizator
from src.Trainers.Trainers import CustomTrainer

os.environ["UNSLOTH_RETURN_LOGITS"] = "0"           # чтобы Unsloth возвращал логиты
os.environ["TOKENIZERS_PARALLELISM"] = "false"      # чтобы предотвратить предупреждения в токенизаторах Huggingface
torch._dynamo.config.cache_size_limit = 128


def train(
    config_file: str,
    train_file: str,
    val_file: str,
    output_dir: str,
    seed: int = 42
) -> None:
    
    set_seed(seed)

    with open(config_file) as r:
        config = json.load(r)

    max_tokens_count = config["max_tokens_count"]
    max_seq_length = config.get("max_seq_length", max_tokens_count)
    
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

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        attn_implementation="flash_attention_2",
    )
    
    # fix tokenizator (spec tokens, chat_tempalte)
    fix_tokenizator(tokenizer, config)

    tokenizer.padding_side = "left"


    train_conv = read_jsonl(train_file)
    val_conv = read_jsonl(val_file)
    random.shuffle(train_conv)

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

    data_collator = DefaultDataCollator()

    # info
    print(f'1-st sample: {train_conv[0]}')
    for input_field, info in zip(
        ["input_ids", "attention_mask", "labels"], 
        ["INPUT_IDS (COLLATOR)", "MASK (COLLATOR)", "LABELS (COLLATOR)"]
    ):
        print(info)
        print(data_collator([train_dataset[0], train_dataset[1]])[input_field][0])


    if config.get("save_tokenizer", False):
        tokenizer.save_pretrained(output_dir)

    if config.get("fix_untrained_tokens", False):
        fix_untrained_tokens(model, tokenizer)

    adapt_config = config.get("adaptation", {})
    if adapt_config.pop('method') == 'lora':
        need_tie = adapt_config.pop('tie_word_embeddings_if_been_tied', True)
        model = FastLanguageModel.get_peft_model(model, **adapt_config)
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


    # if trainer_config.get("report_to", "wandb") == "wandb":
    #     wandb.init(project="rulm_self_instruct", name=config_file)

    trainer_config = config["trainer"]
    training_args = UnslothTrainingArguments(**trainer_config, output_dir=output_dir) 
    trainer = CustomTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        args=training_args,
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
