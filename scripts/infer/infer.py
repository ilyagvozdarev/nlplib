import os, argparse
from itertools import chain, accumulate

import torch
torch.set_float32_matmul_precision('high')

from nlplib.infer.llm import LLM
from nlplib.utils.collect import collect
from nlplib.utils.set_seed import set_seed
from nlplib.utils.io import read_config, read_jsonl, write_jsonl


def read_dataset(conv_dataset_file):
    convs = read_jsonl(conv_dataset_file)
    if not convs:
        print('\nconv_dataset_file is empty!\n')
        return []
    print('\nconversations loaded!\n')

    if len({'prompt', 'inputs'} & convs[0].keys()) != 2:
        return convs

    convs_ = [
        [{'role': 'user', 'content': conv['prompt'].format(**dict(zip(conv['inputs'].keys(), input)))}] 
        for conv in convs
        for input in zip(*conv['inputs'].values())
    ]
    input_field = list(convs[-1]['inputs'].keys())[-1]
    counts = [len(conv['inputs'][input_field]) for conv in convs]
    return convs_, counts


def main(args):
    set_seed(args.seed)

    model_config = read_config(args.model_config)
    chat_params = read_config(args.chat_config)['chat_config']

    llm = LLM(args.model_name, model_config, chat_params, args.engine)

    convs, counts = read_dataset(args.conv_dataset)

    prompts, compls = llm.generate(
        convs, 
        args.batch_size,
        chat_params
    )
    starts = chain([0], accumulate(counts))
    results = [
        {'prompts': prompts[i:i+count], 'compls': compls[i:i+count]} 
        for i, count in zip(starts, counts)
    ]

    del llm
    collect()

    if args.only_completions:
        del results['prompts']

    dir = os.path.dirname(args.out_file)
    if dir:
        os.makedirs(dir, exist_ok=True)
    write_jsonl(results, args.out_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default='vllm', help="vllm / unsloth")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--chat_config", type=str)
    parser.add_argument("--conv_dataset", type=str, help="conversations dataset (json)")
    parser.add_argument("--batch_size", type=int, default=9999)
    parser.add_argument("--out_file", type=str, default='output_dir/output.json')
    parser.add_argument("--only_completions", action="store_true")
    parser.add_argument("--seed", type=int, default='42')
    args = parser.parse_args()

    main(args)