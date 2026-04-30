import os, argparse, json, gc, sys
from itertools import chain, accumulate


# cwd = '/raid_igvozdarev/ASP/VD'
# cwd = '.'
# os.chdir(cwd)
# module_path = r'C:/Users/el1ja/Desktop/repo/modules'
# sys.path.extend([module_path, os.path.dirname(os.path.abspath(__file__))])

repo = r'/raid_igvozdarev/repo'
sys.path.extend([repo])



from nlp_utils.set_seed import set_seed
from nlp_utils.utils.io import *

import torch
torch.set_float32_matmul_precision('high')

from nlp.src.infer.llm import LLM


def collect():
    torch.cuda.ipc_collect()
    gc.collect()
    torch.cuda.empty_cache()


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
    # convs_ = convs_[:5]
    input_field = list(convs[-1]['inputs'].keys())[-1]
    counts = [len(conv['inputs'][input_field]) for conv in convs]
    
    # print(f'\n\n{convs[:2]}\n\n')
    return convs_, counts



def read_config(file):
    ext = os.path.splitext(file)[-1].lower()
    exts = ['.json', '.yaml', '.yml']
    assert ext in exts, f'config ext not in {exts}'
    return (read_json if ext == '.json' else read_yaml)(file)




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
    results = [{'prompts': prompts[i:i+count], 'compls': compls[i:i+count]} for i, count in zip(starts, counts)]

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