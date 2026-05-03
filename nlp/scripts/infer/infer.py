import os, argparse, json, gc, sys


# cwd = '/raid_igvozdarev/ASP/VD'
# cwd = '.'
# os.chdir(cwd)
module_path = r'/raid_igvozdarev/scripts'
# module_path = r'C:/Users/el1ja/Desktop/repo/modules'
# sys.path.extend([module_path, os.path.dirname(os.path.abspath(__file__))])
sys.path.extend([module_path])


from nlp_utils.set_seed import set_random_seed

import torch
torch.set_float32_matmul_precision('high')

from llm import LLM


def collect():
    torch.cuda.ipc_collect()
    gc.collect()
    torch.cuda.empty_cache()


def read_dataset_and_generate(llm, conv_dataset_file, tokenizer_params, batch_size):
    with open(conv_dataset_file, encoding="utf-8") as d:
        conversations = [json.loads(line) for line in d]
    if not conversations:
        print('\nconv_dataset_file is empty!\n')
        return []
    print('\nconversations loaded!\n')
    conversations_sets = conversations
    if not 'instr' in conversations[0]:
        conversations_sets = [{'instr': None, 'conversations': conversations}]

    for conversations_set in conversations_sets:
        print(f'\ninstr = {conversations_set["instr"]}\n')
        conversations_set['prompts'], conversations_set['completions'] = llm.generate(
            conversations_set['conversations'], 
            batch_size,
            tokenizer_params
        )
    
    return conversations_sets


def save_conversations(conversations_sets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "output.jsonl")
    with open(output_file, "w", encoding="utf-8") as out:
        for conv_set in conversations_sets:
            out.write(json.dumps(conv_set, ensure_ascii=False) + '\n')  


def main(args):

    set_random_seed(args.seed)

    with (
        open(args.model_config, encoding="utf-8") as c,
        open(args.tokenizer_config, encoding="utf-8") as t_c
    ):
        model_config = json.load(c)
        tokenizer_params = json.load(t_c)['tokenizer_config']

    llm = LLM(args.model_name, model_config, tokenizer_params, args.engine)

    conversations_sets = read_dataset_and_generate(llm, args.conv_dataset, tokenizer_params, args.batch_size)

    del llm
    collect()

    if args.only_completions:
        for conversations_set in conversations_sets:
            del conversations_set['prompts'], conversations_set['conversations']

    save_conversations(conversations_sets, args.output_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--engine", type=str, default='vllm', help="vllm / unsloth")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--tokenizer_config", type=str)
    parser.add_argument("--conv_dataset", type=str, help="conversations dataset (json)")
    parser.add_argument("--batch_size", type=int, default=9999)
    parser.add_argument("--output_dir", type=str, default='output_dir')
    parser.add_argument("--only_completions", action="store_true")
    parser.add_argument("--seed", type=int, default='42')
    args = parser.parse_args()

    main(args)