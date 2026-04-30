import os, argparse, json, sys

import uvicorn, signal
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from fastapi import FastAPI

# cwd = '/raid_igvozdarev/ASP/VD'
# cwd = '.'
# os.chdir(cwd)
module_path = r'/raid_igvozdarev/scripts'
# module_path = r'C:/Users/el1ja/Desktop/repo/modules'
sys.path.extend([module_path])


from llm import LLM
from infer import read_dataset_and_generate, save_conversations
from nlp_utils.set_seed import set_random_seed

from log_utils import setup_logging
logger = setup_logging(__name__)

import torch
torch.set_float32_matmul_precision('high')


class GenerateRequest(BaseModel):
    conv_dataset: str
    output_dir: str
    batch_size: int = 9999
    tokenizer_params: Dict[str, Any] = {}

class GenerateResponse(BaseModel):
    message: str


llm = None
app = FastAPI(title="Text Generation API")


# для сигнализации завершения
should_exit = False

def signal_handler(signum, frame):
    global should_exit
    should_exit = True
    print("Shutting down gracefully...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.get("/shutdown")
def shutdown():
    os.kill(os.getpid(), signal.SIGTERM)
    return {"message": "Shutting down"}


# @app.get("/shutdown")
# def shutdown():
#     os._exit(0)  # жесткий выход
#     return {"message": "Shutting down"}



@app.post("/llm_generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        conversations_sets = read_dataset_and_generate(
            llm, 
            request.conv_dataset, 
            request.tokenizer_params,
            request.batch_size
        )
        save_conversations(conversations_sets, request.output_dir)
    except Exception as e:
        logger.error(f"LLM generation error: {str(e)}", exc_info=True)
    return GenerateResponse(message="generation results saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default='vllm', help="vllm / unsloth")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--tokenizer_config", type=str)
    parser.add_argument("--seed", type=int, default='42')
    args = parser.parse_args()

    with (
        open(args.model_config, "r") as c,
        open(args.tokenizer_config, encoding="utf-8") as t_c
    ):
        model_config = json.load(c)
        tokenizer_params = json.load(t_c)['tokenizer_config']

    # model init
    llm = LLM(args.model_name, model_config, tokenizer_params, args.engine)


    set_random_seed(args.seed)

    # чтобы в случае его можно было завершить с помощью kill
    print(f"Server PID: {os.getpid()}")

    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info"
    )
