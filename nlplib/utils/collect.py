import torch, gc

def collect():
    torch.cuda.ipc_collect()
    gc.collect()
    torch.cuda.empty_cache()