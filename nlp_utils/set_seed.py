# https://www.kaggle.com/code/datafan07/single-model-rewardtrainer-lora-llm/notebook?scriptVersionId=139527149

import numpy as np
import random
import os
import torch



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


