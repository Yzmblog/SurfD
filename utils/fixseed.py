import numpy as np
import torch
import random


def fixseed(seed):
    print(f'setting seed: {seed}')
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.random.manual_seed(seed)



