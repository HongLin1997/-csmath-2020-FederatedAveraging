# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:34:15 2020

@author: admin
"""

import torch, random, os
import numpy as np

def adjust_learning_rate(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
    return lr


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
        
    torch.backends.cudnn.deterministic = True
    # if False, sacrifice the computation efficiency
    torch.backends.cudnn.benchmark = False
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
           