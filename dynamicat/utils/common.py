import torch
import random
import numpy as np
from transformers import set_seed
from loguru import logger



def print_rank_0(msg, rank=0):
    if rank <= 0:
        logger.info(msg)

def batch_dict_to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except Exception as e:
            logger.error(str(e))
            output[k] = v
    return output

def all_reduce_sum_of_tensor(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor

def set_random_seeds(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)