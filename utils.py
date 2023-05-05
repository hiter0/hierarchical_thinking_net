import numpy as np
import torch
from torchinfo import summary
import random
from itertools import permutations

def print_model_info(model):
    input_size = (1, 3, 224, 224)
    model_summary = summary(model, input_size=input_size,
                            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
    total_flops = model_summary.total_mult_adds
    total_params = model_summary.total_params
    print(f"Total FLOPs: {total_flops} Total Params: {total_params}")

def set_random_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
