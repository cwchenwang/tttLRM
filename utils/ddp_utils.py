import os
import torch
import torch.distributed as dist
from rich import print

def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model
