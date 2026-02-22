import math
from ast import literal_eval
import argparse
import copy
import importlib
import os

import torch
from easydict import EasyDict as edict
import omegaconf
from torch.distributed import init_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from rich import print
from utils.ddp_utils import unwrap_model
from utils import sp_support
import numpy as np
import random

init_process_group(backend="nccl")
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
ddp_node_rank = int(os.environ["GROUP_RANK"])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
torch.cuda.empty_cache()
torch.manual_seed(777 + ddp_rank) 
np.random.seed(777 + ddp_rank)
random.seed(777 + ddp_rank)

print(f"Process {ddp_rank}/{ddp_world_size} is using device {ddp_local_rank}/{ddp_local_world_size} on node {ddp_node_rank}")
torch.distributed.barrier()

parser = argparse.ArgumentParser(description="Override YAML values")
parser.add_argument("config", type=str, help="Path to YAML configuration file")
parser.add_argument("--load", type=str, default="", help="Force to load the weight from somewhere else")
parser.add_argument(
    "--set",
    "-s",
    type=str,
    action="append",
    nargs=2,
    metavar=("KEY", "VALUE"),
    help="New value for the key")
args = parser.parse_args()
config = omegaconf.OmegaConf.load(args.config)

def set_nested_key(data, keys, value):
    key = keys.pop(0)
    try:
        key = int(key)
    except ValueError:
        key = key
    if len(keys) > 0:
        if not isinstance(key, int) and key not in data:
            data[key] = {}
        set_nested_key(data[key], keys, value)
    else:
        try:
            data[key] = literal_eval(value)
        except (SyntaxError, ValueError):
            data[key] = value

if args.set is not None:
    for key_value in args.set:
        key, value = key_value
        if dist.get_rank() == 0:
            print(f"Overriding {key} with {value}")
        key_parts = key.split(".")
        set_nested_key(config, key_parts, value)

config = edict(config)
if ddp_rank == 0:
    print(config)

sp_size = config.get("sp_size", 1)
sp_support.init_sp_group(sp_size=sp_size)
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32

dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]

eval_config = copy.deepcopy(config)
eval_config.training.dataset_path = config.eval_dataset_path
eval_config.evaluation = True
eval_config.training.sample_ar = False
eval_config.training.sample_mixed_length = False
eval_config.training.data_repeat = 1
eval_dataset = Dataset(eval_config)
print(f"Eval dataset loaded! Length: {len(eval_dataset)}")
eval_data_len = len(eval_dataset)

dataloader_seed_generator = torch.Generator()
dataloader_seed_generator.manual_seed(95 + sp_support.get_sp_replica_id())

eval_datasampler = DistributedSampler(
    eval_dataset,
    num_replicas=sp_support.get_sp_replicas(),
    rank=sp_support.get_sp_replica_id(),
    shuffle=False,
    drop_last=False,
)
dataloader_eval_seed_generator = torch.Generator()
dataloader_eval_seed_generator.manual_seed(95)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    generator=dataloader_eval_seed_generator,
    sampler=eval_datasampler,
)

module, class_name = config.model.class_name.rsplit(".", 1)
tttLRM = importlib.import_module(module).__dict__[class_name]
model = tttLRM(config).to(device)
model_overview = model.get_overview()

checkpoint = torch.load(args.load, map_location="cpu")
model.load_state_dict(checkpoint['model'], strict=False)
model = DDP(model, device_ids=[ddp_local_rank])
model = torch.compile(model)  # pytorch 2.0 feature

if config.inference or config.get("evaluation", False):

    if config.inference:
        print(f"Running inference; save results to: {config.inference_out_dir}")
    else:
        os.makedirs(config.evaluation_out_dir, exist_ok=True)
        print(f"Running evaluation; save results to: {config.evaluation_out_dir}")
        if ddp_rank == 0:
            print("Downloading LPIPS model using ddip_rank=0; this is to avoid multiple processes downloading at the same time")
            import lpips

    torch.distributed.barrier()

    eval_datasampler.set_epoch(0)

    model.eval()
    with model.no_sync(), torch.no_grad(), torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        eval_iters = int(math.ceil(eval_data_len / sp_support.get_sp_replicas()))

        for i, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            print(f"Data loaded {i}, rank {dist.get_rank()}, sp_rank {sp_support.get_sp_rank()}, ")
            result = unwrap_model(model)(batch)

            if sp_support.get_sp_rank() == 0:
                model.module.save_evaluations(config.evaluation_out_dir, result, batch)
            torch.distributed.barrier()

            if i >= eval_iters - 1:
                break
            torch.cuda.empty_cache()

    torch.distributed.barrier()
    exit(0)
