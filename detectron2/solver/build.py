# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List
import torch

from detectron2.config import CfgNode

from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR

from detectron2.utils import comm
class AccSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc_period = 2
        self.acc_cnt = 0

    def zero_grad(self):
        if (self.acc_cnt % self.acc_period) == 0:
            if comm.is_main_process():
                print("ZG")
            super().zero_grad()

    def step(self):
        if ((self.acc_cnt + 1) % self.acc_period) == 0:
            super().step()
        self.acc_cnt += 1

def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params: List[Dict[str, Any]] = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
        elif key.endswith(".bias"):
            # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
            # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
            # hyperparameters are by default exactly the same as for regular
            # weights.
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.ACC:
        optimizer = AccSGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
