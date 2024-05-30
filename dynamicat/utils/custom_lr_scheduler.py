from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _wsd_schedule_lambda(
    current_step: int, *, num_warmup_steps: int, num_stable_steps: int, num_decay_steps: int, min_lr: float
):
    if current_step < num_warmup_steps:
        return max(float(current_step) / float(max(1, num_warmup_steps)), min_lr)
    elif current_step < num_warmup_steps + num_stable_steps:
        return 1.0
    else:
        decay_progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
        return max(min_lr, 1.0 - decay_progress)



def get_wsd_scheduler(
    optimizer: Optimizer, warmup_ratio: float, decay_ratio: float, num_training_steps: int, last_epoch: int = -1
):
    assert 0 <= warmup_ratio <= 1 and 0 <= decay_ratio <= 1 and 0 <= warmup_ratio + decay_ratio <= 1
    lr_lambda = partial(
        _wsd_schedule_lambda,
        num_warmup_steps=int(num_training_steps * warmup_ratio),
        num_stable_steps=int(num_training_steps * (1 - warmup_ratio - decay_ratio)),
        num_decay_steps=int(num_training_steps * decay_ratio),
        min_lr=1e-10
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


