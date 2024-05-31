import torch
from loguru import logger

from dynamicat.model.deepspeed_model import DeepSpeedHFModelProvider
from dynamicat.training.training_args import parse_training_args, pretty_print_args
from dynamicat.utils.deepspeed_utils import DeepSpeedConfigBuilder


def main():
    cmd_args = parse_training_args()
    logger.info(f"Command line arguments: {cmd_args}")
    dist_env_enabled = cmd_args.local_rank != -1

    deepspeed_config = DeepSpeedConfigBuilder.make_config_for_training(
        cmd_args.global_batch_size,
        cmd_args.batch_size_per_gpu,
        cmd_args.zero_stage,
        cmd_args.zero_offload,
        cmd_args.use_bf16,
        cmd_args.use_tensorboard,
        return_dict=True,
        enable_hybrid_engine=cmd_args.enable_hybrid_engine,
        tensorboard_save_path=cmd_args.tensorboard_save_path,
        **cmd_args.__dict__
    )
    logger.info(f"DeepSpeed configuration: {deepspeed_config}")

    model = DeepSpeedHFModelProvider.load(
        hf_model_clz=exec(cmd_args.model_clz),
        model_path=cmd_args.model_path,
        cmd_args=cmd_args,
        ds_config=deepspeed_config,
        learning_rate=cmd_args.learning_rate,
        offload=cmd_args.zero_offload,
        scheduler_type=cmd_args.scheduler_type,
        num_warmup_steps=cmd_args.num_warmup_steps,
        num_training_steps=cmd_args.num_training_steps,
        **cmd_args.__dict__
    )

    cmd_args.global_rank = torch.distributed.get_rank() if dist_env_enabled else -1
    cmd_args.world_size = torch.distributed.get_world_size() if dist_env_enabled else 1

    logger.info(pretty_print_args(cmd_args))





