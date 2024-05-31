import argparse
import deepspeed
import os
import torch
from transformers import LlamaForCausalLM

from dynamicat.model.deepspeed_model import DeepSpeedHFModelProvider
from dynamicat.utils.deepspeed_utils import DeepSpeedConfigBuilder

if __name__ == '__main__':
    test_folder_base = os.path.dirname(os.path.dirname(__file__))
    print(test_folder_base)

    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    dist_env_enabled = cmd_args.local_rank != -1

    # if not dist_env_enabled:
    #     device = torch.device("cuda")
    #     logger.info(f"Training on single GPU, device: {device}")
    # else:
    #     device = torch.device("cuda", cmd_args.local_rank)
    #     deepspeed.init_distributed()
    #     logger.info(f"Training on multiple GPUs, device on current rank: {device}")



    print(cmd_args)

    ds_config = DeepSpeedConfigBuilder.make_config_for_training(
        128,
        1,
        3,
        False,
        False,
        False,
        return_dict=True,
        tensorboard_save_path="test_path_1111",
        enable_hybrid_engine=False
    )

    # if dist_env_enabled:
    #     torch.distributed.barrier()

    model_path = os.path.join(test_folder_base, "test_model")

    model = DeepSpeedHFModelProvider.load(
        LlamaForCausalLM,
        model_path,
        cmd_args=cmd_args,
        ds_config=ds_config,
        # optimizer args
        learning_rate=1e-4,
        offload=True,
        # lr scheduler args
        scheduler_type="cosine",
        num_warmup_steps=10,
        num_training_steps=2000
    )

    cmd_args.global_rank = torch.distributed.get_rank() if dist_env_enabled else -1
    # cmd_args.world_size = torch.distributed.get_world_size() if dist_env_enabled else 1

    DeepSpeedHFModelProvider.save(
        model,
        os.path.join(test_folder_base, "test_model_save_ds"),
        global_rank=cmd_args.global_rank,
        is_zero_stage_3=True
    )



