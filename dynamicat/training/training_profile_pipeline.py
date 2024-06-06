import sys
import os
import time

import math
import torch
import deepspeed
import transformers
from loguru import logger
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from torch.utils.tensorboard import SummaryWriter



SOURCE_ROOT_ABS_PATH = os.path.abspath(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir
        )
    )
sys.path.append(SOURCE_ROOT_ABS_PATH)


logger.info(f"adding {SOURCE_ROOT_ABS_PATH=} to python sys path")
logger.info(f"{sys.path=}")



from dynamicat.utils.common import print_rank_0, batch_dict_to_device, all_reduce_sum_of_tensor, all_reduce_mean_of_tensor, set_random_seeds
from dynamicat.collation.planned_data_collator import GeneralDataCollator
from dynamicat.training.training_args import parse_training_args, pretty_format_args
from dynamicat.utils.deepspeed_utils import DeepSpeedConfigBuilder, DeepSpeedModelTrainingUtils
from dynamicat.model.hf_model import HFModelProvider
from dynamicat.utils.performance_metrics import ThroughputMetrics, GPUUtilizationMetrics
from dynamicat.tokenization.hf_tokenzier import GeneralDatasetHfTokenizer
from dynamicat.tensorplanning.tensor_planner_profile import TensorPlannerForProfiling

def main():
    cmd_args = parse_training_args()
    logger.info(f"Command line arguments: {cmd_args}")

    dist_env_enabled = cmd_args.local_rank != -1

    if dist_env_enabled:
        torch.cuda.set_device(cmd_args.local_rank)
        device = torch.device("cuda", cmd_args.local_rank)
        deepspeed.init_distributed()
        cmd_args.global_rank = torch.distributed.get_rank()
        cmd_args.world_size = torch.distributed.get_world_size()

    else:
        device = torch.device("cuda")
        cmd_args.global_rank = -1
        cmd_args.world_size = 1


    cmd_args.accumulation_steps = cmd_args.global_batch_size // (cmd_args.batch_size_per_gpu * cmd_args.world_size)
    assert cmd_args.accumulation_steps > 0 and cmd_args.batch_size_per_gpu * cmd_args.world_size * cmd_args.accumulation_steps == cmd_args.global_batch_size, "invalid accumulation step config"

    # Load deepspeed configuration
    deepspeed_config = DeepSpeedConfigBuilder.make_config_for_training(
        return_dict=True,
        **cmd_args.__dict__
    )
    logger.info(f"DeepSpeed configuration: {deepspeed_config}")

    set_random_seeds(42)

    if dist_env_enabled:
        torch.distributed.barrier()

    # generate tensor plan for profiling
    tensor_planner = TensorPlannerForProfiling(5000, 18000)
    tensor_plan = tensor_planner.get_tensor_plan()
    print_rank_0(f"Tensor plan generated successfully, {len(tensor_plan)} items, {tensor_planner.pretty_format_plan()}", cmd_args.global_rank)

    # Load data collator & data loader for training
    collator = GeneralDataCollator(
        [tensor_planner.FIELD_NAME],
        [],
        True
    )

    data_loader = DataLoader(
        dataset=tensor_plan,
        batch_size=1,
        sampler= SequentialSampler(tensor_plan),
        collate_fn=collator.list_format_input_collate
    )
    num_training_steps_per_epoch = math.ceil(len(data_loader) / cmd_args.accumulation_steps)
    cmd_args.num_training_steps = num_training_steps_per_epoch * 1
    logger.info(f"DataLoader {data_loader} loaded successfully, {len(data_loader)} batches, "
                f"{num_training_steps_per_epoch} training steps per epoch, "
                f"{cmd_args.num_training_steps} total training steps")


    # Load model
    hf_model = HFModelProvider.load(
        hf_model_clz=getattr(transformers, cmd_args.model_clz),
        model_name_or_path=cmd_args.model_path,
        evaluation=False,
        use_flash_attn=cmd_args.use_flash_attn,
    )

    optimizer = DeepSpeedModelTrainingUtils.get_optimizer(
        hf_model,
        learning_rate=cmd_args.learning_rate,
        offload=cmd_args.zero_offload,
    )
    lr_scheduler = DeepSpeedModelTrainingUtils.get_lr_scheduler(
        optimizer,
        scheduler_type=cmd_args.scheduler_type,
        num_warmup_steps=cmd_args.num_warmup_steps,
        num_training_steps=cmd_args.num_training_steps
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=hf_model,
        optimizer=optimizer,
        args=cmd_args,
        config=deepspeed_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True
    )
    logger.info(f"DeepSpeed model loaded successfully, {model=}, {optimizer=}, {lr_scheduler=}")

    # Need this to reduce the memory consumption
    model.gradient_checkpointing_enable()

    # Load metrics
    tokenizer = GeneralDatasetHfTokenizer(cmd_args.model_path)
    tpt_metric = ThroughputMetrics(model.model, tokenizer.load_tokenizer(), True)
    logger.info(f"ThroughputMetrics loaded successfully, {tpt_metric=}, model params={tpt_metric.model_param_count}")
    del tokenizer

    if cmd_args.global_rank <= 0:
        gpu_metric = GPUUtilizationMetrics()
        logger.info(f"GPUUtilizationMetrics loaded successfully, {gpu_metric.device_handles=}")


    # Start training loop
    model.train()
    for epoch in range(1):
        print_rank_0(f"Epoch {epoch+1} of {cmd_args.num_epochs} started", cmd_args.global_rank)
        for step, batch in enumerate(data_loader):
            # step
            step_start_time = time.time()
            batch = batch_dict_to_device(batch, device)
            real_batch_size = batch['input_ids'].shape[0]
            real_seq_length = batch['input_ids'].shape[1]
            outputs = model(
                **batch,
                use_cache=False
            )
            model.backward(outputs.loss)
            model.step()
            step_stop_time = time.time()
            step_end_to_end_time = step_stop_time - step_start_time
            tflops = tpt_metric.get_throughput(step_end_to_end_time, real_batch_size, real_seq_length)

            # log metrics
            real_batch_size_tensor = torch.tensor(real_batch_size, device=device)
            batch_size_sum_across_ranks = all_reduce_sum_of_tensor(real_batch_size_tensor).item()

            step_end_to_end_time_tensor = torch.tensor(step_end_to_end_time, device=device)
            mean_step_end_to_end_time_across_ranks = all_reduce_mean_of_tensor(step_end_to_end_time_tensor).item()

            tflops_tensor = torch.tensor(tflops, device=device)
            mean_tflops_across_ranks = all_reduce_mean_of_tensor(tflops_tensor).item()


            if cmd_args.global_rank <= 0:
                gpu_util = gpu_metric.get_total_memory_utilization()
                used_memory = gpu_metric.get_total_used_memory()
                logger.info(f"count={real_batch_size * real_seq_length}({real_batch_size}*{real_seq_length}), gpu mem util={gpu_util*100:.2f}%, {used_memory=:.2f}GB")






if __name__ == '__main__':

    main()





