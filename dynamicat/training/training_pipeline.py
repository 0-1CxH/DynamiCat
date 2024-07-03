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
from dynamicat.model.deepspeed_model import DeepSpeedHFModelProvider
from dynamicat.tokenization.filebase_dataset import FileBaseDatasetMetadata
from dynamicat.collation.planned_data_collator import GeneralDataCollator
from dynamicat.training.training_args import parse_training_args, pretty_format_args
from dynamicat.utils.deepspeed_utils import DeepSpeedConfigBuilder, DeepSpeedModelTrainingUtils
from dynamicat.model.hf_model import HFModelProvider
from dynamicat.utils.performance_metrics import ThroughputMetrics, GPUUtilizationMetrics
from dynamicat.tokenization.hf_tokenzier import GeneralDatasetHfTokenizer

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

    # load planned tensor file
    tensor_plan = torch.load(cmd_args.planned_tensor_file_path)

    # Load dataset metadata
    if cmd_args.dataset_metadata_path:
        dataset_metadata = FileBaseDatasetMetadata.load_from_file(cmd_args.dataset_metadata_path)
        field_names_to_use = dataset_metadata.field_names
        loss_masked_field_names_to_use = dataset_metadata.loss_masked_field_names
    else:
        if cmd_args.dataset_specific_task_type == "sft":
            field_names_to_use = ["prompt", "chosen"]
            loss_masked_field_names_to_use = ["prompt"]
        elif cmd_args.dataset_specific_task_type == "pt":
            field_names_to_use = ["content"]
            loss_masked_field_names_to_use = []
        else:
            raise ValueError(f"Unknown dataset specific task type: {cmd_args.dataset_specific_task_type}")

    logger.info(f"field_names_to_use: {field_names_to_use}, loss_masked_field_names_to_use: {loss_masked_field_names_to_use}")
    # Load data collator & data loader for training
    collator = GeneralDataCollator(
        field_names_to_use,
        loss_masked_field_names_to_use,
        True
    )
    data_loader = DataLoader(
        dataset=tensor_plan,
        batch_size=1,
        sampler=DistributedSampler(tensor_plan) if dist_env_enabled else SequentialSampler(tensor_plan),
        collate_fn=collator.list_format_input_collate
    )
    num_training_steps_per_epoch = math.ceil(len(data_loader) / cmd_args.accumulation_steps)
    cmd_args.num_training_steps = num_training_steps_per_epoch * cmd_args.num_epochs
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


    # Add tensorboard for loss
    tensorboard_writer = None
    if cmd_args.use_tensorboard:
        tensorboard_writer = SummaryWriter(
            log_dir=os.path.join(cmd_args.tensorboard_save_path, cmd_args.tensorboard_job_name),
            flush_secs=15
        )
    logger.info(f"Tensorboard writer: {tensorboard_writer}")

    # print check list before training starts
    print_rank_0("ARG CHECKLIST:", cmd_args.global_rank)
    print_rank_0(pretty_format_args(cmd_args), cmd_args.global_rank)


    # Start training loop
    model.train()
    for epoch in range(cmd_args.num_epochs):
        print_rank_0(f"Epoch {epoch+1} of {cmd_args.num_epochs} started", cmd_args.global_rank)
        for step, batch in enumerate(data_loader):
            # step
            step_start_time = time.time()
            batch = batch_dict_to_device(batch, device)
            real_batch_size = batch['input_ids'].shape[0]
            real_seq_length = batch['input_ids'].shape[1]
            loss_normalization = real_batch_size / cmd_args.batch_size_per_gpu
            outputs = model(
                **batch,
                use_cache=False
            )
            raw_loss = outputs.loss
            normalized_loss = raw_loss * loss_normalization
            model.backward(normalized_loss)
            model.step()
            step_stop_time = time.time()
            step_end_to_end_time = step_stop_time - step_start_time
            tflops = tpt_metric.get_throughput(step_end_to_end_time, real_batch_size, real_seq_length)
            logger.debug(f"Rank {cmd_args.global_rank}: "
                         f"raw_loss={raw_loss.item()}, "
                         f"real_batch_size={real_batch_size}, "
                         f"normalized_loss={normalized_loss.item()},"
                         f"time_used={step_end_to_end_time},"
                         f"tflops={tflops:.4f}")

            # log metrics
            real_batch_size_tensor = torch.tensor(real_batch_size, device=device)
            batch_size_sum_across_ranks = all_reduce_sum_of_tensor(real_batch_size_tensor).item()
            normalized_loss_sum_across_ranks = all_reduce_sum_of_tensor(normalized_loss).item()
            mean_loss_per_device = normalized_loss_sum_across_ranks / batch_size_sum_across_ranks * cmd_args.batch_size_per_gpu

            step_end_to_end_time_tensor = torch.tensor(step_end_to_end_time, device=device)
            mean_step_end_to_end_time_across_ranks = all_reduce_mean_of_tensor(step_end_to_end_time_tensor).item()
            mean_sample_per_second = batch_size_sum_across_ranks / mean_step_end_to_end_time_across_ranks

            tflops_tensor = torch.tensor(tflops, device=device)
            mean_tflops_across_ranks = all_reduce_mean_of_tensor(tflops_tensor).item()


            print_rank_0(f"{epoch=} of {cmd_args.num_epochs}, {step=} of {num_training_steps_per_epoch}: "
                         f"time used(mean)={mean_step_end_to_end_time_across_ranks:.4f} "
                         f"batch size(sum)={batch_size_sum_across_ranks} "
                         f"loss(mean)={mean_loss_per_device:.4f} "
                         f"samples per sec(mean)={mean_sample_per_second:.3f} "
                         f"tflops(mean)={mean_tflops_across_ranks:.4f}",
                         cmd_args.global_rank)
            if tensorboard_writer:
                if cmd_args.global_rank <= 0:
                    global_step_count = epoch * num_training_steps_per_epoch + step
                    tensorboard_writer.add_scalar("loss (mean)", mean_loss_per_device, global_step_count)
                    tensorboard_writer.add_scalar("batch size (sum)", batch_size_sum_across_ranks, global_step_count)
                    tensorboard_writer.add_scalar("time (mean)", mean_step_end_to_end_time_across_ranks, global_step_count)
                    tensorboard_writer.add_scalar("samples per sec (mean)", mean_sample_per_second, global_step_count)
                    tensorboard_writer.add_scalar("tflops (mean)", mean_tflops_across_ranks, global_step_count)
                    # for gpu_metric_key, gpu_metric_value in gpu_metric.iterate_key_values():
                    #     tensorboard_writer.add_scalar(gpu_metric_key, gpu_metric_value, global_step_count)
                    tensorboard_writer.add_scalar("gpu util (mean)", gpu_metric.get_total_memory_utilization(), global_step_count)

            if (step % cmd_args.save_interval == 0 and step > 0) or (epoch > 0 and step == 0):
                print_rank_0(f"Saving model at {epoch=} {step=}", cmd_args.global_rank)
                DeepSpeedHFModelProvider.save(
                    model,
                    os.path.join(cmd_args.checkpoint_save_path, f"checkpoint-ep{epoch}-step{step}"),
                    global_rank=cmd_args.global_rank,
                    is_zero_stage_3=cmd_args.zero_stage==3
                )

    # finish
    print_rank_0(f"Saving final model", cmd_args.global_rank)
    DeepSpeedHFModelProvider.save(
        model,
        os.path.join(cmd_args.checkpoint_save_path),
        global_rank=cmd_args.global_rank,
        is_zero_stage_3=cmd_args.zero_stage==3
    )




if __name__ == '__main__':

    main()





