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
from dynamicat.tensorplanning.tensor_planner import TensorPlannerFactory
from dynamicat.tokenization.filebase_dataset import FileBaseDatasetMetadata, FileBaseDataset
from dynamicat.tokenization.hf_tokenzier import GeneralDatasetHfTokenizer
from dynamicat.collation.planned_data_collator import GeneralDataCollator
from dynamicat.tokenization.task_specific_filebase_dataset import DefaultTaskSpecificFileBaseDatasetMetadataFactory
from dynamicat.training.training_args import parse_training_args, pretty_format_args
from dynamicat.utils.deepspeed_utils import DeepSpeedConfigBuilder


def main():
    cmd_args = parse_training_args()
    logger.info(f"Command line arguments: {cmd_args}")
    dist_env_enabled = cmd_args.local_rank != -1

    if not dist_env_enabled:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(cmd_args.local_rank)
        device = torch.device("cuda", cmd_args.local_rank)

    set_random_seeds(42)

    if dist_env_enabled:
        deepspeed.init_distributed()

    cmd_args.global_rank = torch.distributed.get_rank() if dist_env_enabled else -1
    cmd_args.world_size = torch.distributed.get_world_size() if dist_env_enabled else 1
    cmd_args.accumulation_steps = cmd_args.global_batch_size // (cmd_args.batch_size_per_gpu * cmd_args.world_size)
    assert cmd_args.accumulation_steps > 0 and cmd_args.batch_size_per_gpu * cmd_args.world_size * cmd_args.accumulation_steps == cmd_args.global_batch_size, "invalid accumulation step config"


    # Load tokenizer
    tokenizer = GeneralDatasetHfTokenizer(cmd_args.tokenizer_path)
    tokenizer.load_tokenizer()

    # Load dataset
    if cmd_args.dataset_metadata_path:
        dataset_metadata =  FileBaseDatasetMetadata.load_from_file(cmd_args.dataset_metadata_path)
    else:
        assert cmd_args.dataset_folder_path, "dataset_folder_path is required"
        if cmd_args.dataset_specific_task_type == "pt":
            dataset_metadata = DefaultTaskSpecificFileBaseDatasetMetadataFactory.make_pretrain_metadata(
                pretrain_dataset_folder_path=cmd_args.dataset_folder_path,
                pretrain_max_sequence_lengths=cmd_args.max_sequence_lengths,
                pretrain_file_format=cmd_args.dataset_file_format
            )
        else: # default to sft
            dataset_metadata = DefaultTaskSpecificFileBaseDatasetMetadataFactory.make_sft_metadata(
                sft_dataset_folder_path=cmd_args.dataset_folder_path,
                sft_max_sequence_lengths=cmd_args.max_sequence_lengths,
                sft_file_format=cmd_args.dataset_file_format
            )

        # TODO: add pt dataset here

    dataset = FileBaseDataset(dataset_metadata)
    dataset.load()
    logger.info(f"Dataset {dataset_metadata} loaded successfully, {len(dataset)} records")
    tokenized_tensors_generator = dataset.dataset_to_tensors(tokenizer.text_to_tensor)
    tokenized_tensors = list(tokenized_tensors_generator)

    # Load tensor planner
    tensor_planner = TensorPlannerFactory.create_tensor_planner(
        plan_type=cmd_args.tensor_planner_type,
        batch_size=cmd_args.batch_size_per_gpu,
        tensor_parameter_count_limit=cmd_args.tensor_parameter_count_limit,
        primary_key=cmd_args.primary_key,
        max_token_diff=cmd_args.max_token_diff,
        max_plan_size=cmd_args.batch_size_per_gpu,
        max_field_length=cmd_args.max_sequence_lengths
    )
    tensor_plan = tensor_planner.plan_tensor_records(tokenized_tensors)
    logger.debug(tensor_plan.formatted_string_of_whole_plan())
    logger.info(f"Tensor plan loaded successfully, {len(tensor_plan)} records")

    # Load data collator & data loader for training
    collator = GeneralDataCollator(
        dataset_metadata.field_names,
        dataset_metadata.loss_masked_field_names,
        True
    )
    logger.info(f"Data collator loaded: {collator}")

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


    # Load deepspeed configuration
    deepspeed_config = DeepSpeedConfigBuilder.make_config_for_training(
        return_dict=True,
        **cmd_args.__dict__
    )
    logger.info(f"DeepSpeed configuration: {deepspeed_config}")

    # Load model
    model = DeepSpeedHFModelProvider.load(
        hf_model_clz=getattr(transformers, cmd_args.model_clz),
        model_name_or_path=cmd_args.model_path,
        cmd_args=cmd_args,
        ds_config=deepspeed_config,
        offload=cmd_args.zero_offload,
        **cmd_args.__dict__
    )

    # Add tensorboard for loss
    tensorboard_writer = None
    if cmd_args.use_tensorboard:
        tensorboard_writer = SummaryWriter(
            log_dir=os.path.join(cmd_args.tensorboard_save_path, cmd_args.tensorboard_job_name),
            flush_secs=15
        )
    logger.info(f"Tensorboard writer: {tensorboard_writer}")

    # print check list before training starts
    print_rank_0("CHECKLIST:")
    print_rank_0(pretty_format_args(cmd_args))


    # Start training loop
    model.train()
    for epoch in range(cmd_args.num_epochs):
        print_rank_0(f"Epoch {epoch+1} of {cmd_args.num_epochs} started", cmd_args.global_rank)
        for step, batch in enumerate(data_loader):
            # step
            step_start_time = time.time()
            batch = batch_dict_to_device(batch, device)
            real_batch_size = batch['input_ids'].shape[0]
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
            logger.debug(f"Rank {cmd_args.global_rank}: "
                         f"raw_loss={raw_loss.item()}, "
                         f"real_batch_size={real_batch_size}, "
                         f"normalized_loss={normalized_loss.item()},"
                         f"time_used={step_end_to_end_time}")

            # log metrics
            step_end_to_end_time_tensor = torch.tensor(step_end_to_end_time, device=device)
            mean_step_end_to_end_time_across_ranks = all_reduce_mean_of_tensor(step_end_to_end_time_tensor).item()
            real_batch_size_tensor = torch.tensor(real_batch_size, device=device)
            batch_size_sum_across_ranks = all_reduce_sum_of_tensor(real_batch_size_tensor).item()
            normalized_loss_sum_across_ranks = all_reduce_sum_of_tensor(normalized_loss).item()
            mean_loss_per_device = normalized_loss_sum_across_ranks / batch_size_sum_across_ranks * cmd_args.batch_size_per_gpu
            print_rank_0(f"{epoch=}, {step=}: time used (mean)={mean_step_end_to_end_time_across_ranks:.3f} "
                         f"batch size(sum)={batch_size_sum_across_ranks} "
                         f"loss(mean)={mean_loss_per_device}", cmd_args.global_rank)
            if tensorboard_writer:
                if cmd_args.global_rank <= 0:
                    global_step_count = epoch * num_training_steps_per_epoch + step
                    tensorboard_writer.add_scalar("loss (mean)", mean_loss_per_device, global_step_count)
                    tensorboard_writer.add_scalar("batch size (sum)", batch_size_sum_across_ranks, global_step_count)
                    tensorboard_writer.add_scalar("time (mean)", mean_step_end_to_end_time_across_ranks, global_step_count)

            if (step % cmd_args.save_interval == 0 and step > 0) or (epoch > 0 and step == 0):
                print_rank_0(f"Saving model at {epoch=} {step=}", cmd_args.global_rank)
                DeepSpeedHFModelProvider.save(
                    model,
                    os.path.join(cmd_args.checkpoint_save_path, f"checkpoint-ep{epoch}-step{step}"),
                    global_rank=cmd_args.global_rank,
                    is_zero_stage_3=cmd_args.zero_offload
                )

    # finish
    print_rank_0(f"Saving final model", cmd_args.global_rank)
    DeepSpeedHFModelProvider.save(
        model,
        os.path.join(cmd_args.checkpoint_save_path),
        global_rank=cmd_args.global_rank,
        is_zero_stage_3=cmd_args.zero_offload
    )




if __name__ == '__main__':

    main()





