import sys
import os

import transformers

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir
        )
    )
)


import torch
from loguru import logger

from dynamicat.model.deepspeed_model import DeepSpeedHFModelProvider
from dynamicat.tensorplanning.tensor_planner import TensorPlannerFactory
from dynamicat.tokenization.filebase_dataset import FileBaseDatasetMetadata, FileBaseDataset
from dynamicat.tokenization.hf_tokenzier import GeneralDatasetHfTokenizer
from dynamicat.tokenization.task_specific_filebase_dataset import DefaultTaskSpecificFileBaseDatasetMetadataFactory
from dynamicat.training.training_args import parse_training_args, pretty_print_args
from dynamicat.utils.deepspeed_utils import DeepSpeedConfigBuilder


def main():
    cmd_args = parse_training_args()
    logger.info(f"Command line arguments: {cmd_args}")
    dist_env_enabled = cmd_args.local_rank != -1

    # Load tokenizer
    tokenizer = GeneralDatasetHfTokenizer(cmd_args.tokenizer_path)
    tokenizer.load_tokenizer()

    # Load dataset
    if cmd_args.dataset_metadata_path:
        dataset_metadata =  FileBaseDatasetMetadata.load_from_file(cmd_args.dataset_metadata_path)
    else:
        assert cmd_args.dataset_folder_path, "dataset_folder_path is required"
        dataset_metadata = DefaultTaskSpecificFileBaseDatasetMetadataFactory.make_sft_metadata(
            sft_dataset_folder_path=cmd_args.dataset_folder_path,
            sft_max_sequence_lengths=cmd_args.max_sequence_lengths,
            sft_file_format=cmd_args.dataset_file_format
        )

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
    logger.info(tensor_plan.formatted_string_of_whole_plan())

    # Load deepspeed configuration
    deepspeed_config = DeepSpeedConfigBuilder.make_config_for_training(
        return_dict=True,
        **cmd_args.__dict__
    )
    logger.info(f"DeepSpeed configuration: {deepspeed_config}")

    # Load model

    model = DeepSpeedHFModelProvider.load(
        hf_model_clz=getattr(transformers, cmd_args.model_clz), #exec(cmd_args.model_clz),
        model_name_or_path=cmd_args.model_path,
        cmd_args=cmd_args,
        ds_config=deepspeed_config,
        offload=cmd_args.zero_offload,
        **cmd_args.__dict__
    )

    cmd_args.global_rank = torch.distributed.get_rank() if dist_env_enabled else -1
    cmd_args.world_size = torch.distributed.get_world_size() if dist_env_enabled else 1

    logger.info(pretty_print_args(cmd_args))


if __name__ == '__main__':

    main()





