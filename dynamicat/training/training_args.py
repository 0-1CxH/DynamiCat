import argparse
import deepspeed


def parse_training_args():
    parser = argparse.ArgumentParser()

    # dist
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    # data
    parser.add_argument('--global_batch_size', type=int, help='global batch size, which is the total batch size across all gpus * gradient accumulation steps')
    parser.add_argument('--batch_size_per_gpu', type=int, help='batch size per gpu for each step', default=1)

    # dataset
    parser.add_argument('--dataset_metadata_path', type=str, help='Path to dataset metadata', default=None)
    parser.add_argument('--dataset_folder_path', type=str, help='Path to dataset folder', default=None)
    parser.add_argument('--max_sequence_lengths', type=int, help='Maximum sequence lengths', default=4096)
    parser.add_argument('--dataset_file_format', type=str, help='Dataset file format', default='jsonl')

    # tokenizer
    parser.add_argument('--tokenizer_path', type=str, help='Path to tokenizer')

    # tensor planner
    parser.add_argument('--tensor_planner_type', type=str, help='Tensor planner type')
    parser.add_argument('--tensor_parameter_count_limit', type=int, help='Tensor parameter count limit', default=15000)
    parser.add_argument('--primary_key', type=str, help='Primary key', default='content')
    parser.add_argument('--max_token_diff', type=int, help='Maximum token difference', default=16)
    # parser.add_argument('--max_plan_size', type=int, help='Maximum plan size', default=64)
    # parser.add_argument('--max_field_length', type=int, help='Maximum field length', default=1024)


    # model
    parser.add_argument('--model_clz', type=str, help='Model class', default='LlamaForCausalLM')
    parser.add_argument('--model_path', type=str, help='Path to model')

    # zero
    parser.add_argument('--zero_stage', type=int, help='DeepSpeed ZERO stage, should be 0, 1, 2, or 3')
    parser.add_argument('--zero_offload', type=bool, help='DeepSpeed ZERO offload', default=True)

    # fp16 or bf16
    parser.add_argument('--use_bf16', action='store_true', help='Use BFLOAT16 precision')
    parser.add_argument('--loss_scale_window', type=int, help='Loss scale window', default=100)

    # optimizer and lr scheduler
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--scheduler_type', type=str, help='Scheduler type', default="cosine")
    parser.add_argument('--num_warmup_steps', type=int, help='Number of warmup steps', default=0)
    # parser.add_argument('--num_training_steps', type=int, help='Number of training steps', default=100000)

    # tensorboard
    parser.add_argument('--use_tensorboard', action='store_true', help='Use tensorboard')
    parser.add_argument('--tensorboard_save_path', type=str, help='Path to save tensorboard logs', default='.')
    parser.add_argument('--tensorboard_job_name', type=str, help='Tensorboard job name', default='training_tensorboard')

    # hybrid engine
    parser.add_argument('--enable_hybrid_engine', action='store_true', help='Enable DeepSpeed hybrid engine')
    parser.add_argument('--inference_tp_size', type=int, help='Inference tensor parallel size', default=1)
    parser.add_argument('--release_inference_cache', action='store_true', help='Release inference cache')
    parser.add_argument('--pin_parameters', action='store_true', help='Pin parameters')
    parser.add_argument('--tp_gather_partition_size', type=int, help='Tensor parallel gather partition size', default=8)
    parser.add_argument('--max_out_tokens', type=int, help='Maximum output tokens', default=512)

    # flash attention
    parser.add_argument('--use_flash_attn', action='store_true', help='Use Flash Attention')

    # training
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')

    # save
    parser.add_argument("--checkpoint_save_path", type=str)
    parser.add_argument("--save_interval", type=int, default=200)




    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()
    return cmd_args


def pretty_format_args(parsed_cmd_args):
    assert hasattr(parsed_cmd_args, 'world_size'), "Command line arguments should have world size"

    if parsed_cmd_args.dataset_metadata_path:
        dataset_args = f"\tdataset_metadata_path: {parsed_cmd_args.dataset_metadata_path}\n"
    else:
        dataset_args = f"\tdataset_folder_path: {parsed_cmd_args.dataset_folder_path}\n"
    dataset_args += (f"\tmax_sequence_lengths: {parsed_cmd_args.max_sequence_lengths}\n"
                     f"\tdataset_file_format: {parsed_cmd_args.dataset_file_format}\n")

    if parsed_cmd_args.tensor_planner_type == 'LengthDifferenceRestricted':
        tensor_planner_args = (f"\tmax_token_diff: {parsed_cmd_args.max_token_diff}\n"
                               f"\tprimary_key: {parsed_cmd_args.primary_key}\n")
    elif parsed_cmd_args.tensor_planner_type == 'MaxLengthRestricted':
        tensor_planner_args = (f"\tmax_field_length: {parsed_cmd_args.max_sequence_lengths}\n"
                               f"\tprimary_key: {parsed_cmd_args.primary_key}\n")
    elif parsed_cmd_args.tensor_planner_type == 'GPUMemoryRestricted':
        tensor_planner_args = f"\ttensor_parameter_count_limit: {parsed_cmd_args.tensor_parameter_count_limit}\n"
    else: # fixed batch size
        tensor_planner_args = f"\tbatch_size: {parsed_cmd_args.batch_size_per_gpu}\n"

    return  (
        f"DATA CONFIG:\n"
        f"\tglobal batch size: {parsed_cmd_args.global_batch_size}\n"
        f"\tbatch size per GPU: {parsed_cmd_args.batch_size_per_gpu}\n"
        f"\tworld size: {parsed_cmd_args.world_size}\n"
        f"\taccumulation steps: {parsed_cmd_args.accumulation_steps}\n"
        f"DATASET CONFIG:\n"
        f"{dataset_args}"
        f"MODEL CONFIG:\n"
        f"\tmodel class: {parsed_cmd_args.model_clz}\n"
        f"\tmodel path: {parsed_cmd_args.model_path}\n"
        f"TOKENIZER CONFIG:\n"
        f"\ttokenizer path: {parsed_cmd_args.tokenizer_path}\n"
        f"TENSOR PLANNER CONFIG:\n"
        f"\ttensor planner type: {parsed_cmd_args.tensor_planner_type}\n"
        f"{tensor_planner_args}"
        f"ZERO CONFIG:\n"
        f"\tzero stage: {parsed_cmd_args.zero_stage}\n"
        f"\tzero offload: {parsed_cmd_args.zero_offload}\n"
        f"FP16 CONFIG:\n"
        f"\tuse BF16: {parsed_cmd_args.use_bf16}\n"
        f"\tuse FP16: {not parsed_cmd_args.use_bf16}\n"
        f"OPTIMIZER & LR SCHEDULER CONFIG:\n"
        f"\tlearning rate: {parsed_cmd_args.learning_rate}\n"
        f"\tscheduler type: {parsed_cmd_args.scheduler_type}\n"
        f"\tnumber of warmup steps: {parsed_cmd_args.num_warmup_steps}\n"
        f"\tnumber of training steps: {parsed_cmd_args.num_training_steps}\n"
        f"TENSORBOARD CONFIG:\n"
        f"\tuse tensorboard: {parsed_cmd_args.use_tensorboard}\n"
        f"\ttensorboard save path: {parsed_cmd_args.tensorboard_save_path}\n"
        f"\ttensorboard job name: {parsed_cmd_args.tensorboard_job_name}\n"
        f"TRAINING CONFIG:\n"
        f"\tnumber of epochs: {parsed_cmd_args.num_epochs}\n"
        f"\tsteps per epoch: {parsed_cmd_args.num_training_steps // parsed_cmd_args.num_epochs}"
        f"SAVE CONFIG:\n"
        f"\tmodel save path: {parsed_cmd_args.checkpoint_save_path}"
        f"\tsave each {parsed_cmd_args.save_interval} steps"

    )