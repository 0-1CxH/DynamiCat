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
    parser.add_argument("--dataset_specific_task_type", type=str, default="sft")
    parser.add_argument("--planned_tensor_file_path", type=str)
    # model
    parser.add_argument('--model_clz', type=str, help='Model class', default='LlamaForCausalLM')
    parser.add_argument('--model_path', type=str, help='Path to model')

    # zero
    parser.add_argument('--zero_stage', type=int, help='DeepSpeed ZERO stage, should be 0, 1, 2, or 3')
    parser.add_argument('--zero_offload', action='store_true', help='DeepSpeed ZERO offload')

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


    return  (
        f"DATA CONFIG:\n"
        f"\tglobal batch size: {parsed_cmd_args.global_batch_size}\n"
        f"\tbatch size per GPU: {parsed_cmd_args.batch_size_per_gpu}\n"
        f"\tworld size: {parsed_cmd_args.world_size}\n"
        f"\taccumulation steps: {parsed_cmd_args.accumulation_steps}\n"
        f"DATASET CONFIG:\n"
        f"\tdataset metadata path: {parsed_cmd_args.dataset_metadata_path}\n"
        f"\tdataset specific task type: {parsed_cmd_args.dataset_specific_task_type}\n"
        f"\tplanned tensor file path: {parsed_cmd_args.planned_tensor_file_path}\n"
        f"MODEL CONFIG:\n"
        f"\tmodel class: {parsed_cmd_args.model_clz}\n"
        f"\tmodel path: {parsed_cmd_args.model_path}\n"
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
        f"\tsteps per epoch: {parsed_cmd_args.num_training_steps // parsed_cmd_args.num_epochs}\n"
        f"SAVE CONFIG:\n"
        f"\tmodel save path: {parsed_cmd_args.checkpoint_save_path}\n"
        f"\tsave interval: {parsed_cmd_args.save_interval}\n"

    )