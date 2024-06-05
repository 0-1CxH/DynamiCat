import os
import sys
import torch


SOURCE_ROOT_ABS_PATH = os.path.abspath(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir
        )
    )
sys.path.append(SOURCE_ROOT_ABS_PATH)

import argparse
from loguru import logger
from dynamicat.tensorplanning.tensor_planner import TensorPlannerFactory


def parse_tensor_planning_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Input file path")
    parser.add_argument("--output_path", type=str, help="Output file path")
    parser.add_argument('--tensor_planner_type', type=str, help='Tensor planner type', required=True)
    parser.add_argument('--tensor_parameter_count_limit', type=int, help='Tensor parameter count limit')
    parser.add_argument("--plan_order_type", type=str, help="Plan order type")
    parser.add_argument('--primary_key', type=str, help='Primary key')
    parser.add_argument('--max_token_diff', type=int, help='Maximum token difference')
    parser.add_argument('--batch_size', type=int, help='Maximum plan size')
    parser.add_argument('--max_field_length', type=int, help='Maximum field length')
    parser.add_argument("--enable_smart_batching", action="store_true", help="Enable smart batching")
    return parser.parse_args()

def plan_tensor(cmd_args):
    assert cmd_args.tensor_planner_type in TensorPlannerFactory.PlanTypeToClzMap, f"Invalid tensor planner type: {cmd_args.tensor_planner_type}"
    print(cmd_args)

    tensor_planner = TensorPlannerFactory.create_tensor_planner(
        plan_type=cmd_args.tensor_planner_type,
        batch_size=cmd_args.batch_size, # use in FixedBatchSize and MaxLengthRestricted
        tensor_parameter_count_limit=cmd_args.tensor_parameter_count_limit, # use in GPUMemoryRestricted
        primary_key=cmd_args.primary_key, # use in LengthDifferenceRestricted and MaxLengthRestricted
        max_token_diff=cmd_args.max_token_diff, # use in LengthDifferenceRestricted
        max_plan_size=cmd_args.batch_size, # use in LengthDifferenceRestricted
        max_field_length=cmd_args.max_field_length # use in MaxLengthRestricted
    )

    # load tensor records from input file
    tensor_records = torch.load(cmd_args.input_path)
    logger.info(f"Tensor records loaded successfully, {len(tensor_records)} records")

    if cmd_args.tensor_planner_type in ["FixedBatchSize", "MaxLengthRestricted"]:
        tensor_plan = tensor_planner.plan_tensor_records(tensor_records, cmd_args.enable_smart_batching)
    elif cmd_args.tensor_planner_type == "GPUMemoryRestricted":
        tensor_plan = tensor_planner.plan_tensor_records(tensor_records, cmd_args.plan_order_type)
    else: # use smart batching by default
        tensor_plan = tensor_planner.plan_tensor_records(tensor_records)

    logger.debug(tensor_plan.formatted_string_of_whole_plan())
    logger.info(f"Tensor planned successfully, {len(tensor_plan)} plan items")
    logger.info(f"Tensor plan stats: {tensor_plan.get_plan_items_stats()}")
    torch.save(tensor_plan, cmd_args.output_path)
    logger.info(f"Tensor plan saved to {cmd_args.output_path}")

if __name__ == '__main__':
    cmd_args = parse_tensor_planning_args()
    plan_tensor(cmd_args)