import torch
import os
from dynamicat.tensorplanning.tensor_planner import TensorPlannerFactory

if __name__ == '__main__':
    test_folder_base = os.path.dirname(os.path.dirname(__file__))
    print(test_folder_base)

    tensor_records_jsonl = torch.load(os.path.join(test_folder_base, "test_jsonl_data.pt"))
    tensor_records_txt = torch.load(os.path.join(test_folder_base, "test_txt_data.pt"))

    tp1 = TensorPlannerFactory.create_tensor_planner(plan_type="FixedBatchSize", batch_size=4)
    tp2 = TensorPlannerFactory.create_tensor_planner(plan_type="GPUMemoryRestricted", tensor_parameter_count_limit=15000)
    tp3 = TensorPlannerFactory.create_tensor_planner(plan_type="LengthDifferenceRestricted", max_token_diff=16, max_plan_size=64, primary_key="prompt")
    tp4 = TensorPlannerFactory.create_tensor_planner(plan_type="MaxLengthRestricted", max_field_length=1024, primary_key="content")

    plan1 = tp1.plan_tensor_records(tensor_records_jsonl)
    plan2 = tp2.plan_tensor_records(tensor_records_jsonl)
    plan3 = tp3.plan_tensor_records(tensor_records_jsonl)
    plan4 = tp4.plan_tensor_records(tensor_records_txt)

    for _ in [plan1, plan2, plan3, plan4]:
        print(_.formatted_string_of_whole_plan())

