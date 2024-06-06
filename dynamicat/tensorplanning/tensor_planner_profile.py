import math
import torch

from dynamicat.tensorplanning.tensor_plan_base import GeneralTensorPlan, GeneralTensorPlanItem



class TensorPlannerForProfiling:
    FIELD_NAME = "profile_content"
    def __init__(
            self,
            tensor_plan_param_count_start,
            tensor_plan_param_count_end,
            tensor_plan_param_count_step = 1000,
            min_sequence_length = 20,
            max_sequence_length = 5120,
    ):

        self.tensor_plan_param_count_start = tensor_plan_param_count_start
        self.tensor_plan_param_count_end = tensor_plan_param_count_end


        self.param_count_probing_range = range(
            tensor_plan_param_count_start,
            tensor_plan_param_count_end,
            tensor_plan_param_count_step
        )

        # 20, 80, 320, 1280, 5120
        self.sequence_length_probing_range = [
            20 * 2 ** i for i in range(
                int(math.log2(min_sequence_length)),
                int(math.log2(max_sequence_length)) + 1
            )
        ]

        self.tensor_plan = None

    def get_tensor_plan(self):
        tensor_plan = GeneralTensorPlan()
        for current_param_count in self.param_count_probing_range:
            for current_sequence_length in self.sequence_length_probing_range:
                tensor_plan_item = GeneralTensorPlanItem()
                current_batch_size = math.ceil(current_param_count // current_sequence_length) + 1
                if current_batch_size == 0:
                    continue
                # get current_batch_size random tensors with the current_sequence_length and add them to the tensor_plan_item
                for _ in range(current_batch_size):
                    tensor_record = {
                        self.FIELD_NAME: torch.randint(0, 100, (current_sequence_length,))
                    }
                    tensor_plan_item.add_tensor_record_if_possible(tensor_record)
                tensor_plan_item.param_count = current_batch_size * current_sequence_length
                tensor_plan.add_tensor_plan_item(tensor_plan_item)
        # delete param count out of range
        tensor_plan.tensor_plan_items = [item for item in tensor_plan.tensor_plan_items if self.tensor_plan_param_count_start <= item.param_count <= self.tensor_plan_param_count_end]
        tensor_plan.tensor_plan_items.sort(key=lambda x: x.param_count)
        self.tensor_plan = tensor_plan
        return tensor_plan

    def pretty_format_plan(self):
        if not self.tensor_plan:
            return None
        s = ""
        # print 8 each line for better readability
        count = 0
        for tensor_plan_item in self.tensor_plan:
            s += f"{tensor_plan_item.param_count:6d}: {len(tensor_plan_item.tensor_records):4d} * {tensor_plan_item.tensor_records[0][self.FIELD_NAME].shape[0]:4d} "
            count += 1
            if count % 8 == 0:
                s += "\n"
        return s

