from abc import abstractmethod

import torch
from loguru import logger

from dynamicat.tensorplanning.tensor_plan_base import (
    GeneralTensorPlan, GeneralTensorPlanItem,
    GPUMemoryRestrictedTensorPlanItem, GPUCountRestrictedTensorPlan,
    KeyFieldLengthDifferenceRestrictedTensorPlan, KeyFieldLengthDifferenceRestrictedTensorPlanItem
)


class TensorPlannerBase:
    def __init__(self, plan_type):
        self.plan_type = plan_type

    def __str__(self):
        return f"{__class__.__name__}(plan_type={self.plan_type})"

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def plan_tensor_records(self, tensor_records):
        # return tensor plan base on plan type
        raise NotImplementedError

    def plan_tensor_records_and_save(self, tensor_records, save_path):
        tensor_plan = self.plan_tensor_records(tensor_records)
        torch.save(tensor_plan, save_path)
        return tensor_plan


class SmartBatchingMixIn:

    SINGLE_TENSOR_LENGTH = lambda x: x.shape[-1]
    TENSORS_DICT_TOTAL_LENGTH = lambda x: sum([v.shape[-1] for v in x.values()])

    @staticmethod
    def single_tensor_length_by_field(field):
        return lambda x: x[field].shape[-1]

    @staticmethod
    def exec_smart_batching(tensor_records, tensor_numeration_func=TENSORS_DICT_TOTAL_LENGTH):
        # tensor_numeration_func: (tensor) -> int
        tensor_records_with_numeration = [(tensor_record, tensor_numeration_func(tensor_record)) for tensor_record in tensor_records]
        # tensor_records_with_numeration: [(tensor, number_order_key), ...]
        tensor_records_with_numeration.sort(key=lambda x: x[1])
        return [tensor_record for tensor_record, _ in tensor_records_with_numeration]



class FixedBatchSizeTensorPlanner(TensorPlannerBase, SmartBatchingMixIn):
    def __init__(self, batch_size, **kwargs):
        super().__init__("FixedBatchSize")
        self.batch_size = batch_size

    def __str__(self):
        return f"{__class__.__name__}(batch_size={self.batch_size})"

    def plan_tensor_records(self, tensor_records, enable_smart_batching=True):
        tensor_plan = GeneralTensorPlan()

        if enable_smart_batching:
            tensor_records = self.exec_smart_batching(tensor_records)

        for batch_start_idx in range(0, len(tensor_records), self.batch_size):
            tensor_plan_item = GeneralTensorPlanItem(tensor_records[batch_start_idx: batch_start_idx + self.batch_size])
            tensor_plan.add_tensor_plan_item(tensor_plan_item)
        return tensor_plan


class GPUMemoryRestrictedTensorPlanner(TensorPlannerBase, SmartBatchingMixIn):
    def __init__(self, tensor_parameter_count_limit, **kwargs):
        super().__init__("GPUMemoryRestricted")
        self.tensor_parameter_count_limit = tensor_parameter_count_limit

    def __str__(self):
        return f"{__class__.__name__}(tensor_parameter_count_limit={self.tensor_parameter_count_limit})"

    def plan_tensor_records(self, tensor_records):
        tensor_plan = GPUCountRestrictedTensorPlan()
        tensor_records = self.exec_smart_batching(tensor_records)
        current_tensor_plan_item = GPUMemoryRestrictedTensorPlanItem()
        for tensor_record in tensor_records:
            if not current_tensor_plan_item.add_tensor_record_if_possible(tensor_record, self.tensor_parameter_count_limit):
                tensor_plan.add_tensor_plan_item(current_tensor_plan_item)
                current_tensor_plan_item = GPUMemoryRestrictedTensorPlanItem([tensor_record])
        if current_tensor_plan_item:
            tensor_plan.add_tensor_plan_item(current_tensor_plan_item)
        logger.info(tensor_plan.get_plan_items_stats())
        return tensor_plan


class KeyFieldLengthDifferenceRestrictedTensorPlanner(TensorPlannerBase, SmartBatchingMixIn):
    def __init__(self, max_token_diff, max_plan_size, primary_key, **kwargs):
        super().__init__("LengthDifferenceRestricted")
        self.max_token_diff = max_token_diff
        self.max_plan_size = max_plan_size
        self.primary_key = primary_key


    def __str__(self):
        return f"{__class__.__name__}(max_token_diff={self.max_token_diff}, max_plan_size={self.max_plan_size}, primary_key={self.primary_key})"


    def plan_tensor_records(self, tensor_records):
        tensor_plan = KeyFieldLengthDifferenceRestrictedTensorPlan()
        tensor_records = self.exec_smart_batching(tensor_records, self.single_tensor_length_by_field(self.primary_key))

        current_tensor_plan_item = KeyFieldLengthDifferenceRestrictedTensorPlanItem(self.primary_key)
        for tensor_record in tensor_records:
            if not current_tensor_plan_item.add_tensor_record_if_possible(tensor_record, self.max_token_diff, self.max_plan_size):
                tensor_plan.add_tensor_plan_item(current_tensor_plan_item)
                current_tensor_plan_item = KeyFieldLengthDifferenceRestrictedTensorPlanItem(self.primary_key, [tensor_record])
        if current_tensor_plan_item:
            tensor_plan.add_tensor_plan_item(current_tensor_plan_item)
        logger.info(tensor_plan.get_plan_items_stats())
        return tensor_plan


class KeyFieldMaxLengthRestrictedTensorPlanner(TensorPlannerBase, SmartBatchingMixIn):
    def __init__(self, max_field_length, batch_size, primary_key, **kwargs):
        super().__init__("MaxLengthRestricted")
        self.max_field_length = max_field_length
        self.batch_size = batch_size
        self.primary_key = primary_key

    def __str__(self):
        return f"{__class__.__name__}(max_field_length={self.max_field_length}, primary_key={self.primary_key})"

    def plan_tensor_records(self, tensor_records, enable_smart_batching=False):
        tensor_plan = GeneralTensorPlan()
        if enable_smart_batching:
            tensor_records = self.exec_smart_batching(tensor_records, self.single_tensor_length_by_field(self.primary_key))

        current_tensor_plan_item = GeneralTensorPlanItem()
        for tensor_record in tensor_records:
            if tensor_record.get(self.primary_key).numel() <= self.max_field_length:
                tensor_records_to_add = [tensor_record]
            else: # split the tensor
                key_field_tensor = tensor_record.get(self.primary_key)
                tensor_records_to_add = [
                    {self.primary_key: key_field_tensor[..., i: i + self.max_field_length]}
                    for i in range(0, key_field_tensor.shape[-1], self.max_field_length)
                ]
            for tensor_record_to_add in tensor_records_to_add:
                if len(current_tensor_plan_item) >= self.batch_size:
                    tensor_plan.add_tensor_plan_item(current_tensor_plan_item)
                    current_tensor_plan_item = GeneralTensorPlanItem()
                current_tensor_plan_item.add_tensor_record_if_possible(tensor_record_to_add)
        if current_tensor_plan_item:
            tensor_plan.add_tensor_plan_item(current_tensor_plan_item)
        return tensor_plan



class TensorPlannerFactory:

    PlanTypeToClzMap = {
        "FixedBatchSize": FixedBatchSizeTensorPlanner,
        "GPUMemoryRestricted": GPUMemoryRestrictedTensorPlanner,
        "LengthDifferenceRestricted": KeyFieldLengthDifferenceRestrictedTensorPlanner,
        "MaxLengthRestricted": KeyFieldMaxLengthRestrictedTensorPlanner
    }
    @staticmethod
    def create_tensor_planner(plan_type, **kwargs):
        clz = TensorPlannerFactory.PlanTypeToClzMap.get(plan_type)
        if not clz:
            raise ValueError(f"{plan_type=} is not yet supported")
        instance = clz(**kwargs)
        logger.info(f"Using TensorPlanner of {plan_type=}, {instance=}")
        return instance
