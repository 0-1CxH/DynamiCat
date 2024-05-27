from abc import abstractmethod

import torch

from dynamicat.tensorplanning.tensor_plan_base import GeneralTensorPlan, GeneralTensorPlanItem


class TensorPlannerBase:
    def __init__(self, plan_type):
        self.plan_type = plan_type

    @abstractmethod
    def plan_tensor_records(self, tensor_records):
        # return tensor plan base on plan type
        pass

    def plan_tensor_records_and_save(self, tensor_records, save_path):
        tensor_plan = self.plan_tensor_records(tensor_records)
        torch.save(tensor_plan, save_path)
        return tensor_plan


class SmartBatchingMixIn:

    SINGLE_TENSOR_LENGTH = lambda x: x.shape[-1]
    TENSORS_DICT_TOTAL_LENGTH = lambda x: sum([v.shape[-1] for v in x.values()])

    @staticmethod
    def exec_smart_batching(tensor_records, tensor_numeration_func):
        # tensor_numeration_func: (tensor) -> int
        tensor_records_with_numeration = [(tensor_record, tensor_numeration_func(tensor_record)) for tensor_record in tensor_records]
        # tensor_records_with_numeration: [(tensor, number_order_key), ...]
        tensor_records_with_numeration.sort(key=lambda x: x[1])
        return [tensor_record for tensor_record, _ in tensor_records_with_numeration]



class FixedBatchSizeTensorPlanner(TensorPlannerBase, SmartBatchingMixIn):
    def __init__(self, batch_size):
        super().__init__("FixedBatchSize")
        self.batch_size = batch_size

    def plan_tensor_records(self, tensor_records, enable_smart_batching=True):
        tensor_plan = GeneralTensorPlan()

        if enable_smart_batching:
            tensor_records = self.exec_smart_batching(tensor_records, self.TENSORS_DICT_TOTAL_LENGTH)

        for batch_start_idx in range(0, len(tensor_records), self.batch_size):
            tensor_plan_item = GeneralTensorPlanItem(tensor_records[batch_start_idx: batch_start_idx + self.batch_size])
            tensor_plan.add_tensor_plan_item(tensor_plan_item)
        return tensor_plan