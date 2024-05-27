
class GeneralTensorPlanItem:
    def __init__(self, tensor_records=None):
        if tensor_records:
            self.tensor_records = tensor_records
        else:
            self.tensor_records = []

    def add_tensor_record(self, tensor_record):
        self.tensor_records.append(tensor_record)

    def __len__(self):
        return len(self.tensor_records)


class GeneralTensorPlan:
    def __init__(self, tensor_plan_items=None):
        if tensor_plan_items:
            self.tensor_plan_items = tensor_plan_items
        else:
            self.tensor_plan_items = []

    def add_tensor_plan_item(self, tensor_plan_item):
        self.tensor_plan_items.append(tensor_plan_item)

    def __len__(self):
        return len(self.tensor_plan_items)

    def __iter__(self):
        for tensor_plan_item in self.tensor_plan_items:
            yield tensor_plan_item

    def __getitem__(self, idx):
        return self.tensor_plan_items[idx]
