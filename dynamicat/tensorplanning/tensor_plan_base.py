
class GeneralTensorPlanItem:
    def __init__(self, tensor_records=None):
        if tensor_records:
            self.tensor_records = tensor_records
        else:
            self.tensor_records = []

    def add_tensor_record(self, tensor_record):
        raise NotImplementedError

    def __len__(self):
        return len(self.tensor_records)

    def __str__(self):
        return f"{__class__.__name__}({len(self)})"

    def __repr__(self):
        return self.__str__()

    def iterate_tensor_records(self):
        for rec in self.tensor_records:
            yield rec



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

    def get_plan_items_stats(self):
        return {
            "plan_item_count": len(self)
        }

    def iterate_plan_items(self):
        for plan_item in self.tensor_plan_items:
            yield plan_item

    def __str__(self):
        return f"{__class__.__name__}({self.get_plan_items_stats()})"

    def __repr__(self):
        return self.__str__()

    def formatted_string_of_whole_plan(self):
        s = self.__str__()
        for plan_item in self.iterate_plan_items():
            s += " "*2 + plan_item.__str__() + "\n"
            for idx, rec in enumerate(plan_item.iterate_tensor_records()):
                s += str(idx) + " "*3 + "\n    ".join([f"{field}, size={rec[field].numel()}, preview={rec[field][...,:5]} ... {rec[field][...,-5:]}" for field in rec]) + "\n"
        return s




class GPUMemoryRestrictedTensorPlanItem(GeneralTensorPlanItem):
    def __init__(self, tensor_records=None):
        super().__init__(tensor_records)
        self.tensor_record_count = len(self.tensor_records)
        self.max_of_tensor_parameter_count = 0
        if self.tensor_record_count > 0:
            self.max_of_tensor_parameter_count = max([self.count_tensor_record_parameters(rec) for rec in self.tensor_records])

    @staticmethod
    def count_tensor_record_parameters(tensor_record):
        return sum([_.numel() for _ in tensor_record.values()])

    def add_tensor_record_if_possible(self, tensor_record, tensor_parameter_count_limit):
        # if add, return True, else False
        tensor_count_if_add = self.tensor_record_count + 1
        tensor_record_parameter_count = self.count_tensor_record_parameters(tensor_record)
        max_of_tensor_parameter_count_if_add = max(self.max_of_tensor_parameter_count, tensor_record_parameter_count)
        if tensor_count_if_add * max_of_tensor_parameter_count_if_add > tensor_parameter_count_limit:
            return False
        else:
            self.tensor_records.append(tensor_record)
            self.tensor_record_count = tensor_count_if_add
            self.max_of_tensor_parameter_count = max_of_tensor_parameter_count_if_add
            return True

    def get_tensor_parameter_count_after_padding(self):
        return self.tensor_record_count * self.max_of_tensor_parameter_count

    def __len__(self):
        return self.get_tensor_parameter_count_after_padding()

    def __str__(self):
        return f"{__class__.__name__}({len(self)})"


class GPUCountRestrictedTensorPlan(GeneralTensorPlan):

    def __init__(self, tensor_plan_items=None):
        super().__init__(tensor_plan_items)

    def get_plan_items_stats(self):
        max_of_plan_item_total_tensor_parameter_count_after_padding = 0
        max_of_plan_item_tensor_record_count = 0
        max_of_plan_item_max_of_tensor_parameter_count = 0

        for plan_item in self.tensor_plan_items:
            max_of_plan_item_total_tensor_parameter_count_after_padding = max(max_of_plan_item_total_tensor_parameter_count_after_padding, plan_item.get_tensor_parameter_count_after_padding())
            max_of_plan_item_tensor_record_count = max(max_of_plan_item_tensor_record_count, plan_item.tensor_record_count)
            max_of_plan_item_max_of_tensor_parameter_count = max(max_of_plan_item_max_of_tensor_parameter_count, plan_item.max_of_tensor_parameter_count)

        return {
            "plan_item_count": len(self),
            "max_of_plan_item_total_tensor_parameter_count_after_padding": max_of_plan_item_total_tensor_parameter_count_after_padding,
            "max_of_plan_item_tensor_record_count": max_of_plan_item_tensor_record_count,
            "max_of_plan_item_max_of_tensor_parameter_count": max_of_plan_item_max_of_tensor_parameter_count
        }

    def __str__(self):
        return f"{__class__.__name__}({self.get_plan_items_stats()})"


class KeyFieldLengthDifferenceRestrictedTensorPlanItem(GeneralTensorPlanItem):

    def __init__(self, primary_key, tensor_records=None):
        super().__init__(tensor_records)
        self.tensor_record_count = len(self.tensor_records)
        self.primary_key = primary_key
        self.primary_key_field_min_length = float("inf")
        self.primary_key_field_max_length = 0
        if self.tensor_record_count > 0:
            self.primary_key_field_min_length = min([rec.get(primary_key).numel() for rec in self.tensor_records])
            self.primary_key_field_max_length = max([rec.get(primary_key).numel() for rec in self.tensor_records])



    def add_tensor_record_if_possible(self, tensor_record, max_token_diff, max_plan_item_size):
        # if add, return True, else False
        tensor_count_if_add = self.tensor_record_count + 1
        primary_key_field_length = tensor_record.get(self.primary_key).numel()
        if tensor_count_if_add > max_plan_item_size:
            return False
        if primary_key_field_length - self.primary_key_field_min_length > max_token_diff:
            return False
        self.tensor_records.append(tensor_record)
        self.tensor_record_count += 1
        self.primary_key_field_min_length = min(self.primary_key_field_min_length, primary_key_field_length)
        self.primary_key_field_max_length = max(self.primary_key_field_max_length, primary_key_field_length)
        return True

    def get_length_difference(self):
        return self.primary_key_field_max_length - self.primary_key_field_min_length

    def __str__(self):
        return f"{__class__.__name__}({len(self)})"



class KeyFieldLengthDifferenceRestrictedTensorPlan(GeneralTensorPlan):

    def __init__(self, tensor_plan_items=None):
        super().__init__(tensor_plan_items)

    def get_plan_items_stats(self):
        max_of_plan_item_length_difference = 0
        min_of_plan_item_length_difference = float("inf")

        for plan_item in self.tensor_plan_items:
            max_of_plan_item_length_difference = max(max_of_plan_item_length_difference, plan_item.get_length_difference())
            min_of_plan_item_length_difference = min(min_of_plan_item_length_difference, plan_item.get_length_difference())

        return {
            "plan_item_count": len(self),
            "max_of_plan_item_length_difference": max_of_plan_item_length_difference,
            "min_of_plan_item_length_difference": min_of_plan_item_length_difference
        }

    def __str__(self):
        return f"{__class__.__name__}({self.get_plan_items_stats()})"
