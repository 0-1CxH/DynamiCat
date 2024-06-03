import torch

from dynamicat.tensorplanning.tensor_plan_base import GeneralTensorPlanItem


class GeneralDataCollator:

    DEFAULT_LABEL_MASK_ID = -100

    def __init__(self, fields_sequence, loss_masked_fields, with_label, padding_side="right"):
        self.fields_sequence = fields_sequence
        self.loss_masked_fields = loss_masked_fields
        self.with_label = with_label
        self.padding_side = padding_side

    def list_format_input_collate(self, tensor_plan_item_list):
        assert len(tensor_plan_item_list) == 1
        return self.collate(tensor_plan_item_list[0])

    def collate(self, tensor_plan_item: GeneralTensorPlanItem):

        all_input_ids = []
        all_attention_mask = []
        if self.with_label:
            all_labels = []


        for tensor_record in tensor_plan_item.iterate_tensor_records():
            input_ids_to_concat = []
            attention_mask_to_concat = []
            if self.with_label:
                labels_to_concat = []
            for field in self.fields_sequence:
                input_ids_to_concat.append(tensor_record[field])
                attention_mask_to_concat.append(torch.ones_like(tensor_record[field], dtype=torch.bool))
                if self.with_label:
                    if field not in self.loss_masked_fields: # if field is not in loss_masked_fields, then it is not masked
                        labels_to_concat.append(tensor_record[field]) # so we use the original field as label
                    else: # if field is in loss_masked_fields, then it is masked
                        labels_to_concat.append(torch.full_like(tensor_record[field], self.DEFAULT_LABEL_MASK_ID)) # so we use the default label padding id
            all_input_ids.append(torch.cat(input_ids_to_concat, dim=-1))
            all_attention_mask.append(torch.cat(attention_mask_to_concat, dim=-1))
            if self.with_label:
                all_labels.append(torch.cat(labels_to_concat, dim=-1))

        max_seq_len = 0

        for input_ids in all_input_ids:
            max_seq_len = max(max_seq_len, input_ids.shape[-1])

        padded_input_ids = torch.full((len(all_input_ids), max_seq_len), 0)
        padded_attention_mask = torch.zeros((len(all_input_ids), max_seq_len), dtype=torch.bool)
        if self.with_label:
            padded_labels = torch.full((len(all_input_ids), max_seq_len), self.DEFAULT_LABEL_MASK_ID)

        for i in range(len(all_input_ids)):
            if self.padding_side == "right":
                padded_input_ids[i, :all_input_ids[i].shape[-1]] = all_input_ids[i]
                padded_attention_mask[i, :all_attention_mask[i].shape[-1]] = all_attention_mask[i]
                if self.with_label:
                    padded_labels[i, :all_labels[i].shape[-1]] = all_labels[i]
            else:
                padded_input_ids[i, -all_input_ids[i].shape[-1]:] = all_input_ids[i]
                padded_attention_mask[i, -all_attention_mask[i].shape[-1]:] = all_attention_mask[i]
                if self.with_label:
                    padded_labels[i, -all_labels[i].shape[-1]:] = all_labels[i]

        ret = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask
        }
        if self.with_label:
            ret["labels"] = padded_labels

        return ret

    def __call__(self, tensor_plan_item: GeneralTensorPlanItem):
        return self.collate(tensor_plan_item)

    def __str__(self):
        return f"{__class__.__name__}(fields_sequence={self.fields_sequence}, masked_fields={self.loss_masked_fields}, padding_side={self.padding_side})"

    def __repr__(self):
        return self.__str__()


