from dataclasses import dataclass

@dataclass
class GeneralDatasetMetadata:
    def __init__(self, dataset_metadata: dict):
        self.dataset_name = dataset_metadata.get("dataset_name", "default_dataset")
        self.field_names = dataset_metadata.get("field_names")
        assert self.field_names, "field_names is required"
        self.field_max_lengths = dataset_metadata.get("field_max_lengths", [1024] * len(self.field_names))
        assert len(self.field_max_lengths) == len(self.field_names), "field_max_lengths should have the same length as field_names"
        # self.field_padding_sides = dataset_metadata.get("field_padding_sides", ["right"] * len(self.field_names))
        # assert len(self.field_padding_sides) == len(self.field_names), "field_padding_sides should have the same length as field_names"
        self.field_truncation_sides = dataset_metadata.get("field_truncation_sides", ["right"] * len(self.field_names))
        assert len(self.field_truncation_sides) == len(self.field_names), "field_truncation_sides should have the same length as field_names"


    def iterate_fields(self):
        for idx, field_name in enumerate(self.field_names):
            yield {
                "field_name": field_name,
                "field_max_length": self.field_max_lengths[idx],
                # "field_padding_side": self.field_padding_sides[idx],
                "field_truncation_side": self.field_truncation_sides[idx]
            }

    def get_field_count(self):
        return len(self.field_names)

    def __str__(self):
        return f"{__class__.__name__}({self.dataset_name=}), " + ", ".join([str(_) for _ in self.iterate_fields()])


class GeneralDataset:
    def __init__(self, metadata: GeneralDatasetMetadata):
        self.metadata = metadata

    def load(self):
        raise NotImplementedError

    def iterate(self) -> iter:
        # must return iterable
        raise NotImplementedError

    def make_tokenization_configs(self):
        for field in self.metadata.iterate_fields():
            yield field['field_name'], {
                # "padding_side": field["field_padding_side"],
                "truncation_side": field["field_truncation_side"],
                "max_length": field["field_max_length"]
            }

    def __str__(self):
        return f"{__class__.__name__}(metadata={self.metadata=})"



class GeneralDatasetTokenizer:
    def __init__(self):
        self.tokenizer = self.load_tokenizer()
        self.tokenize_function = self.tokenizer.__call__
        # self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"

    def load_tokenizer(self):
        raise NotImplementedError

    def text_to_tensor(self, text, **tokenization_configs):
        # if "padding_side" in tokenization_configs:
        #     self.tokenizer.padding_side = tokenization_configs["padding_side"]
        if "truncation_side" in tokenization_configs:
            self.tokenizer.truncation_side = tokenization_configs["truncation_side"]
        if "max_length" in tokenization_configs:
            self.tokenizer.max_length = tokenization_configs["max_length"]
        return self.tokenize_function(text)


    def dataset_to_tensors(self, dataset: GeneralDataset):
        for record in dataset.iterate():
            current_record_tensors = {}
            for field, tokenization_configs in dataset.make_tokenization_configs():
                current_record_tensors[field] = self.text_to_tensor(record[field], **tokenization_configs)
            yield current_record_tensors