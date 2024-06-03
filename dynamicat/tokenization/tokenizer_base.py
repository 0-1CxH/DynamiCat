import json
from abc import abstractmethod
from dataclasses import dataclass

import torch
from loguru import logger

from dynamicat.utils import mproc_map


@dataclass
class GeneralDatasetMetadata:
    def __init__(self, dataset_metadata: dict):
        self.dataset_name = dataset_metadata.get("dataset_name", "default_dataset")
        self.field_names = dataset_metadata.get("field_names")
        assert self.field_names, "field_names is required"
        self.max_seq_len = dataset_metadata.get("max_seq_len") # None means no truncation
        self.field_max_lengths = dataset_metadata.get("field_max_lengths", [None] * len(self.field_names)) # None represents no truncation
        if None in self.field_max_lengths:
            assert self.max_seq_len is not None or not all(self.field_max_lengths), "if field_max_lengths has default, max_seq_len should not be default"
        if self.max_seq_len:
            assert sum([i for i in self.field_max_lengths if
                        i is not None]) <= self.max_seq_len, "sum of field_max_lengths should be less than max_seq_len"
        assert len(self.field_max_lengths) == len(self.field_names), "field_max_lengths should have the same length as field_names"
        self.field_truncation_sides = dataset_metadata.get("field_truncation_sides", ["right"] * len(self.field_names))
        if not self.field_max_lengths:
            assert len(self.field_truncation_sides) == len(self.field_names), "field_truncation_sides should have the same length as field_names"


    def iterate_fields(self):
        for idx, field_name in enumerate(self.field_names):
            yield {
                "field_name": field_name,
                "max_sequence_length": self.max_seq_len,
                "field_max_length": self.field_max_lengths[idx],
                "field_truncation_side": self.field_truncation_sides[idx] if self.field_truncation_sides else "none"
            }

    def get_field_count(self):
        return len(self.field_names)
    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, "r") as f:
            return cls(json.load(f))

    def __str__(self):
        return f"{__class__.__name__}({self.dataset_name=}), " + ", ".join([str(_) for _ in self.iterate_fields()])


class GeneralDatasetBase:
    def __init__(self, metadata: GeneralDatasetMetadata):
        self.metadata = metadata

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def iterate(self) -> iter:
        # must return iterable of dicts
        raise NotImplementedError

    def make_tokenization_configs(self):
        for field in self.metadata.iterate_fields():
            yield field['field_name'], {
                "truncation_side": field["field_truncation_side"],
                "max_sequence_length": field["max_sequence_length"],
                "field_max_length": field["field_max_length"]
            }

    def __str__(self):
        return f"{__class__.__name__}(metadata={self.metadata=})"

    @staticmethod
    def _single_record_to_tensor_static_wrapper(text_to_tensor_func):
        # return a wrapped function that takes a record and returns a dict of tensors
        raise NotImplementedError


    def dataset_to_tensors(self, text_to_tensor_func, use_mproc=True):
        if use_mproc:
            tokenize_configs_iter = [_ for _ in self.make_tokenization_configs()]
            for _ in mproc_map(
                func=self._single_record_to_tensor_static_wrapper,
                items=[(text_to_tensor_func, record, tokenize_configs_iter) for record in self.iterate()]
            ):
                yield _
        else:
            for record in self.iterate():
                input_item = (text_to_tensor_func, record, self.make_tokenization_configs())
                yield self._single_record_to_tensor_static_wrapper(input_item)

    def tokenize_dataset_and_save_pt_file(self, text_to_tensor_func, save_path, use_mproc=True):
        tensors = list(self.dataset_to_tensors(text_to_tensor_func, use_mproc))
        logger.info(f"saving tensors of length {len(tensors)} to {save_path}")
        torch.save(tensors, save_path)
        return tensors



class GeneralDatasetTokenizer:
    def __init__(self):
        self.tokenizer = self.load_tokenizer()
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"

    @abstractmethod
    def load_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def text_to_tensor(self, text, **tokenization_configs):
        raise NotImplementedError


