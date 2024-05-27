import json
import os.path

from loguru import logger

from dynamicat.tokenization.tokenizer_base import GeneralDatasetMetadata, GeneralDatasetBase
from dynamicat.utils import traverse_files_with_suffix


class FileBaseDatasetMetadata(GeneralDatasetMetadata):

    supported_file_formats = [
        "json",
        "jsonl",
        "txt",
    ]

    def __init__(self, dataset_metadata: dict):
        super().__init__(dataset_metadata)
        self.file_format = dataset_metadata.get("file_format")
        assert self.file_format in self.supported_file_formats, f"file format not supported, use one of {self.supported_file_formats}"
        self.folder_path = dataset_metadata.get("folder_path")
        assert self.folder_path, "folder_path is required"


    def __str__(self):
        return (f"{__class__.__name__}({self.dataset_name=}<format={self.file_format}, folder_path={self.folder_path}>)" +
                ", ".join([str(_) for _ in self.iterate_fields()]))


class FileBaseDataset(GeneralDatasetBase):

    def __init__(self, metadata: FileBaseDatasetMetadata):
        super().__init__(metadata)
        self.metadata = metadata
        self.data_store = []

    def load(self):
        for file_path in traverse_files_with_suffix(self.metadata.folder_path, self.metadata.file_format):
            with open(file_path, "r") as f:
                if self.metadata.file_format == "jsonl": # each line is a json object
                    for line in f:
                        self.data_store.append(json.loads(line))
                elif self.metadata.file_format == "json": # single json list object, split as json objects
                    json_list_obj = json.load(f)
                    for json_obj in json_list_obj:
                        self.data_store.append(json_obj)
                elif self.metadata.file_format == "txt": # one file as a record
                    self.data_store.append({"text": f.read()})
                else:
                    raise NotImplementedError

    def iterate(self):
        for data in self.data_store:
            yield data

    def __str__(self):
        return f"{__class__.__name__}(metadata={self.metadata=})"

    def __len__(self):
        return len(self.data_store)

    @staticmethod
    def _single_record_to_tensor_static_wrapper(input_item):
        text_to_tensor_func, record, tokenize_configs_iter = input_item
        current_record_tensor = {}
        used_token_length = 0
        for field, tokenization_configs in tokenize_configs_iter:
            max_seq_len = tokenization_configs.get("max_sequence_length")
            field_max_length = tokenization_configs.get("field_max_length")
            if not max_seq_len:
                field_max_length = None
            else:
                if not field_max_length:
                    field_max_length = max(0, max_seq_len - used_token_length)
                else:
                    field_max_length = min(field_max_length, max_seq_len - used_token_length)
            tokenization_configs["field_max_length"] = field_max_length
            current_record_tensor[field] = text_to_tensor_func(record[field], **tokenization_configs)
            used_token_length += current_record_tensor[field].shape[1]
            logger.debug(f"{field}, {tokenization_configs}, {used_token_length=}")
        return current_record_tensor

    def tokenize_dataset_and_save_pt_file(self, text_to_tensor_func, save_path=None, use_mproc=True):
        if not save_path:
            dataset_folder_path = self.metadata.folder_path
            save_path = os.path.join(
                os.path.dirname(dataset_folder_path),
                f"{os.path.basename(dataset_folder_path)}.pt"
            )

        return super().tokenize_dataset_and_save_pt_file(text_to_tensor_func, save_path, use_mproc)




