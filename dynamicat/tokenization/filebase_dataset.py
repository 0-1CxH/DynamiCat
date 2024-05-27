import json

from dynamicat.tokenization.tokenizer_base import GeneralDatasetMetadata, GeneralDataset
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
        return f"{super().__str__()}, {self.file_format=}"


class FileBaseDataset(GeneralDataset):

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
                    self.data_store.append(f.read())
                else:
                    raise NotImplementedError

    def iterate(self):
        for data in self.data_store:
            yield data

    def __str__(self):
        return f"{__class__.__name__}(metadata={self.metadata=})"

    def __len__(self):
        return len(self.data_store)




