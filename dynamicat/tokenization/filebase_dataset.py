from dynamicat.tokenization.tokenizer_base import GeneralDatasetMetadata


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


    def load(self, file_path):
        if self.file_format == "json":
            return self.load_json(file_path)

