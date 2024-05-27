from dynamicat.tokenization.filebase_dataset import FileBaseDataset, FileBaseDatasetMetadata

import os

if __name__ == '__main__':
    test_folder_base = os.path.dirname(os.path.dirname(__file__))
    print(test_folder_base)
    d = FileBaseDataset(
        FileBaseDatasetMetadata({
            "dataset_name": "test_json_data",
            "field_names": ["prompt", "chosen"],
            "field_max_lengths": [512, 128],
            "field_padding_sides": ["left", "right"],
            "field_truncation_sides": ["left", "right"],
            "file_format": "jsonl",
            "folder_path": os.path.join(test_folder_base, "test_jsonl_data")
        })
    )
    d.load()
    for data in d.iterate():
        print(data)
    for _ in d.make_tokenization_configs():
        print(_)