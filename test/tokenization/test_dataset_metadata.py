from dynamicat.tokenization.filebase_dataset import FileBaseDatasetMetadata
from dynamicat.tokenization.task_specific_filebase_dataset import DefaultTaskSpecificFileBaseDataset
from dynamicat.tokenization.tokenizer_base import GeneralDatasetMetadata
import os

test_folder_base = os.path.dirname(os.path.dirname(__file__))
print(test_folder_base)

m = GeneralDatasetMetadata({
        "dataset_name": "dataset_1",
        "field_names": ["text", "label"],
        "field_max_lengths": [1024, 128],
        "field_truncation_sides": ["left", "right"]
    })

m2 = GeneralDatasetMetadata({
    "dataset_name": "dataset_2",
    "field_names": ["text", "label"],
    "max_seq_len": 1024,
    "field_max_lengths": [128, None],
    "field_truncation_sides": ["left", "right"]
})

m3 = FileBaseDatasetMetadata({
    "dataset_name": "test_json_data",
    "field_names": ["prompt", "chosen"],
    "max_seq_len": 4096,
    "field_max_lengths": [3072, None],
    "field_truncation_sides": ["left", "right"],
    "file_format": "jsonl",
    "folder_path": os.path.join(test_folder_base, "test_jsonl_data")
})

m4 = GeneralDatasetMetadata({
    "dataset_name": "dataset_4",
    "field_names": ["text"],
})

m5 = DefaultTaskSpecificFileBaseDataset.make_sft_metadata(
    os.path.join(test_folder_base, "test_jsonl_data")
    )

m6 = DefaultTaskSpecificFileBaseDataset.make_pretrain_metadata(
    os.path.join(test_folder_base, "test_txt_data")
    )

m7 = DefaultTaskSpecificFileBaseDataset.make_inference_metadata(
    os.path.join(test_folder_base, "test_jsonl_data")
)

if __name__ == '__main__':


    for _ in [m, m2, m3, m4, m5, m6, m7]:
        print(_)
        print(_.get_field_count())