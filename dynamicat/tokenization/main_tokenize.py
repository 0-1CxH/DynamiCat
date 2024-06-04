import os
import sys
SOURCE_ROOT_ABS_PATH = os.path.abspath(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir
        )
    )
sys.path.append(SOURCE_ROOT_ABS_PATH)

import argparse
from loguru import logger
from dynamicat.tokenization.filebase_dataset import FileBaseDatasetMetadata, FileBaseDataset
from dynamicat.tokenization.hf_tokenzier import GeneralDatasetHfTokenizer
from dynamicat.tokenization.task_specific_filebase_dataset import DefaultTaskSpecificFileBaseDatasetMetadataFactory


def parse_tokenization_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_metadata_path", type=str, help="Path to dataset metadata file")
    parser.add_argument("--dataset_folder_path", type=str, help="Path to dataset folder")
    parser.add_argument("--max_sequence_lengths", type=int, help="Max sequence lengths")
    parser.add_argument("--dataset_specific_task_type", type=str, default="sft", help="Dataset specific task type")
    parser.add_argument("--dataset_file_format", type=str, help="File format of dataset")
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer")
    return parser.parse_args()

def tokenize(cmd_args):
    # Load tokenizer
    tokenizer = GeneralDatasetHfTokenizer(cmd_args.tokenizer_path)
    tokenizer.load_tokenizer()

    # Load dataset
    if cmd_args.dataset_metadata_path:
        dataset_metadata =  FileBaseDatasetMetadata.load_from_file(cmd_args.dataset_metadata_path)
    else:
        assert cmd_args.dataset_folder_path, "dataset_folder_path is required"
        if cmd_args.dataset_specific_task_type == "pt":
            dataset_metadata = DefaultTaskSpecificFileBaseDatasetMetadataFactory.make_pretrain_metadata(
                pretrain_dataset_folder_path=cmd_args.dataset_folder_path,
                pretrain_max_sequence_lengths=cmd_args.max_sequence_lengths,
                pretrain_file_format=cmd_args.dataset_file_format
            )
        else: # default to sft
            dataset_metadata = DefaultTaskSpecificFileBaseDatasetMetadataFactory.make_sft_metadata(
                sft_dataset_folder_path=cmd_args.dataset_folder_path,
                sft_max_sequence_lengths=cmd_args.max_sequence_lengths,
                sft_file_format=cmd_args.dataset_file_format
            )
    dataset = FileBaseDataset(dataset_metadata)
    dataset.load()
    logger.info(f"Dataset {dataset_metadata} loaded successfully, {len(dataset)} records")
    dataset.tokenize_dataset_and_save_pt_file(tokenizer.text_to_tensor)


if __name__ == '__main__':
    cmd_args = parse_tokenization_args()
    tokenize(cmd_args)