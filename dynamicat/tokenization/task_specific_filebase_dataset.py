import os

from dynamicat.tokenization.filebase_dataset import FileBaseDatasetMetadata


class DefaultTaskSpecificFileBaseDatasetMetadataFactory:

    @staticmethod
    def make_sft_metadata(
            sft_dataset_folder_path: str,
            sft_dataset_name = None,
            sft_field_names=None,
            sft_max_sequence_lengths=4096,
            sft_field_max_lengths=None,
            sft_field_truncation_sides=None,
            sft_file_format="jsonl"
    ):
        if sft_dataset_name is None:
            sft_dataset_name = os.path.basename(sft_dataset_folder_path)
        if sft_field_names is None:
            sft_field_names = ["prompt", "chosen"]
        if sft_field_max_lengths is None:
            sft_field_max_lengths = [int(0.75*sft_max_sequence_lengths), None]
        if sft_field_truncation_sides is None:
            sft_field_truncation_sides = ["left", "right"]

        return FileBaseDatasetMetadata({
            "dataset_name": sft_dataset_name,
            "field_names": sft_field_names,
            "max_seq_len": sft_max_sequence_lengths,
            "field_max_lengths": sft_field_max_lengths,
            "field_truncation_sides": sft_field_truncation_sides,
            "file_format": sft_file_format,
            "folder_path": sft_dataset_folder_path
        })


    @staticmethod
    def make_pretrain_metadata(
            pretrain_dataset_folder_path: str,
            pretrain_dataset_name = None,
            pretrain_field_names=None,
            pretrain_max_sequence_lengths=None,
            pretrain_field_max_lengths=None,
            pretrain_field_truncation_sides=None,
            pretrain_file_format="txt"
    ):
        if pretrain_dataset_name is None:
            pretrain_dataset_name = os.path.basename(pretrain_dataset_folder_path)
        if pretrain_field_names is None:
            pretrain_field_names = ["content"]
        if pretrain_field_max_lengths is None:
            pretrain_field_max_lengths = [None]

        return FileBaseDatasetMetadata({
            "dataset_name": pretrain_dataset_name,
            "field_names": pretrain_field_names,
            "max_seq_len": pretrain_max_sequence_lengths,
            "field_max_lengths": pretrain_field_max_lengths,
            "field_truncation_sides": pretrain_field_truncation_sides,
            "file_format": pretrain_file_format,
            "folder_path": pretrain_dataset_folder_path
        })

    @staticmethod
    def make_inference_metadata(
            inference_dataset_folder_path: str,
            inference_dataset_name = None,
            inference_field_names=None,
            inference_max_sequence_lengths=4096,
            inference_field_max_lengths=None,
            inference_field_truncation_sides=None,
            inference_file_format="jsonl"
    ):
        if inference_dataset_name is None:
            inference_dataset_name = os.path.basename(inference_dataset_folder_path)
        if inference_field_names is None:
            inference_field_names = ["question"]
        if inference_field_max_lengths is None:
            inference_field_max_lengths = [inference_max_sequence_lengths - 100]
        if inference_field_truncation_sides is None:
            inference_field_truncation_sides = ["left"]

        return FileBaseDatasetMetadata({
            "dataset_name": inference_dataset_name,
            "field_names": inference_field_names,
            "max_seq_len": inference_max_sequence_lengths,
            "field_max_lengths": inference_field_max_lengths,
            "field_truncation_sides": inference_field_truncation_sides,
            "file_format": inference_file_format,
            "folder_path": inference_dataset_folder_path
        })