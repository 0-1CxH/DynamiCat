from dynamicat.tokenization.tokenizer_base import GeneralDatasetMetadata

if __name__ == '__main__':
    m = GeneralDatasetMetadata({
        "dataset_name": "dataset_1",
        "field_names": ["text", "label"],
        "field_max_lengths": [1024, 128],
        "field_padding_sides": ["right", "left"],
        "field_truncation_sides": ["left", "right"]
    })
    print(m)
    print(m.get_field_count())