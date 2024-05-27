import os
import time

from dynamicat.tokenization.hf_tokenzier import GeneralDatasetHfTokenizer


if __name__ == '__main__':
    test_folder_base = os.path.dirname(os.path.dirname(__file__))
    print(test_folder_base)
    t = GeneralDatasetHfTokenizer(os.path.join(test_folder_base, "test_tokenizer"))
    t.load_tokenizer()
    print(t)
    sentence = "<|im_start|>test\nHello, world! This is a test sentence and should be long enough to be truncated.<|im_end|>"
    print(t.text_to_tensor(
        sentence,
        truncation_side="left",
        max_length=8
    ))
    print(t.text_to_tensor(
        sentence,
        truncation_side="right",
        max_length=8
    ))
    print(t.text_to_tensor(
        sentence,
        max_length=100
    ))

    # test dataset to tensor
    from test.tokenization.test_datasets import d, d2
    for _ in [d2]:
        print(_)
        _.load()
        print(len(_))
        for data in _.iterate():
            print(data)
        for c in _.make_tokenization_configs():
            print(c)
        # print(list(_.dataset_to_tensors(t.text_to_tensor, use_mproc=True)))
        _.tokenize_dataset_and_save_pt_file(t.text_to_tensor)
        time.sleep(5)