import os

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
    from test.tokenization.test_datasets import d
    d.load()
    # for data in d.iterate():
    #     print(data)
    print(len(d))
    for _ in d.make_tokenization_configs():
        print(_)
    # print(list(d.dataset_to_tensors(t.text_to_tensor, use_mproc=True)))
    print(d.tokenize_dataset_and_save_pt_file(t.text_to_tensor))