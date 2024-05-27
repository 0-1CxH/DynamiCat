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