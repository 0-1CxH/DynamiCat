import os
from transformers import LlamaForCausalLM
from dynamicat.model.hf_model import HFModelProvider
from test.tokenization.test_tokenize import t

test_folder_base = os.path.dirname(os.path.dirname(__file__))
print(test_folder_base)

model_path = os.path.join(test_folder_base, "test_model")
model_save_path = os.path.join(test_folder_base, "test_model_save")

if __name__ == '__main__':
    model = HFModelProvider.load(LlamaForCausalLM, model_path)
    model = HFModelProvider.set_model_special_tokens(model, t)
    HFModelProvider.save(model, model_save_path)
    t.save_tokenizer(model_save_path)

