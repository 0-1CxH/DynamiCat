from calflops import calculate_flops
from transformers import AutoModel
from transformers import AutoTokenizer


model_name = "../test/test_model"
tokenizer_path = "../test/test_tokenizer"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

for batch_size, max_seq_length in [(32,32), (16, 16), (8,8)]:
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=(batch_size,max_seq_length),
                                          transformer_tokenizer=tokenizer,

                                        include_backPropagation=True,
                                          output_as_string=False)
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    print(flops/10**12, macs/10**12, params)
    print(flops/(batch_size*max_seq_length*10**12) * 75 * 75 / 5)


# need * 4/3