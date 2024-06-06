import pandas as pd
import json

sft_prompt_template = "<|im_start|>user\n{prompt}<|im_end|>\n"
sft_chosen_template = "<|im_start|>assistant\n{chosen}<|im_end|>\n"
sft_role_content_template = "<|im_start|>{role}\n{content}<|im_end|>\n"
def process_ultrachat(data_path, save_path):
    df = pd.read_parquet(data_path)

    f_out = open(save_path, 'w')

    for conversation in df['messages']:
        sft_data_current_conversation = []
        for message in conversation:
            sft_data_current_conversation.append(
                sft_role_content_template.format(role=message['role'], content=message['content'])
            )
        for i in range(1, len(sft_data_current_conversation), 2):
            f_out.write(
                json.dumps({
                    "prompt": "".join(sft_data_current_conversation[:i]),
                    "chosen": "".join(sft_data_current_conversation[i])
                }, ensure_ascii=False) + "\n"
            )

    f_out.close()

if __name__ == '__main__':
    import sys
    process_ultrachat(sys.argv[1], sys.argv[2])

