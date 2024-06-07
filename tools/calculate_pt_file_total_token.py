import torch


def calculate_pt_file_total_token(pt_file):
    tokenized_pt_file_content = torch.load(pt_file)
    total_token = 0
    for tensor_record in tokenized_pt_file_content:
        for field in tensor_record:
            total_token += tensor_record[field].shape[-1]
    return total_token

if __name__ == '__main__':
    import sys
    pt_file = sys.argv[1]
    token_count = calculate_pt_file_total_token(pt_file)
    print(f"Total token count in {pt_file}: {token_count}=({token_count/1e9:.2f} billion)=({token_count/1e6:.2f} million)=({token_count/1e3:.2f} thousand)")