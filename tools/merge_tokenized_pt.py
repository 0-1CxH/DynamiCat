import os

import torch

def merge_pt_files(pt_files_folder, output_file):
    merged_tokenized_pt_file_content = []
    for file in os.listdir(pt_files_folder):
        if file.endswith(".pt"):
            tokenized_pt_file_content = torch.load(os.path.join(pt_files_folder, file))
            print(f"Loaded {len(tokenized_pt_file_content)} records from {file}")
            merged_tokenized_pt_file_content.extend(tokenized_pt_file_content)
    print(f"Merged {len(merged_tokenized_pt_file_content)} records and save to {output_file}")
    torch.save(merged_tokenized_pt_file_content, output_file)

if __name__ == '__main__':
    import sys
    pt_files_folder = sys.argv[1]
    output_file = sys.argv[2]
    merge_pt_files(pt_files_folder, output_file)
