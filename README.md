
# DynamiCat


## Abstract
**On public dataset ranging from 1M to 500M tokens, DynamiCat provides 20% to 280% improvement in throughput compared to common fixed-batch-size SFT.**


## Methodology

### Dynamic batch size

When doing SFT, there are data of different lengths in one dataset, the common data loader only provides training batch of fixed batch size and the batch size depends on the max length of the data in dataset, basically, the batch size is set to max_token_capacity_of_gpu / max_token_length_in_dataset.

However, there are more data in dataset that is much shorter than the longest one, we can set bigger batch size for the shorter ones, and smaller batch size for the longer ones, this is **dynamic batch size**.

For example: 
```text
GPU token capacity: 10K
dataset token lengths: [1K, 1K, 1K, 2K, 2K, 3K, 5K, 5K]
common fixed-batch-size data loder: 
       batch size = 10K / max([1K, 1K, 1K, 2K, 2K, 3K, 5K, 5K]) = 2
       data batches = [[1K, 1K], [1K, 2K], [2K, 3K], [5K, 5K]]]
       total: 4 
DynamiCat dynamic-batch-size data loader:
        greedy put data in batches until read the GPU token capacity
        data batches = [[1K, 1K, 1K, 2K, 2K, 3K], [5K, 5K]]
        total: 2
```

Note: when the batch size is not fixed, the real learning rate should scale based on the real batch size.

### Dynamic padding

For each batch, data records of different length are padded to the same length in the collation phase by adding attention mask, loss mask to the collated batch data.

Commonly, the padding are set to the max sequence length of the model, however, we can dynamically set the padding length to the max length of the current batch rather than the model length that is much longer, this is **dynamic padding**.

For example:
```text
model max sequence length: 16K
data batches = [[1K, 1K, 1K, 2K, 2K, 3K], [5K, 5K]]
common fixed-length padding data collator:
        collated batches = [[16K, 16K, 16K, 16K, 16K, 16K], [16K, 16K]]
        total: 16K * 5 + 16K * 2 = 112K
DynamiCat dynamic-padding data collator:
        collated batches = [[3K, 3K, 3K, 3K, 3K, 3K], [5K, 5K]]
        total: 3K * 6 + 5K * 2 = 28K
```

### Smart batching

In one word, smart batching means sorting the data records by token length to minimize the overhead of padding, this is a usually used method to optimize dynamic padding. 

For example:
```text
GPU token capacity: 10K
model max sequence length: 16K
Without smart batching:
        dataset token lengths: [5K, 3K, 1K, 5K, 2K, 1K, 2K, 1K]
        data batches = [[5K, 3K, 1K], [5K, 2K, 1K, 2K], [1K]]
        collated batches = [[5K, 5K, 5K], [5K, 5K, 5K, 5K], [1K]]
        total: 5K * 3 + 5K * 4 + 1K + 1 = 36K
With smart batching:
        dataset token lengths: [1K, 1K, 1K, 2K, 2K, 3K, 5K, 5K]
        data batches = [[1K, 1K, 1K, 2K, 2K, 3K], [5K, 5K]]
        collated batches = [[3K, 3K, 3K, 3K, 3K, 3K], [5K, 5K]]
        total: 3K * 6 + 5K * 2 = 28K
```


## Experiment


### Environment
- 8x RTX4090 (24G) GPUs
- Model: Yi-6B
- Deepspeed Zero3 + offload + FlashAttention enabled

### Datasets

| dataset         | unpacked token count | unpacked record count | max token length | link                                                           |
|-----------------|----------------------|-----------------------|------------------|----------------------------------------------------------------|
| gsm8k           | 1.74M                | 8.7K                  | 551              | https://huggingface.co/datasets/openai/gsm8k                    | 
| alpaca cleaned  | 9.28M                | 51.8K                 | 1473             | https://huggingface.co/datasets/yahma/alpaca-cleaned           |
| belle chat 0.4M | 107.27M              | 396.0K                 | 879              | https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M |
| ultra chat 200k | 476.23M              | 612.9K                | 4096             | https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k | 


### Observations

#### Common sft

batch size: max batch sizes that fits into GPU memory

| dataset         | batch size | throughput(TFLOPS, per device) | GPU mem util(%) | time(sec) | final loss (smoothed) |
|-----------------|------------|--------------------------------|-----------------|-----------|-----------------------|
| gsm8k           | 24         | 49.53                          | 88.19           | 363       |                       | 
| alpaca cleaned  | 8          | 25.49                          | 76.88           | 4372      | 0.931                 |
| belle chat 0.4M | 12         | 33.83                          | 77.18           | 23757     | 1.631                 |
| ultra chat 200k | 2          | 16.91                          | 76.64           | 147096    |                       |


#### DynamiCat sft 

batch token capacity: find the max token capacity that a batch holds and maximize the batch size that fits into GPU memory 

| dataset         | batch token capacity | throughput(TFLOPS, per device) | GPU mem util(%) | time(sec) | final loss (smoothed) | 
|-----------------|----------------------|--------------------------------|-----------------|-----------|-----------------------|
| gsm8k           | 13.5K                | 62.55                          | 93.02           | 164       |                       | 
| alpaca cleaned  | 12K                  | 64.43                          | 90.90           | 812       | 0.947                 | 
| belle chat 0.4M | 13K                  | 70.42                          | 98.99           | 8451      | 1.614                 | 
| ultra chat 200k | 11.5K                | 64.20                          | 97.10           | 41106     |                       |

### Comparison & Results
compare DynamiCat to common sft:

| dataset         | throughput(ratio) | GPU memory utilization (percentage points) | training time (ratio) | loss delta | 
|-----------------|-------------------|--------------------------------------------|-----------------------|------------|
| gsm8k           | 1.26x             | +4.83                                      | 0.45x                 |            |                  
| alpaca cleaned  | 2.53x             | +14.02                                     | 0.19x                 | +0.016     | 
| belle chat 0.4M | 2.08x             | +21.81                                     | 0.36x                 | -0.017     |
| ultra chat 200k | 3.80x             | +20.46                                     | 0.28x                 |            |


![compare_alpaca.png](experiment%2Fimg%2Fcompare_alpaca.png)
![belle_compare.png](experiment%2Fimg%2Fbelle_compare.png)
TODO: change image to new tags and add other datasets 

# Quick Start

```bash
bash scripts/tokenize/run_tokenize_default_sft.sh -i JSONL_DATA_PATH -t TOKENIZER_PATH
bash scripts/plan/run_plan_gpu_mem.sh -i TOKENIZED_PT_PATH -c TOKEN_CAPACITY_LIMIT -r [options: bs_asc, bs_desc, pc_asc, pc_desc, random, none]
deepspeed dynamicat/training/training_pipeline.py \
  --global_batch_size 32 \
  --batch_size_per_gpu 4 \
  --planned_tensor_file_path PLANNED_TENSOR_PT_PATH \
  --dataset_specific_task_type sft \
  --model_path MODEL_PATH \
  --zero_stage 3 \
  --use_bf16 \
  --learning_rate 1e-5 \
  --use_tensorboard \
  --num_epochs 3 \
  --checkpoint_save_path MODEL_SAVE_PATH
```


# Test Cases

## Step 1: Tokenization

```bash
bash scripts/tokenize/run_tokenize_default_sft.sh -i  test/test_jsonl_data -t test/test_tokenizer/
bash scripts/tokenize/run_tokenize_default_pt.sh -i test/test_txt_data -t test/test_tokenizer/
bash scripts/tokenize/run_tokenize_by_metadata_file.sh -i test/test_metadata_file.json -t test/test_tokenizer/
```

## Step 2: Profile Training (Optional)

```bash
bash scripts/run_training_profiling.sh test/test_model
```


## Step 3: Tensor Planning

```bash
# Fixed batch size (smart batching enabled/ disabled)
bash scripts/plan/run_plan_fixed_batch_size.sh -i test/test_jsonl_data.pt -b 4 -s 1
bash scripts/plan/run_plan_fixed_batch_size.sh -i test/test_jsonl_data.pt -b 4
# GPU memory restricted
bash scripts/plan/run_plan_gpu_mem.sh -i test/test_jsonl_data.pt -c 1500
bash scripts/plan/run_plan_gpu_mem.sh -i test/test_jsonl_data.pt -c 15000 -r [plan_order_type, options are: bs_asc, bs_desc, pc_asc, pc_desc, random, none]
# Key field length difference restricted
bash scripts/plan/run_plan_length_diff.sh -i test/test_jsonl_data.pt -k prompt -d 10 -b 8
# Key field max length restricted (smart batching enabled/ disabled)
bash scripts/plan/run_plan_max_length.sh -i test/test_txt_data.pt -k content -m 100 -b 8 -s 1
bash scripts/plan/run_plan_max_length.sh -i test/test_txt_data.pt -k content -m 100 -b 8
```


## Step 4: Training

```bash
deepspeed dynamicat/training/training_pipeline.py \
  --global_batch_size 32 \
  --batch_size_per_gpu 4 \
  --planned_tensor_file_path test/test_jsonl_data_GPUMemoryRestricted_15000.pt \
  --dataset_specific_task_type sft \
  --model_path test/test_model \
  --zero_stage 3 \
  --use_bf16 \
  --learning_rate 1e-5 \
  --use_tensorboard \
  --num_epochs 3 \
  --checkpoint_save_path test/test_model_save_ds_train
 
deepspeed dynamicat/training/training_pipeline.py \
  --global_batch_size 32 \
  --batch_size_per_gpu 4 \
  --planned_tensor_file_path test/test_txt_data_MaxLength_100_BatchSize_8.pt \
  --dataset_specific_task_type pt \
  --model_path test/test_model \
  --zero_stage 3 \
  --use_bf16 \
  --learning_rate 1e-5 \
  --use_tensorboard \
  --num_epochs 3 \
  --checkpoint_save_path test/test_model_save_ds_train_pt
```

## Notes

when processing record count > 500K , multiprocess tokenization can possibly raise error, split the data before using mproc tokenizer 

[//]: # (# modules documentation)

[//]: # ()
[//]: # (| Module                                            | Description                                                                             |)

[//]: # (|---------------------------------------------------|-----------------------------------------------------------------------------------------|)

[//]: # (| GeneralDatasetTokenizer &#40;abstract&#41;                | multi process tokenizer base class                                                      |)

[//]: # (| GeneralDatasetHfTokenizer                         | multi process Huggingface tokenization                                                  |)

[//]: # (| GeneralDatasetBase &#40;abstract&#41;                     | base class for different dataset types                                                  |)

[//]: # (| FileBaseDataset                                   | base class for file-based dataset                                                       |)

[//]: # (| DefaultTaskSpecificFileBaseDatasetMetadataFactory | default metadata of file-based dataset  for different tasks                             |)

[//]: # (| GeneralTensorPlanner &#40;abstract&#41;                   | base class for different tensor planners                                                |)

[//]: # (| SmartBatchingTensorPlannerMixin  &#40;mix-in&#41;         | smart batching functions                                                                |)

[//]: # (| FixedBatchSizeTensorPlanner                       | tensor planner for arranging data in fixed batch size                                   |)

[//]: # (| GPUMemoryRestrictedTensorPlanner                  | tensor planner for arranging data with GPU memory limit                                 |)

[//]: # (| KeyFieldLengthDifferenceRestrictedTensorPlanner   | tensor planner for arranging data with key field length difference limit and batch size |)

[//]: # (| KeyFieldMaxLengthRestrictedTensorPlanner          | tensor planner for arranging data with key field max length limit and batch size        |)

[//]: # (| GeneralDataCollator                               | collate the planned tensors to form the batch                                           |)

[//]: # (| HFModelProvider                                   | load and save hf models                                                                 | )

[//]: # (| DeepSpeedModelProvider                            | load and save deepspeed model                                                           |)

[//]: # (| utils                                             | including multi-processing, multi threads, traverse, mirror traverse tools              | )

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # ( GeneralTrainingPipeline)

[//]: # ()
[//]: # (train the model with the planned dataset)

[//]: # ()
[//]: # ( DynamiCatSFTPipeline)

[//]: # ()
[//]: # (...)

[//]: # ()
[//]: # (Utils)

[//]: # ()
[//]: # ( RayDistUtils)

[//]: # ()
[//]: # (...)