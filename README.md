
# DynamiCat

Efficient LLM training/inference pipeline with dynamic batch size, smart batching and dynamic padding that maximizes GPU memory utilization, training speed and inference throughput.


![compare_loss_curve.png](img%2Fcompare_loss_curve.png)

X-axis represents training step, y-axis represents training loss. Green line is loss curve of normal fine-tuning pipeline, orange line is loss curve of DynamiCat pipeline (with dynamic batch size, smart batching and dynamic padding)


Experiment environment:
- 8x A800 80GB GPUs
- Model size: 3B
- Dataset size: 50k

| Pipeline                    | Training time (per epoch) | GPU memory utilization |
|-----------------------------|---------------------------|------------------------|
| Normal fine-tuning pipeline | 160 min                   | 74.97%                 |
| DynamiCat pipeline          | 27 min                    | 89.21%                 |

TODO: test with more public datasets and models to replace this table 



# support
pre-training, fine-tuning, inference


# Test Cases

## Step 1: Tokenization

```bash
bash scripts/tokenize/run_tokenize_default_sft.sh -i  test/test_jsonl_data -t test/test_tokenizer/
bash scripts/tokenize/run_tokenize_default_pt.sh -i test/test_txt_data -t test/test_tokenizer/
bash scripts/tokenize/run_tokenize_by_metadata_file.sh -i test/test_metadata_file.json -t test/test_tokenizer/
```
## Step 2: Tensor Planning

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

## Step 3: Training

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



# modules documentation

| Module                                            | Description                                                                             |
|---------------------------------------------------|-----------------------------------------------------------------------------------------|
| GeneralDatasetTokenizer (abstract)                | multi process tokenizer base class                                                      |
| GeneralDatasetHfTokenizer                         | multi process Huggingface tokenization                                                  |
| GeneralDatasetBase (abstract)                     | base class for different dataset types                                                  |
| FileBaseDataset                                   | base class for file-based dataset                                                       |
| DefaultTaskSpecificFileBaseDatasetMetadataFactory | default metadata of file-based dataset  for different tasks                             |
| GeneralTensorPlanner (abstract)                   | base class for different tensor planners                                                |
| SmartBatchingTensorPlannerMixin  (mix-in)         | smart batching functions                                                                |
| FixedBatchSizeTensorPlanner                       | tensor planner for arranging data in fixed batch size                                   |
| GPUMemoryRestrictedTensorPlanner                  | tensor planner for arranging data with GPU memory limit                                 |
| KeyFieldLengthDifferenceRestrictedTensorPlanner   | tensor planner for arranging data with key field length difference limit and batch size |
| KeyFieldMaxLengthRestrictedTensorPlanner          | tensor planner for arranging data with key field max length limit and batch size        |
| GeneralDataCollator                               | collate the planned tensors to form the batch                                           |
| HFModelProvider                                   | load and save hf models                                                                 | 
| DeepSpeedModelProvider                            | load and save deepspeed model                                                           |
| utils                                             | including multi-processing, multi threads, traverse, mirror traverse tools              | 







 GeneralTrainingPipeline

train the model with the planned dataset

 DynamiCatSFTPipeline

...

Utils

 RayDistUtils

...