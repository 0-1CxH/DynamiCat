
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