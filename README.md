
# DynamiCat

Efficient LLM training pipeline with dynamic batch size, smart batching and dynamic padding that maximizes GPU memory utilization and training speed.


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


# modules documentation



## GeneralDatasetTokenizer

multi process tokenization for general dataset of different purposes / formats

## GeneralDatasetHfTokenizer

### PretrainDatasetTokenizer

### SFTDatasetTokenizer
(multi-turn conversation using mask)
tokenize the dataset of the SFT format, usually contains prompt and response

### InferenceDatasetTokenizer

tokenize the dataset of the Inference format, usually only contains prompt

### DPODatasetTokenizer

tokenize the dataset of the DPO format, usually contains prompt, chosen and rejected responses

### ....


## GeneralTensorPlanner

plan on the tokenized tensors to make them meets some requirements


### SmartBatchingTensorPlannerMixin


### FixedBatchSizeTensorPlanner



### GPUMemoryRestrictedTensorPlanner

set the max memory usage of the GPU, and plan on the tensors to make them fit the memory

### TokenDifferenceRestrictedTensorPlanner

set the max token difference between the tensors, and plan on the tensors to make them fit the requirement

### ...


## GeneralPlannedDatasetCollator

collate the planned tensors to form the batch

### GPUMemoryRestrictedDatasetCollator

collate the planned tensors with the GPU memory restriction (can be used with GPUMemoryRestrictedTensorPlanner)

### TokenDifferenceRestrictedDatasetCollator

collate the planned tensors with the token difference restriction (can be used with TokenDifferenceRestrictedTensorPlanner)

### ...


## GeneralTrainingPipeline

train the model with the planned dataset

### DynamiCatSFTPipeline

### ...

## Utils

some useful functions

### MProcUtils

### RayDistUtils

### DeepSpeedUtils

### ...