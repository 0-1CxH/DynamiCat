deepspeed dynamicat/training/sft_pipeline.py \
  --global_batch_size 32 \
  --batch_size_per_gpu 4 \
  --dataset_folder_path test/test_jsonl_data \
  --tokenizer_path test/test_tokenizer \
  --tensor_planner_type "GPUMemoryRestricted" \
  --tensor_parameter_count_limit 3000 \
  --model_path test/test_model \
  --zero_stage 2 \
  --zero_offload False \
  --use_bf16 \
  --learning_rate 1e-5 \
  --use_tensorboard