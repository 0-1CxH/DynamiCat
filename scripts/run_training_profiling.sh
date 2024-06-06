deepspeed dynamicat/training/training_profile_pipeline.py \
  --global_batch_size 32 \
  --batch_size_per_gpu 4 \
  --model_path $1 \
  --zero_stage 3 \
  --zero_offload \
  --use_bf16 \
  --learning_rate 1e-5