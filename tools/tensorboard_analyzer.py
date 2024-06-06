from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

dpath = "/ML-A100/team/infra/qhu/DynamiCat/dev/tensorboards/belle_F12"
e = EventAccumulator(dpath)
e.Reload()

print(e.Tags())

flops_series = [_.value for _ in e.Scalars('tflops (mean)')]
tflops_during_training = sum(flops_series)/len(flops_series)
total_time_of_training = sum([_.value for _ in e.Scalars('time (mean)')])
all_used_data = []
all_free_data = []
for tag in e.Tags()['scalars']:
    if tag.endswith('_used'):
        all_used_data.extend([_.value for _ in e.Scalars(tag)])
    if tag.endswith('_free'):
        all_free_data.extend([_.value for _ in e.Scalars(tag)])
gpu_mem_util_during_training = sum(all_used_data)/ (sum(all_used_data) + sum(all_free_data))

print(f"tflops_during_training: {tflops_during_training}\n"
      f"total_time_of_training: {total_time_of_training}\n"
      f"gpu_mem_util_during_training: {gpu_mem_util_during_training}")
