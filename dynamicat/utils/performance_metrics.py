from calflops import calculate_flops
import pynvml
class ThroughputMetrics:
    def __init__(self, hf_model, hf_tokenizer, gradient_checkpointing):
        # run profile
        profile_batch_size = 16
        profile_seq_length = 16
        model_pprofile_flop, model_macs, self.model_param_count = calculate_flops(model=hf_model,
                                              input_shape=(profile_batch_size, profile_seq_length),
                                              transformer_tokenizer=hf_tokenizer,
                                              include_backPropagation=True,
                                              output_as_string=False)
        self.model_tflop_base = model_pprofile_flop / (profile_batch_size * profile_seq_length * 10**12)
        if gradient_checkpointing:
            self.model_tflop_base *= 4/3

    def get_throughput(self, step_time, batch_size, seq_length):
        return self.model_tflop_base * batch_size * seq_length / step_time

class GPUUtilizationMetrics:

    B_TO_GB_SCALE = 1024 * 1024 * 1024
    def __init__(self):
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        self.device_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
    def get_gpu_utilization(self):
        gpu_info = [
            {
                "used": pynvml.nvmlDeviceGetMemoryInfo(handle).used / self.B_TO_GB_SCALE,
                # "total": pynvml.nvmlDeviceGetMemoryInfo(handle).total / self.B_TO_GB_SCALE,
                "free": pynvml.nvmlDeviceGetMemoryInfo(handle).free / self.B_TO_GB_SCALE,
                "utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            } for handle in self.device_handles
        ]
        return gpu_info

    def iterate_key_values(self):
        for idx, gpu_info in enumerate(self.get_gpu_utilization()):
            for gpu_info_key in gpu_info:
                yield f"gpu{idx}_{gpu_info_key}", gpu_info[gpu_info_key]

    def get_total_memory_utilization(self):
        gpu_info = self.get_gpu_utilization()
        return sum([gpu["used"] for gpu in gpu_info]) / sum([gpu["used"] + gpu["free"] for gpu in gpu_info])

    def get_total_used_memory(self):
        gpu_info = self.get_gpu_utilization()
        return sum([gpu["used"] for gpu in gpu_info])


if __name__ == '__main__':
    print(GPUUtilizationMetrics().get_gpu_utilization())
