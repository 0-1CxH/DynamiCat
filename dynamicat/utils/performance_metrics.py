from calflops import calculate_flops

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

