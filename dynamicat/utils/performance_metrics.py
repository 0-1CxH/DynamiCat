class ThroughputMetrics:
    def __init__(self, hf_model, gradient_checkpointing):
        self.num_layers = getattr(hf_model.config, "num_hidden_layers", getattr(hf_model.config, "n_layer", None))
        self.hidden_size = getattr(hf_model.config, "hidden_size", getattr(hf_model.config, "n_embd", None))
        self.vocab_size = getattr(hf_model.config, "vocab_size", None)
        if not all((self.num_layers, self.hidden_size, self.vocab_size)):
            raise ValueError("Could not determine number of layers, hidden size, and vocab size of the model")
        self.gradient_checkpointing = gradient_checkpointing
        self.model_params_count = sum([p.numel() for p in hf_model.parameters()])

    def calculate_flops(self, checkpoint_activations_factor, batch_size, seq_length):
        flops_per_iteration = ((24 * checkpoint_activations_factor * batch_size * seq_length * self.num_layers * (self.hidden_size**2)) *
                               (1.0 + (seq_length / (6.0 * self.hidden_size)) + (self.vocab_size / (16.0 * self.num_layers * self.hidden_size))))
        return flops_per_iteration

    def get_throughput(self, step_time, batch_size, seq_length):
        checkpoint_activations_factor = 4 if self.gradient_checkpointing else 3
        train_flops_per_iteration = self.calculate_flops(checkpoint_activations_factor, batch_size, seq_length)
        train_tflops = train_flops_per_iteration / (step_time * (10**12))
        return train_tflops
