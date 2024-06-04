from deepspeed import DeepSpeedConfig
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler, SchedulerType
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from dynamicat.utils.custom_lr_scheduler import get_wsd_scheduler


class DeepSpeedConfigBuilder:

    @classmethod
    def _make_compulsory_training_config_of_zero_optimization(
            cls,
            zero_stage: int,
            offload: bool
    ):
        offload_device = "cpu" if offload else "none"
        return {
            "stage": zero_stage,
            "offload_param": {
                "device": offload_device
            },
            "offload_optimizer": {
                "device": offload_device
            },
            "reduce_bucket_size": 5e5,
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e5,
            "stage3_prefetch_bucket_size": 3e5,
            "stage3_model_persistence_threshold": 3e5,
            "memory_efficient_linear": False
        }

    @classmethod
    def _make_compulsory_eval_config_of_zero_optimization(
            cls,
            zero_stage: int,
            offload: bool
    ):
        offload_device = "cpu" if offload else "none"
        return {
            "stage": zero_stage,
            "stage3_param_persistence_threshold": 1e4,
            "offload_param": {
                "device": offload_device
            },
            "memory_efficient_linear": False
        }

    @classmethod
    def _make_compulsory_training_config_for_hybrid_engine(
            cls,
            enable_hybrid_engine=False,
            inference_tp_size=1,
            release_inference_cache=False,
            pin_parameters=True,
            tp_gather_partition_size=8,
            max_out_tokens=512,
            **kwargs

    ):
        return {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        }

    @classmethod
    def _make_optional_training_config_for_tensorboard(
            cls,
            enable_tensorboard = True,
            tensorboard_save_path = ".",
            tensorboard_job_name = "deepspeed_tensorboard", # will be sub folder in tensorboard_save_path
            **kwargs
    ):
        return {
            "tensorboard": {
                "enabled": enable_tensorboard,
                "output_path": tensorboard_save_path,
                "job_name": tensorboard_job_name,
                }
        }

    @classmethod
    def _make_optional_config_for_fp16(cls, loss_scale_window=100, **kwargs):
        return {
            "fp16": {
                "enabled": True,
                "loss_scale_window": loss_scale_window,
            }
        }

    @classmethod
    def _make_optional_config_for_bf16(cls):
        return {
            "fp16": {
                "enabled": False,
            },
            "bf16":{
                "enabled": True
            }
        }

    @classmethod
    def make_config_for_training(
            cls,
            global_batch_size,
            batch_size_per_gpu,
            zero_stage,
            zero_offload,
            use_bf16,
            use_tensorboard,
            return_dict=True,
            **kwargs

    ):

        result_config = {
            "train_batch_size": global_batch_size,
            "train_micro_batch_size_per_gpu": batch_size_per_gpu,
            "steps_per_print": 10,
            "gradient_clipping": 1.0,
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
            "zero_optimization": cls._make_compulsory_training_config_of_zero_optimization(zero_stage, zero_offload),
            "hybrid_engine": cls._make_compulsory_training_config_for_hybrid_engine(**kwargs),
        }
        if use_bf16:
            result_config.update(cls._make_optional_config_for_bf16())
        else:
            result_config.update(cls._make_optional_config_for_fp16(**kwargs))

        if use_tensorboard:
            result_config.update(cls._make_optional_training_config_for_tensorboard(**kwargs))

        if return_dict:
            return result_config
        else:
            return DeepSpeedConfig(result_config)

    @classmethod
    def make_config_for_eval(
            cls,
            global_batch_size,
            batch_size_per_gpu,
            zero_stage,
            zero_offload,
            use_bf16,
            return_dict=True,
            **kwargs
    ):
        result_config = {
            "train_batch_size": global_batch_size,
            "train_micro_batch_size_per_gpu": batch_size_per_gpu,
            "steps_per_print": 10,
            "gradient_clipping": 1.0,
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
            "zero_optimization": cls._make_compulsory_eval_config_of_zero_optimization(zero_stage, zero_offload),

        }
        if use_bf16:
            result_config.update(cls._make_optional_config_for_bf16())
        else:
            result_config.update(cls._make_optional_config_for_fp16(**kwargs))

        if return_dict:
            return result_config
        else:
            return DeepSpeedConfig(result_config)


class DeepSpeedModelTrainingUtils:

    @classmethod
    def _get_optimizer_model_parameters_with_weight_decay(
            cls,
            model,
            weight_decay,
            no_decay_name_list=("bias", 'layernorm.weight'),
    ):
        if weight_decay == 0.0:
            return [
                {
                    "params": [parameter for _, parameter in model.named_parameters()],
                    "weight_decay": weight_decay
                }
            ]

        params_need_weight_decay = []
        params_no_need_weight_decay = []
        for param_name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if any(_ in param_name for _ in no_decay_name_list):
                    params_no_need_weight_decay.append(parameter)
                else:
                    params_need_weight_decay.append(parameter)
        grouped_model_parameters = []
        if len(params_need_weight_decay) > 0:
            grouped_model_parameters.append(
                {
                    "params": params_need_weight_decay,
                    "weight_decay": weight_decay
                }
            )
        if len(params_no_need_weight_decay) > 0:
            grouped_model_parameters.append(
                {
                    "params": params_no_need_weight_decay,
                    "weight_decay": 0.0
                }
            )
        return grouped_model_parameters

    @classmethod
    def get_optimizer(
            cls,
            model,
            learning_rate,
            offload,
            weight_decay=0.0,
    ):
        grouped_model_params = cls._get_optimizer_model_parameters_with_weight_decay(model, weight_decay)
        optimizer_clz = DeepSpeedCPUAdam if offload else FusedAdam
        optimizer = optimizer_clz(
            grouped_model_params,
            lr=learning_rate,
            betas=(0.9, 0.95)
        )
        logger.debug(f"Optimizer loaded: {optimizer}")
        return optimizer

    @classmethod
    def get_lr_scheduler(
            cls,
            optimizer,
            scheduler_type,
            num_warmup_steps=None,
            num_training_steps=None,
            scheduler_specific_kwargs=None,
            lr_lambda=None,
    ):
        if scheduler_type in SchedulerType.__members__.values():
            scheduler = get_scheduler(
                scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=scheduler_specific_kwargs
            )
        elif scheduler_type == "wsd":
            scheduler = get_wsd_scheduler(
                    optimizer=optimizer,
                    num_training_steps=num_training_steps,
                    **scheduler_specific_kwargs
            )
        else:
            assert lr_lambda is not None
            scheduler = LambdaLR(optimizer, lr_lambda, -1)
        logger.debug(f"LR Scheduler loaded: {scheduler}")
        return scheduler

