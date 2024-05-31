import os
import torch
import deepspeed
from loguru import logger
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from dynamicat.model.hf_model import HFModelProvider
from dynamicat.utils.deepspeed_utils import DeepSpeedModelTrainingUtils


# def load deepspeed model



class DeepSpeedHFModelProvider(DeepSpeedModelTrainingUtils, HFModelProvider):

    @classmethod
    def load(
            cls,
            hf_model_clz,
            model_name_or_path,
            evaluation=False,
            use_flash_attn=False,
            disable_dropout=False,
            cmd_args=None,
            ds_config=None,
            **kwargs
    ):

        assert cmd_args is not None, "Command line arguments should not be None"
        assert ds_config is not None, "DeepSpeed Config should not be None"

        assert hasattr(cmd_args, "local_rank"), "Command line arguments should have local rank"

        dist_env_enabled = cmd_args.local_rank != -1

        if dist_env_enabled:
            deepspeed.init_distributed()


        hf_model = super().load(
            hf_model_clz,
            model_name_or_path,
            evaluation,
            use_flash_attn,
            disable_dropout,
        )

        optimizer = cls.get_optimizer(hf_model, **kwargs)
        lr_scheduler = cls.get_lr_scheduler(optimizer, **kwargs)


        if dist_env_enabled:
            torch.distributed.barrier()

        deepspeed_model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=hf_model,
            optimizer=optimizer,
            args=cmd_args,
            config=ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True)

        logger.info(f"DeepSpeed model loaded successfully, {deepspeed_model_engine=}, {optimizer=}, {lr_scheduler=}")

        return deepspeed_model_engine

    @classmethod
    def save(cls, model_to_save, save_folder, global_rank=None, is_zero_stage_3=None):
        assert global_rank is not None, "Global rank should not be None"
        assert is_zero_stage_3 is not None, "Zero stage 3 should not be None"

        if global_rank == -1: # Single GPU
            if is_zero_stage_3:
                raise ValueError("Zero stage 3 is not supported for single GPU")
            else:
                super().save(model_to_save, save_folder)
        else: # Multi GPU
            if is_zero_stage_3:
                cls._save_deepspeed_model_of_zero_stage_3(model_to_save, save_folder, global_rank)
            else:
                if global_rank == 0: # Only rank 0 saves the model
                    super().save(model_to_save, save_folder)



    @classmethod
    def _save_deepspeed_model_of_zero_stage_3(cls, model_to_save, save_folder, global_rank):
        model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save
        os.makedirs(save_folder, exist_ok=True)
        logger.info(f"saving model {model_to_save} with config: {model_to_save.config} to {save_folder}")
        # save config
        cls._save_config_file(model_to_save, save_folder)
        # save weights
        output_model_file = os.path.join(save_folder, "pytorch_model.bin")
        output_state_dict = {}
        for param_name, parameters in model_to_save.named_parameters():
            if "lora" in param_name:
                logger.warning(f"Skipping {param_name} as it is a LoRA parameter")
                continue
            if hasattr(parameters, 'ds_id'):
                if parameters.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                    with deepspeed.zero.GatheredParameters([parameters], enabled=True):
                         parameter_to_save = parameters.data
            else:
                parameter_to_save = parameters
            if global_rank == 0:
                output_state_dict[param_name] = parameter_to_save.cpu()
                # logger.debug(f"adding {param_name} with shape {parameter_to_save.shape}")

        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
            logger.info(f"saved model weights to {output_model_file}")
        del output_state_dict



