import os
import torch
import deepspeed
from loguru import logger
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from dynamicat.model.hf_model import HFModelProvider


class DeepSpeedHFModelProvider(HFModelProvider):

    @classmethod
    def save(cls, model_to_save, save_folder, global_rank=None, is_zero_stage_3=None):
        assert global_rank is not None, "Global rank should not be None"
        assert is_zero_stage_3 is not None, "Zero stage 3 should not be None"

        if global_rank == -1: # Single GPU
            if is_zero_stage_3:
                raise ValueError("Zero stage 3 is not supported for single GPU")
            else:
                logger.info("On single GPU, use normal HF save.")
                super().save(model_to_save, save_folder)
        else: # Multi GPU
            if is_zero_stage_3:
                logger.info("On multiple GPUs, Zero stage 3 enabled, need collect params before saving.")
                cls._save_deepspeed_model_of_zero_stage_3(model_to_save, save_folder, global_rank)
            else:
                if global_rank == 0: # Only rank 0 saves the model
                    logger.info("On multiple GPUs, save on rank 0 only.")
                    super().save(model_to_save, save_folder)


    @classmethod
    def _save_deepspeed_model_of_zero_stage_3(cls, model_to_save, save_folder, global_rank):
        model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save
        os.makedirs(save_folder, exist_ok=True)
        if global_rank <= 0:
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
                logger.trace(f"collecting {param_name}({parameters.ds_status}) from ds_id={parameters.ds_id}")
                if parameters.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                    with deepspeed.zero.GatheredParameters([parameters], enabled=True):
                         parameter_to_save = parameters.data.cpu()
            else:
                parameter_to_save = parameters.cpu()
            if global_rank <= 0:
                output_state_dict[param_name] = parameter_to_save
        if global_rank <= 0:
            torch.save(output_state_dict, output_model_file)
            logger.info(f"saved model weights to {output_model_file}")
        del output_state_dict

