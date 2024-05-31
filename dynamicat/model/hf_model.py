import os
import torch
from transformers import AutoConfig
from loguru import logger
from dynamicat.tokenization.hf_tokenzier import GeneralDatasetHfTokenizer

class HFModelProvider:
    @classmethod
    def load(
            cls,
            hf_model_clz,
            model_name_or_path,
            evaluation=False,
            use_flash_attn=False,
            disable_dropout=False
    ):
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if disable_dropout:
            model_config.dropout = 0.0
        if use_flash_attn:
            logger.info("Flash attention enabled")
        if evaluation:  # eval
            model = hf_model_clz.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype="auto"
            )
        else:  # train
            model = hf_model_clz.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config,
                use_flash_attention_2=use_flash_attn,
                trust_remote_code=True
            )
        logger.info(f"Model loaded from {model_name_or_path} with config: {model.config}, the model is {model}")
        return model

    @classmethod
    def set_model_special_tokens(cls, model, hf_tokenizer: GeneralDatasetHfTokenizer):
        _internal_tokenizer = hf_tokenizer.tokenizer
        model.config.bos_token_id = _internal_tokenizer.bos_token_id
        model.config.eos_token_id = _internal_tokenizer.eos_token_id
        model.config.pad_token_id = _internal_tokenizer.pad_token_id

        logger.info(f"Set model special tokens: {model.config.bos_token_id=}, {model.config.eos_token_id=}, {model.config.pad_token_id=}")
        return model

    @classmethod
    def _save_config_file(cls, model_to_save, save_folder):
        # save config
        output_config_file = os.path.join(save_folder, "config.json")
        model_to_save.config.to_json_file(output_config_file)
        logger.info(f"saved model config to {output_config_file}")

    @classmethod
    def save(cls, model_to_save, save_folder):
        model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save
        os.makedirs(save_folder, exist_ok=True)
        logger.info(f"saving model {model_to_save} with config: {model_to_save.config} to {save_folder}")
        # save config
        cls._save_config_file(model_to_save, save_folder)
        # save weights
        output_model_file = os.path.join(save_folder, "pytorch_model.bin")
        save_dict = model_to_save.state_dict()
        torch.save(save_dict, output_model_file)
        logger.info(f"saved model weights to {output_model_file}")







