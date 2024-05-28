from transformers import AutoConfig
from loguru import logger

def load_hf_model(model_class, model_name_or_path, evaluation=False,
                  use_flash_attn=False, disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if disable_dropout:
        model_config.dropout = 0.0
    if use_flash_attn:
        logger.info("Flash attention enabled")
    if evaluation: # eval
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype="auto"
        )
    else: # train
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            use_flash_attention_2=use_flash_attn,
            trust_remote_code=True
        )
    logger.info(f"Model loaded from {model_name_or_path} with config: {model.config}, the model is {model}")
    return model

def set_model_special_tokens(model, tokenizer):
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"{model.config.bos_token_id=}, {model.config.end_token_id=}, {model.config.pad_token_id=}")
    return model