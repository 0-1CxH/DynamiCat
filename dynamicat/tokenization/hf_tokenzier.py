import os
from loguru import logger
from transformers import AutoTokenizer
from dynamicat.tokenization.tokenizer_base import GeneralDatasetTokenizer


class GeneralDatasetHfTokenizer(GeneralDatasetTokenizer):
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        assert os.path.exists(tokenizer_path), "tokenizer path does not exist"
        self.tokenizer = None
        super().__init__()

    def load_tokenizer(self, use_fast=False):
        if self.tokenizer:
            return self.tokenizer
        if "llama" in self.tokenizer_path or "Llama" in self.tokenizer_path:
            logger.debug("Llama tokenizer detected, using llama tokenizer.")
            from transformers.models.llama.tokenization_llama import LlamaTokenizer
            tokenizer_cls = LlamaTokenizer

        else:
            logger.debug("Using AutoTokenizer.")
            tokenizer_cls = AutoTokenizer

        self.tokenizer = tokenizer_cls.from_pretrained(
            self.tokenizer_path,
            use_fast=use_fast,
            trust_remote_code=True,
            # legacy=False # white space after special token problem: https://github.com/huggingface/transformers/issues/25073
        )
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        logger.debug(f"{self}")
        return self.tokenizer

    def save_tokenizer(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        self.tokenizer.save_pretrained(save_folder)
        logger.info(f"tokenizer {self.tokenizer} saved to {save_folder}")

    def __str__(self):
        if not self.tokenizer:
            return f"{__class__.__name__}(tokenizer_path={self.tokenizer_path})<Not Loaded>"
        else:
            return (f"{__class__.__name__}(tokenizer_path={self.tokenizer_path})<Loaded, length={len(self.tokenizer)}>\n"
                    f"BOS: {self.tokenizer.bos_token}, "
                    f"EOS: {self.tokenizer.eos_token}, "
                    f"PAD: {self.tokenizer.pad_token} \n"
                    f"BOS id: {self.tokenizer.bos_token_id}, "
                    f"EOS id: {self.tokenizer.eos_token_id}, "
                    f"PAD id: {self.tokenizer.pad_token_id}")


    def text_to_tensor(self, text, **tokenization_configs):
        truc_side = tokenization_configs.get("truncation_side")
        max_len = tokenization_configs.get("field_max_length")
        need_truncation = True
        if truc_side:
            if truc_side == "none":
                need_truncation = False
            else:
                self.tokenizer.truncation_side = truc_side
        if not max_len:
            need_truncation = False

        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=need_truncation,
            max_length=max_len,
            add_special_tokens=False
        ).input_ids