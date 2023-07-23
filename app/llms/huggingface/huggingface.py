"""
HuggingFaceLLM Implementation
"""
from typing import AsyncIterator, Dict, List

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)

from app.base import BaseLLM
from app.config import settings

from .utils import get_dtype_from_str


class HuggingFaceLLM(BaseLLM):
    """
    HuggingFaceLLM Implementation
    """

    def __init__(self, params: Dict[str, str]) -> None:
        model_dir = super().get_model_dir(
            settings.models_dir, settings.model_family, settings.setup_params["repo_id"]
        )

        setup_params = {
            k: v
            for k, v in settings.setup_params.items()
            if k not in ("repo_id", "tokenizer_repo_id", "config_params")
        }

        config = AutoConfig.from_pretrained(
            settings.setup_params["repo_id"], **setup_params
        )

        config_params = settings.setup_params.get("config_params", {})
        config_params = config_params or {}
        for attr_name, attr_value in config_params.items():
            if hasattr(config, attr_name) and isinstance(attr_value, dict):
                attr_config = getattr(config, attr_name)
                attr_config.update(attr_value)
                attr_value = attr_config
            setattr(config, attr_name, attr_value)
        self.device = params.get("device_map", "cpu")

        if "torch_dtype" in params:
            params["torch_dtype"] = get_dtype_from_str(params["torch_dtype"])

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.setup_params["repo_id"],
            config=config,
            cache_dir=model_dir,
            **params
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.setup_params["tokenizer_repo_id"],
            return_token_type_ids=False,
            cache_dir=model_dir,
            **params
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, params: Dict[str, str]) -> str:
        """
        Generate text from Huggingface model using the input prompt and parameters
        """
        input_ids = self.tokenizer(
            prompt,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        gen_tokens = self.model.generate(
            **input_ids, pad_token_id=self.tokenizer.pad_token_id, **params
        )
        result = self.tokenizer.batch_decode(
            gen_tokens[:, input_ids["input_ids"].shape[1] :]
        )
        return result[0]

    async def agenerate(
        self, prompt: str, params: Dict[str, str]
    ) -> AsyncIterator[str]:
        """
        Generate text stream Huggingface model using the input prompt and parameters
        """
        input_ids = self.tokenizer(
            prompt,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        self.model.generate(
            **input_ids,
            streamer=streamer,
            pad_token_id=self.tokenizer.pad_token_id,
            **params or None
        )
        for text in streamer:
            yield text

    def embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings using the input text
        """

        pipe = pipeline(
            "feature-extraction",
            framework="pt",
            model=self.model,
            tokenizer=self.tokenizer,
            tokenize_kwargs={
                "return_token_type_ids": False,
                "return_attention_mask": True,
            },
        )
        return pipe(text)[0][0]
