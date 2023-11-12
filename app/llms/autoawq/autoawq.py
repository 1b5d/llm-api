"""
AutoAWQ LLM-API implementation
"""
import logging
import os
from typing import AsyncIterator, Dict, List

import huggingface_hub
from awq import AutoAWQForCausalLM
from transformers import AutoConfig, AutoTokenizer, TextIteratorStreamer, pipeline

from app.base import BaseLLM
from app.config import settings

logger = logging.getLogger("llm-api.autoawq")


class AutoAWQ(BaseLLM):
    """LLM-API implementation to support AWQ quantization using AutoAWQ"""

    def _download(self, model_path, model_dir):
        if os.path.exists(model_path):
            logger.info("found an existing model %s", model_path)
            return

        logger.info("downloading model to %s", model_path)

        huggingface_hub.hf_hub_download(
            repo_id=settings.setup_params["repo_id"],
            filename=settings.setup_params["filename"],
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            cache_dir=os.path.join(settings.models_dir, ".cache"),
        )

        huggingface_hub.hf_hub_download(
            repo_id=settings.setup_params["repo_id"],
            filename="config.json",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            cache_dir=os.path.join(settings.models_dir, ".cache"),
        )

    def __init__(self, params: Dict[str, str]) -> None:
        model_dir = super().get_model_dir(
            settings.models_dir,
            settings.model_family,
            settings.setup_params["filename"],
        )

        model_path = os.path.join(
            model_dir,
            settings.setup_params["filename"],
        )

        self._download(model_path, model_dir)

        self.device = params.get("device_map", "cuda:0")
        del params["device_map"]

        self.config = AutoConfig.from_pretrained(settings.setup_params["repo_id"])

        self.model = AutoAWQForCausalLM.from_quantized(
            model_dir, settings.setup_params["filename"], **params
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.setup_params["tokenizer_repo_id"], cache_dir=model_dir, **params
        )

        logger.info("setup done successfully for %s", model_path)

    def generate(self, prompt: str, params: Dict[str, str]) -> str:
        """
        Generate text from the model using the input prompt and parameters
        """
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        gen_tokens = self.model.generate(**input_ids, **params)

        result = self.tokenizer.batch_decode(
            gen_tokens[:, input_ids["input_ids"].shape[1] :]
        )

        return result[0]

    async def agenerate(
        self, prompt: str, params: Dict[str, str]
    ) -> AsyncIterator[str]:
        """
        Generate text stream from model using the input prompt and parameters
        """
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        self.model.generate(**input_ids, streamer=streamer, **params or None)
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
        )
        return pipe(text)[0][0]
