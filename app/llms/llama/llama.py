"""
Alpaca LLM implementation
"""

import logging
import os
from typing import Any, AsyncIterator, Dict, List

import huggingface_hub
from llama_cpp import Llama
from sentencepiece import SentencePieceProcessor

from app.base import BaseLLM
from app.config import settings

from .convert import convert_one_file
from .migrate import migrate

logger = logging.getLogger("llm-api.llama")


class LlamaLLM(BaseLLM):
    """
    Llama LLM implementation
    """

    def _setup(self):
        model_path = os.path.join(
            settings.models_dir,
            f"ggml-{settings.model_family}-{settings.model_name}-q4.bin",
        )

        if os.path.exists(model_path):
            logger.info("found an existing model %s", model_path)
            return model_path

        logger.info("downloading model to %s", model_path)

        huggingface_hub.hf_hub_download(
            repo_id=settings.setup_params["repo_id"],
            filename=settings.setup_params["filename"],
            local_dir=settings.models_dir,
            local_dir_use_symlinks=False,
            cache_dir=os.path.join(settings.models_dir, ".cache"),
        )

        os.rename(
            os.path.join(settings.models_dir, settings.setup_params["filename"]),
            model_path,
        )

        if settings.setup_params["convert"]:
            tokenizer_model_path = os.path.join(settings.models_dir, "tokenizer.model")
            logger.info("downloading tokenizer model %s", tokenizer_model_path)

            huggingface_hub.hf_hub_download(
                repo_id="decapoda-research/llama-7b-hf",
                filename="tokenizer.model",
                local_dir=settings.models_dir,
                local_dir_use_symlinks=False,
                cache_dir=os.path.join(settings.models_dir, ".cache"),
            )

            tokenizer = (
                SentencePieceProcessor(  # pylint: disable=too-many-function-args
                    tokenizer_model_path
                )
            )
            logger.info("converting model %s", model_path)
            convert_one_file(model_path, tokenizer)

        if settings.setup_params["migrate"]:
            logger.info("migrating model %s", model_path)
            migrate(model_path)
            # clean up backed model since we won't need it
            logger.info("cleaning up ..")
            os.remove(model_path + ".orig")

        logger.info("setup done successfully for %s", model_path)
        return model_path

    def __init__(self, params: Dict[str, str]) -> None:
        n_ctx = params.get("ctx_size", 2000)
        n_parts = params.get("n_parts", -1)
        seed = params.get("seed", 1337)
        n_threads = params.get("num_threads", 4)
        n_batch = params.get("batch_size", 2048)
        last_n_tokens_size = params.get("last_n_tokens_size", 64)

        model_path = self._setup()

        self.llama = Llama(
            model_path,
            n_ctx=n_ctx,
            n_parts=n_parts,
            seed=seed,
            f16_kv=True,
            logits_all=False,
            vocab_only=False,
            use_mlock=True,
            embedding=True,
            n_threads=n_threads,
            n_batch=n_batch,
            last_n_tokens_size=last_n_tokens_size,
        )

    def generate(self, prompt: str, params: Dict[str, str]) -> str:
        """
        Generate text from Llama using the input prompt and parameters
        """
        if params is None:
            params = {}
        n_predict = params.get("n_predict", 300)
        temp = params.get("temp", 0.1)
        top_k = params.get("top_k", 40)
        top_p = params.get("top_p", 0.95)
        stop: Any = params.get("stop", [])
        repeat_penalty = params.get("repeat_penalty", 1.3)
        result = self.llama(
            prompt=prompt,
            max_tokens=n_predict,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            repeat_penalty=repeat_penalty,
        )
        return result["choices"][0]["text"]

    async def agenerate(
        self, prompt: str, params: Dict[str, str]
    ) -> AsyncIterator[str]:
        """
        Generate text stream from Llama using the input prompt and parameters
        """
        if params is None:
            params = {}
        n_predict = params.get("n_predict", 300)
        temp = params.get("temp", 0.1)
        top_k = params.get("top_k", 40)
        top_p = params.get("top_p", 0.95)
        stop: Any = params.get("stop", [])
        repeat_penalty = params.get("repeat_penalty", 1.3)
        generator = self.llama(
            prompt=prompt,
            max_tokens=n_predict,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            repeat_penalty=repeat_penalty,
            stream=True,
        )
        for item in generator:
            yield item["choices"][0]["text"]

    def embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for Llama using the input text
        """
        return self.llama.create_embedding(text)["data"][0]["embedding"]
