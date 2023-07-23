"""
Llama LLM implementation
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

    def _setup(self):
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

        if settings.setup_params.get("convert", False):
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
            try:
                convert_one_file(model_path, tokenizer)
            except Exception as exp:  # pylint: disable=broad-exception-caught
                logger.warning("Could not convert the model %s", str(exp))

        if settings.setup_params.get("migrate", False):
            logger.info("migrating model %s", model_path)
            migrate(model_path)
            # clean up backed model since we won't need it
            logger.info("cleaning up ..")
            if os.path.exists(model_path + ".orig"):
                os.remove(model_path + ".orig")

        logger.info("setup done successfully for %s", model_path)
        return model_path

    def __init__(  # pylint: disable=too-many-locals
        self, params: Dict[str, str]
    ) -> None:
        n_ctx = params.get("n_ctx", 2000)
        n_parts = params.get("n_parts", -1)
        n_gpu_layers = int(params.get("n_gpu_layers", 0))
        seed = params.get("seed", 1337)
        use_mmap = params.get("use_mmap", True)
        n_threads = params.get("num_threads", 4)
        n_batch = params.get("batch_size", 2048)
        last_n_tokens_size = params.get("last_n_tokens_size", 64)
        lora_base = params.get("lora_base")
        lora_path = params.get("lora_path")
        low_vram = params.get("low_vram", False)
        tensor_split = params.get("tensor_split", None)
        rope_freq_base = params.get("rope_freq_base", 10000.0)
        rope_freq_scale = params.get("rope_freq_scale", 1.0)
        verbose = params.get("verbose", True)

        model_path = self._setup()

        self.llama = Llama(
            model_path,
            n_ctx=n_ctx,
            n_parts=n_parts,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            f16_kv=True,
            logits_all=False,
            vocab_only=False,
            use_mmap=use_mmap,
            use_mlock=True,
            embedding=True,
            n_threads=n_threads,
            n_batch=n_batch,
            last_n_tokens_size=last_n_tokens_size,
            lora_base=lora_base,
            lora_path=lora_path,
            low_vram=low_vram,
            tensor_split=tensor_split,
            rope_freq_base=rope_freq_base,
            rope_freq_scale=rope_freq_scale,
            verbose=verbose,
        )

    def generate(self, prompt: str, params: Dict[str, str]) -> str:
        """
        Generate text from Llama using the input prompt and parameters
        """
        if params is None:
            params = {}
        suffix = params.get("suffix")
        max_tokens = params.get("max_tokens", 128)
        temperature = params.get("temperature", 0.8)
        top_p = params.get("top_p", 0.95)
        logprobs = params.get("logprobs")
        echo = params.get("echo", False)
        stop: Any = params.get("stop", [])
        frequency_penalty = params.get("frequency_penalty", 0.0)
        presence_penalty = params.get("presence_penalty", 0.0)
        repeat_penalty = params.get("repeat_penalty", 1.1)
        top_k = params.get("top_k", 40)

        result = self.llama(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=False,
        )
        return result["choices"][0]["text"]

    async def agenerate(  # pylint: disable=too-many-locals
        self, prompt: str, params: Dict[str, str]
    ) -> AsyncIterator[str]:
        """
        Generate text stream from Llama using the input prompt and parameters
        """
        if params is None:
            params = {}
        suffix = params.get("suffix")
        max_tokens = params.get("max_tokens", 128)
        temperature = params.get("temperature", 0.8)
        top_p = params.get("top_p", 0.95)
        logprobs = params.get("logprobs")
        echo = params.get("echo", False)
        stop: Any = params.get("stop", [])
        frequency_penalty = params.get("frequency_penalty", 0.0)
        presence_penalty = params.get("presence_penalty", 0.0)
        repeat_penalty = params.get("repeat_penalty", 1.1)
        top_k = params.get("top_k", 40)

        generator = self.llama(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=True,
        )
        for item in generator:
            yield item["choices"][0]["text"]

    def embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for Llama using the input text
        """
        return self.llama.create_embedding(text)["data"][0]["embedding"]
