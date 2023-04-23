"""
Llama GPTQ implementation
"""
import logging
import os
import sys
from typing import AsyncIterator, Dict, List

import huggingface_hub
import torch  # pylint: disable=import-error
from safetensors.torch import load_file as safe_load
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.modeling_utils import no_init_weights

from app.base import BaseLLM
from app.config import settings

sys.path.append(os.path.join(os.path.dirname(__file__), "GPTQ-for-LLaMa"))

try:
    from modelutils import find_layers
    from quant import make_quant
except ImportError as exp:
    raise ImportError(
        "the GPTQ-for-LLaMa lib is missing, please install it first"
    ) from exp


logger = logging.getLogger("llm-api.gptq_llama")


class GPTQLlamaLLM(BaseLLM):
    """
    Llama LLM implementation
    """

    def _download(self, model_path, model_dir):  # pylint: disable=duplicate-code
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

        huggingface_hub.hf_hub_download(
            repo_id=settings.setup_params["repo_id"],
            filename="tokenizer.model",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            cache_dir=os.path.join(settings.models_dir, ".cache"),
        )

        huggingface_hub.hf_hub_download(
            repo_id=settings.setup_params["repo_id"],
            filename="tokenizer_config.json",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            cache_dir=os.path.join(settings.models_dir, ".cache"),
        )

    def _setup(self):
        model_dir = os.path.join(settings.models_dir, settings.model_family)
        model_path = os.path.join(
            model_dir,
            settings.setup_params["filename"],
        )

        self._download(model_path, model_dir)

        logger.info("setup done successfully for %s", model_path)
        return model_path

    def __init__(self, params: Dict[str, str]) -> None:
        model_path = self._setup()
        group_size = params.get("group_size", 128)
        wbits = params.get("wbits", 4)
        cuda_visible_devices = params.get("cuda_visible_devices", "0")
        dev = params.get("device", "cuda:0")

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self.device = torch.device(dev)
        self.model = self._load_quant(
            settings.setup_params["repo_id"], model_path, wbits, group_size, self.device
        )

        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=False)

    def _load_quant(
        self, model, checkpoint, wbits, groupsize, device
    ):  # pylint: disable=too-many-arguments
        config = LlamaConfig.from_pretrained(model)

        def noop():
            pass

        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop

        torch.set_default_dtype(torch.half)
        with no_init_weights():
            torch.set_default_dtype(torch.half)
            model = LlamaForCausalLM(config)
            torch.set_default_dtype(torch.float)
            model = model.eval()  # pylint: disable=no-member
            layers = find_layers(model)
            for name in ["lm_head"]:
                if name in layers:
                    del layers[name]
            make_quant(model, layers, wbits, groupsize)

            logger.info("Loading model ...")
            print("Loading model ...")
            if checkpoint.endswith(".safetensors"):
                if device == -1:
                    device = "cpu"
                model.load_state_dict(safe_load(checkpoint, device))
            else:
                model.load_state_dict(torch.load(checkpoint))
            model.seqlen = 2048
            logger.info("Done loading model.")

        return model

    def generate(self, prompt: str, params: Dict[str, str]) -> str:
        """
        Generate text from Llama using the input prompt and parameters
        """

        min_length = params.get("min_length", 10)
        max_length = params.get("max_length", 50)
        top_p = params.get("top_p", 0.95)
        temperature = params.get("temp", 0.8)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                min_length=min_length,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
            )
        return self.tokenizer.decode([el.item() for el in generated_ids[0]])

    async def agenerate(
        self, prompt: str, params: Dict[str, str]
    ) -> AsyncIterator[str]:
        """
        Generate text stream from Llama using the input prompt and parameters
        """
        raise NotImplementedError("agenerate endpoint is not yet implemented")
        if False:  # pylint: disable=using-constant-test,disable=unreachable
            yield

    def embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for Llama using the input text
        """
        raise NotImplementedError("embeddings endpoint is not yet implemented")
