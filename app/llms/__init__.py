"""
LLMs are mapped here to thier named that used in config
"""

from typing import Type


def _load_llama():
    from .llama.llama import LlamaLLM  # pylint: disable=C0415

    return LlamaLLM


def _load_gptq_llama():
    from .gptq_llama.gptq_llama import GPTQLlamaLLM  # pylint: disable=C0415

    return GPTQLlamaLLM


def _load_hugging_face():
    from .huggingface.huggingface import HuggingFaceLLM  # pylint: disable=C0415

    return HuggingFaceLLM


def _load_autoawq():
    from .autoawq.autoawq import AutoAWQ  # pylint: disable=C0415

    return AutoAWQ


model_families = {
    "llama": _load_llama,
    "gptq_llama": _load_gptq_llama,
    "huggingface": _load_hugging_face,
    "autoawq": _load_autoawq,
}


def get_model_class(name: str) -> Type:
    """
    Retreives the LLM implementation class name by it's family name
    """
    if name not in model_families:
        raise RuntimeError(f"model {name} is not supported")
    return model_families[name]()
