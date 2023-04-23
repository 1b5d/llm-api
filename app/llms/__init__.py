"""
LLMs are mapped here to thier named that used in config
"""

from typing import Type

from .gptq_llama.gptq_llama import GPTQLlamaLLM
from .llama.llama import LlamaLLM

model_families = {
    "alpaca": LlamaLLM,
    "llama": LlamaLLM,
    "vicuna": LlamaLLM,
    "gptq_llama": GPTQLlamaLLM,
}


def get_model_class(name: str) -> Type:
    """
    Retreives the LLM implementation class name by it's family name
    """
    if name not in model_families:
        raise RuntimeError(f"model {name} is not supported")
    return model_families[name]
