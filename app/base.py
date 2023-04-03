"""
An interface which defines generic LLM related operations
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List


class BaseLLM(ABC):

    """
    A base class for LLMs
    """

    @abstractmethod
    def generate(self, prompt: str, params: Dict[str, str]) -> str:
        """
        generate text using LLM based on an input prompt
        """

    @abstractmethod
    async def agenerate(
        self, prompt: str, params: Dict[str, str]
    ) -> AsyncIterator[str]:
        """
        asynchronously generate text using LLM based on an input prompt
        """
        # avoid mypy error https://github.com/python/mypy/issues/5070
        if False:  # pylint: disable=using-constant-test
            yield

    @abstractmethod
    def embeddings(self, text: str) -> List[float]:
        """
        create embeddings from the input text
        """
