"""Base abstract class for LLM clients."""

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model_name: str, temperature: float = 0.3, max_tokens: int = 4096):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def complete(self, prompt: str, system_prompt: str = "") -> str:
        """Send a prompt and return the completion text."""
        pass

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Send messages and return the assistant response."""
        pass
