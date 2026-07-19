"""Shared, injectable dependencies for the agent graph.

The provider deliberately creates the OpenAI client on first use.  Importing a
deterministic rule or collecting tests therefore never requires credentials.
"""

from typing import Protocol

from src.llm.gpt_client import GPTClient
from src.prompt_engineering.templates import PromptTemplateManager


class LLMClient(Protocol):
    def complete(self, prompt: str, system_prompt: str = "") -> str: ...
    def chat(self, messages: list[dict]) -> str: ...


class LazyLLMClient:
    """Small dependency-injection seam with a production default."""

    def __init__(self) -> None:
        self._client: LLMClient | None = None

    def configure(self, client: LLMClient | None) -> None:
        """Inject a fake client in tests, or ``None`` to restore lazy default."""
        self._client = client

    def _get(self) -> LLMClient:
        if self._client is None:
            self._client = GPTClient()
        return self._client

    def complete(self, prompt: str, system_prompt: str = "") -> str:
        return self._get().complete(prompt, system_prompt=system_prompt)

    def chat(self, messages: list[dict]) -> str:
        return self._get().chat(messages)


llm_client = LazyLLMClient()
prompt_templates = PromptTemplateManager()


def configure_llm_client(client: LLMClient | None) -> None:
    """Public hook used by tests and offline integrations."""
    llm_client.configure(client)
