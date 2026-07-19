"""OpenAI GPT client."""

import os
from openai import OpenAI
from dotenv import load_dotenv
from src.llm.base import BaseLLMClient
from src.llm.cache import LLMCache
from src.utils.logger import get_logger
from config import load_yaml_config

load_dotenv()

_model_config = load_yaml_config("model_config.yaml")
_default_model = _model_config["default_model"]
_default_params = _model_config["models"][_default_model]

logger = get_logger(__name__)


class GPTClient(BaseLLMClient):
    """Client for OpenAI GPT models."""

    def __init__(
        self,
        model_name: str = _default_params["model_name"],
        temperature: float = _default_params["temperature"],
        max_tokens: int = _default_params["max_tokens"],
        seed: int | None = _default_params.get("seed"),
        supports_temperature: bool = _default_params.get("supports_temperature", True),
    ):
        super().__init__(model_name, temperature, max_tokens)
        self.seed = seed
        self.supports_temperature = supports_temperature
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.client = OpenAI(api_key=api_key)
        # Responses may contain reconstructed business or personal data.  A
        # persistent plaintext cache is therefore opt-in, never the default.
        self._cache = LLMCache() if os.getenv("LLM_CACHE_ENABLED", "0") == "1" else None

    def complete(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)

    def chat(self, messages: list[dict]) -> str:
        cached = self._cache.get(self.model_name, self.temperature, self.seed, messages) if self._cache else None
        if cached is not None:
            logger.debug("LLM cache hit")
            return cached

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
        }
        if self.supports_temperature:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        if self._cache:
            self._cache.set(self.model_name, self.temperature, self.seed, messages, content)
        return content
