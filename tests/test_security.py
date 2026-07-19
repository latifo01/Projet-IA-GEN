from pathlib import Path

import pytest

from src.agents.dependencies import LazyLLMClient
from src.security.input_sources import validate_dataset_source


class FakeLLM:
    def complete(self, prompt: str, system_prompt: str = "") -> str:
        return "offline"

    def chat(self, messages: list[dict]) -> str:
        return "offline-chat"


def test_llm_dependency_can_be_injected_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = LazyLLMClient()
    provider.configure(FakeLLM())
    assert provider.complete("hello") == "offline"


def test_local_source_is_restricted_to_configured_root(tmp_path, monkeypatch):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    csv_path = safe_root / "tiny.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")
    monkeypatch.setenv("DATASET_ALLOWED_ROOTS", str(safe_root))
    assert validate_dataset_source(str(csv_path)) == str(csv_path.resolve())

    outside = tmp_path / "outside.csv"
    outside.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        validate_dataset_source(str(outside))


def test_url_requires_https_allowlist(monkeypatch):
    monkeypatch.setenv("DATASET_ALLOWED_HOSTS", "data.example.org")
    assert validate_dataset_source("https://data.example.org/sample.csv").startswith("https://")
    with pytest.raises(ValueError, match="allow-listed"):
        validate_dataset_source("http://data.example.org/sample.csv")
    with pytest.raises(ValueError, match="not in"):
        validate_dataset_source("https://internal.example.org/sample.csv")
