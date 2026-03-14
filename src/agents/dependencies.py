"""Shared dependencies for all agents (singleton instances)."""

from src.llm.gpt_client import GPTClient
from src.prompt_engineering.templates import PromptTemplateManager

llm_client = GPTClient()
prompt_templates = PromptTemplateManager()
