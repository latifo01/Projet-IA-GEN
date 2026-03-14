"""Template management for prompts."""

from pathlib import Path
import yaml


class PromptTemplateManager:
    """Load and format prompt templates from YAML configuration."""

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "prompt_templates.yaml"
        else:
            config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self.templates = yaml.safe_load(f)

    def get_version(self) -> str:
        """Return the prompt templates version declared in the YAML."""
        return str(self.templates.get("version", "unknown"))

    def get_template(self, name: str) -> str:
        """Get a raw template by name."""
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found. Available: {list(self.templates.keys())}")
        return self.templates[name]

    def format_template(self, name: str, **kwargs) -> str:
        """Get a template and format it with variables."""
        template = self.get_template(name)
        return template.format(**kwargs)
