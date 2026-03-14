"""Configuration package."""

import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent


def load_yaml_config(filename: str) -> dict:
    """Load a YAML config file from the config directory."""
    filepath = CONFIG_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
