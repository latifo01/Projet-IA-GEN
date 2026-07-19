"""Validate dataset sources before pandas can access them.

Local API access is restricted to configured roots. HTTPS sources are disabled
unless their hostname is explicitly allow-listed. This prevents an exposed API
from becoming an arbitrary file reader or SSRF primitive.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

MAX_DATASET_BYTES = int(os.getenv("MAX_DATASET_BYTES", str(25 * 1024 * 1024)))


def _allowed_roots() -> tuple[Path, ...]:
    configured = os.getenv("DATASET_ALLOWED_ROOTS", "data")
    return tuple(Path(value.strip()).resolve() for value in configured.split(os.pathsep) if value.strip())


def _allowed_hosts() -> set[str]:
    return {value.strip().lower() for value in os.getenv("DATASET_ALLOWED_HOSTS", "").split(",") if value.strip()}


def validate_dataset_source(raw: str) -> str:
    """Return a normalised safe source or raise ``ValueError``."""
    # ``urlparse('C:\\data.csv')`` interprets ``C`` as a URL scheme. Only an
    # explicit scheme separator enters the remote-source branch.
    parsed = urlparse(raw)
    if "://" in raw:
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("Only explicitly allow-listed HTTPS dataset URLs are accepted.")
        if parsed.hostname.lower() not in _allowed_hosts():
            raise ValueError("Dataset URL host is not in DATASET_ALLOWED_HOSTS.")
        if not parsed.path.lower().endswith(".csv"):
            raise ValueError("Dataset URL must point to a CSV file.")
        return raw

    candidate = Path(raw).resolve()
    if candidate.suffix.lower() != ".csv":
        raise ValueError("Dataset must be a CSV file.")
    if not any(candidate == root or root in candidate.parents for root in _allowed_roots()):
        raise ValueError("Dataset path is outside DATASET_ALLOWED_ROOTS.")
    if not candidate.is_file():
        raise ValueError("Dataset file does not exist.")
    if candidate.stat().st_size > MAX_DATASET_BYTES:
        raise ValueError(f"Dataset exceeds the {MAX_DATASET_BYTES}-byte limit.")
    return str(candidate)
