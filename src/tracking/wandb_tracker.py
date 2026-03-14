"""W&B artifact registry: datasets, fitted models, comparison tables. Optional — no-op if W&B unavailable.

LLM/agent tracing is handled by LangSmith (native LangGraph integration).
W&B is used exclusively for artifact versioning, metrics dashboard, and comparison tables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import get_logger

logger = get_logger(__name__)

_wandb = None
_run = None  # active wandb.Run


def _is_enabled() -> bool:
    return _run is not None


def init(project: str = "preprocessing-pipeline", run_name: str | None = None) -> bool:
    """Initialize W&B run (artifact registry + metrics). Returns True if successful."""
    global _wandb, _run

    load_dotenv()

    if not os.getenv("WANDB_API_KEY"):
        logger.info("W&B disabled: WANDB_API_KEY not set")
        return False

    try:
        import wandb
        _wandb = wandb
    except ImportError:
        logger.info("W&B disabled: wandb not installed")
        return False

    try:
        _run = wandb.init(project=project, name=run_name, reinit="finish_previous")
        logger.info("W&B initialized: project=%s, run=%s", project, _run.name)
        return True
    except Exception as e:
        logger.warning("W&B init failed: %s", str(e))
        _run = None
        return False


def finish():
    """Finalize the W&B run."""
    global _run
    if _run is not None:
        _run.finish()
        logger.info("W&B run finished")
        _run = None


def log_metrics(metrics: dict, step: str | None = None):
    """Log flat metrics to W&B. Prefix keys with step/ if step provided."""
    if not _is_enabled():
        return
    prefixed = {f"{step}/{k}" if step else k: v for k, v in metrics.items()
                if isinstance(v, (int, float))}
    _run.log(prefixed)


def log_summary(key: str, value):
    """Set a run summary value (shown in W&B dashboard columns)."""
    if not _is_enabled():
        return
    _run.summary[key] = value


def log_artifact(name: str, artifact_type: str, paths: list[str], metadata: dict | None = None):
    """Log files as a versioned W&B Artifact."""
    if not _is_enabled():
        return
    art = _wandb.Artifact(name, type=artifact_type, metadata=metadata or {})
    for p in paths:
        path = Path(p)
        if path.is_file():
            art.add_file(str(path))
        elif path.is_dir():
            art.add_dir(str(path))
    _run.log_artifact(art)
    logger.info("W&B artifact logged: %s (%s)", name, artifact_type)


def log_datasets(train_path: str, test_path: str, stage: str = "final"):
    """Log train/test CSVs as a versioned dataset artifact."""
    paths = [p for p in [train_path, test_path] if Path(p).is_file()]
    if paths:
        log_artifact(f"dataset-{stage}", "dataset", paths)


def log_fitted_artifacts(artifacts_dir: str = "data/outputs/artifacts"):
    """Log fitted scalers/encoders as a model artifact."""
    p = Path(artifacts_dir)
    if p.is_dir() and any(p.iterdir()):
        log_artifact("fitted-artifacts", "model", [str(p)])


def log_comparison_table(proposals: list[dict], executed: list[dict], table_name: str = "transform_comparison"):
    """Log a W&B Table comparing proposed vs executed actions."""
    if not _is_enabled():
        return

    columns = ["column", "proposed_action", "proposed_reason", "executed", "executed_action"]
    data = []

    executed_map = {e.get("column", ""): e for e in executed}

    for p in proposals:
        col = p.get("column", "")
        ex = executed_map.pop(col, None)
        data.append([
            col,
            p.get("action", ""),
            p.get("reason", ""),
            bool(ex),
            ex.get("action", "") if ex else "skipped",
        ])

    for col, ex in executed_map.items():
        data.append([col, "", "", True, ex.get("action", "")])

    table = _wandb.Table(columns=columns, data=data)
    _run.log({table_name: table})
    logger.info("W&B table logged: %s (%d rows)", table_name, len(data))


def log_outlier_table(proposals: list[dict], executed: list[dict]):
    """Log outlier proposals vs execution as a W&B Table."""
    log_comparison_table(proposals, executed, table_name="outlier_comparison")
