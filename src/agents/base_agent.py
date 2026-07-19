"""BaseAgent: W&B tracking, train/test split, dataset hash, target_meta."""

import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
from src.agents.state import PipelineState
from src.agents.dependencies import llm_client, prompt_templates
from src.tracking import wandb_tracker as tracker
from src.agents.schemas import DomainAndAnomaliesResponse, DiagnosisResponse
from src.handlers.error_handler import validate_llm_response

logger = get_logger(__name__)

NUMERIC_MATCH_THRESHOLD = 0.95
DATE_MATCH_THRESHOLD = 0.95
TARGET_NAN_BLOCK_THRESHOLD = 0.3


def _validate_target(df: pd.DataFrame, target: str) -> dict:
    """Validate the target column. Returns target_meta dict.

    Raises ValueError if target is invalid or has >30% NaN.
    """
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available: {list(df.columns)}"
        )

    nunique = df[target].nunique()
    dtype = df[target].dtype
    pct_nan = float(df[target].isnull().mean())

    # Block on excessive NaN
    if pct_nan > TARGET_NAN_BLOCK_THRESHOLD:
        raise ValueError(
            f"Target '{target}' has {pct_nan*100:.1f}% missing values "
            f"(>{TARGET_NAN_BLOCK_THRESHOLD*100:.0f}% threshold). "
            f"Clean the target column before running the pipeline."
        )

    # Infer task type
    if dtype in ("object", "category") or str(dtype) == "string":
        task_type = "classification"
    elif nunique <= 20:
        task_type = "classification"
    else:
        task_type = "regression"

    # Build target_meta
    target_meta = {
        "task_type": task_type,
        "nunique": int(nunique),
        "pct_nan": round(pct_nan * 100, 2),
    }

    if task_type == "classification":
        class_counts = df[target].value_counts()
        majority_ratio = float(class_counts.iloc[0] / class_counts.sum())
        target_meta["n_classes"] = len(class_counts)
        target_meta["imbalance_ratio"] = round(majority_ratio, 3)
        target_meta["class_distribution"] = {str(k): int(v) for k, v in class_counts.items()}
        if majority_ratio > 0.9:
            logger.warning("Target '%s' heavily imbalanced (%.1f%%)", target, majority_ratio * 100)
    else:
        target_meta["mean"] = round(float(df[target].mean()), 4)
        target_meta["std"] = round(float(df[target].std()), 4)

    logger.info("Target '%s': task_type=%s, %d unique, %.1f%% NaN", target, task_type, nunique, pct_nan * 100)
    return target_meta


def _audit_dtypes(df: pd.DataFrame, target: str) -> list[dict]:
    """Detect mistyped columns using the full column with strict 95% threshold."""
    audit = []
    for col in df.columns:
        if col == target:
            continue
        dtype = str(df[col].dtype)

        if dtype in ("object", "string"):
            non_null = df[col].dropna()
            n = len(non_null)
            if n == 0:
                continue

            # Vectorized numeric detection via pd.to_numeric
            numeric_count = int(pd.to_numeric(non_null, errors="coerce").notna().sum())
            ratio = numeric_count / n
            if ratio >= NUMERIC_MATCH_THRESHOLD:
                audit.append({
                    "column": col, "current_type": dtype, "suggested_type": "numeric",
                    "match_ratio": round(float(ratio), 3),
                    "reason": f"{numeric_count}/{n} numeric ({ratio:.1%})",
                })
                continue

            # Vectorized date detection via pd.to_datetime
            date_count = int(pd.to_datetime(non_null, errors="coerce").notna().sum())
            ratio = date_count / n
            if ratio >= DATE_MATCH_THRESHOLD:
                audit.append({
                    "column": col, "current_type": dtype, "suggested_type": "datetime",
                    "match_ratio": round(float(ratio), 3),
                    "reason": f"{date_count}/{n} date patterns ({ratio:.1%})",
                })
                continue

        if dtype in ("int64", "int32", "float64"):
            unique_vals = set(df[col].dropna().unique())
            if unique_vals <= {0, 1} or unique_vals <= {0.0, 1.0}:
                audit.append({
                    "column": col, "current_type": dtype, "suggested_type": "boolean",
                    "match_ratio": 1.0, "reason": "only values 0 and 1",
                })
            elif df[col].nunique(dropna=True) <= 10 and dtype in ("int64", "int32"):
                # Low-cardinality integer likely ordinal/categorical (e.g. rating 1-5)
                nunique = df[col].nunique(dropna=True)
                audit.append({
                    "column": col, "current_type": dtype, "suggested_type": "ordinal",
                    "match_ratio": 1.0,
                    "reason": f"integer with only {nunique} unique values, likely ordinal/categorical",
                })

    if audit:
        logger.info("Dtype audit: %d columns mistyped", len(audit))
    return audit


QUASI_CONSTANT_THRESHOLD = 0.995  # 99.5% same value


def _detect_trivial_columns(df: pd.DataFrame, target: str) -> list[dict]:
    """Detect auto-drop: 100% NaN, constant, quasi-constant, unique IDs, duplicates."""
    auto_drop = []
    feature_cols = [c for c in df.columns if c != target]

    for col in feature_cols:
        pct_nan = df[col].isnull().mean()
        if pct_nan == 1.0:
            auto_drop.append({"column": col, "reason": "100% missing values"})
        elif df[col].nunique(dropna=True) == 0:
            auto_drop.append({"column": col, "reason": "no distinct values"})
        elif df[col].nunique(dropna=True) == 1:
            auto_drop.append({"column": col, "reason": "constant column"})
        elif (
            df[col].nunique(dropna=True) == len(df)
            and str(df[col].dtype) in ("object", "string", "category")
        ):
            auto_drop.append({"column": col, "reason": "unique ID column"})
        else:
            # Quasi-constant: top value covers >= 99.5% of non-null values
            non_null = df[col].dropna()
            if len(non_null) > 0:
                top_freq = non_null.value_counts(normalize=True).iloc[0]
                if top_freq >= QUASI_CONSTANT_THRESHOLD:
                    auto_drop.append({
                        "column": col,
                        "reason": f"quasi-constant ({top_freq*100:.1f}% same value)",
                    })

    # Duplicate columns (using .equals for float robustness)
    already_dropped = {d["column"] for d in auto_drop}
    remaining = [c for c in feature_cols if c not in already_dropped]
    for i, col_a in enumerate(remaining):
        for col_b in remaining[i + 1:]:
            if col_b not in already_dropped and df[col_a].equals(df[col_b]):
                auto_drop.append({"column": col_b, "reason": f"duplicate of '{col_a}'"})
                already_dropped.add(col_b)

    if auto_drop:
        logger.info("Auto-drop: %s", [d["column"] for d in auto_drop])
    return auto_drop


def _build_col_summary(df: pd.DataFrame, target: str, auto_drop_names: set) -> dict:
    """Build column summary with skewness and outlier_ratio for numerics."""
    col_summary = {}
    feature_cols = [c for c in df.columns if c != target]
    for col in feature_cols:
        if col in auto_drop_names:
            continue
        info = {"type": str(df[col].dtype)}
        pct_nan = df[col].isnull().mean()
        if pct_nan > 0:
            info["pct_nan"] = round(float(pct_nan) * 100, 1)
        if str(df[col].dtype) == "bool":
            info["cardinality"] = 2
            info["semantic_type"] = "binary_categorical"
        elif df[col].dtype in ("object", "category") or str(df[col].dtype) == "string":
            cardinality = int(df[col].nunique())
            info["cardinality"] = cardinality
            info["semantic_type"] = "binary_categorical" if cardinality == 2 else "nominal_categorical"
        elif pd.api.types.is_numeric_dtype(df[col]):
            clean = df[col].dropna().astype(float)
            nunique = int(df[col].nunique())
            col_min = clean.min() if len(clean) > 0 else None
            col_max = clean.max() if len(clean) > 0 else None
            info["min"] = round(float(col_min), 4) if pd.notna(col_min) else None
            info["max"] = round(float(col_max), 4) if pd.notna(col_max) else None
            info["nunique"] = nunique
            # Semantic type for numeric columns
            unique_vals = set(clean.unique())
            if unique_vals <= {0.0, 1.0}:
                info["semantic_type"] = "binary_categorical"
            elif nunique <= 10:
                info["semantic_type"] = "ordinal_or_discrete"
            else:
                info["semantic_type"] = "continuous"
            # Skewness
            if len(clean) > 2:
                info["skewness"] = round(float(clean.skew()), 3)
            # Outlier ratio (IQR-based)
            if len(clean) > 4:
                q1 = clean.quantile(0.25)
                q3 = clean.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outlier_mask = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
                    info["outlier_ratio"] = round(float(outlier_mask.mean()) * 100, 1)
        col_summary[col] = info
    return col_summary


MAX_MISSING_PATTERN_COLS = 80  # Cap to avoid O(n^2) blowup on wide datasets


def _detect_missing_patterns(df: pd.DataFrame, target: str, skip_cols: set) -> list[dict]:
    """Detect correlated missingness between feature columns (potential MAR/MNAR).

    Run once in analyze() and stored in df_info — avoids O(n^2) recomputation on reject.
    Capped at MAX_MISSING_PATTERN_COLS columns to keep runtime tractable.
    """
    patterns = []
    cols_with_nan = [c for c in df.columns if c not in skip_cols and c != target and df[c].isnull().any()]
    if len(cols_with_nan) < 2:
        return patterns
    if len(cols_with_nan) > MAX_MISSING_PATTERN_COLS:
        logger.warning(
            "Missing pattern detection: %d columns with NaN (cap=%d), "
            "keeping top %d by NaN ratio",
            len(cols_with_nan), MAX_MISSING_PATTERN_COLS, MAX_MISSING_PATTERN_COLS,
        )
        # Keep the columns with most NaN — most likely to show correlated patterns
        cols_with_nan = sorted(cols_with_nan, key=lambda c: df[c].isnull().mean(), reverse=True)[:MAX_MISSING_PATTERN_COLS]
    missing_matrix = df[cols_with_nan].isnull().astype(int)
    for i, col_a in enumerate(cols_with_nan):
        for col_b in cols_with_nan[i + 1:]:
            corr = missing_matrix[col_a].corr(missing_matrix[col_b])
            if abs(corr) > 0.5:
                patterns.append({"col_a": col_a, "col_b": col_b, "correlation": round(float(corr), 3)})
                logger.warning(
                    "Missing pattern: '%s' and '%s' have correlated missingness (r=%.2f) -- "
                    "potential MAR/MNAR, mean/median imputation may introduce bias",
                    col_a, col_b, corr,
                )
    return patterns


def analyze(state: PipelineState) -> dict:
    """Load dataset, validate target, audit, infer domain+anomalies (1 LLM call), diagnose (1 LLM call)."""
    tracker.init(run_name=Path(state["df_path"]).stem)
    df = pd.read_csv(state["df_path"])
    target = state["target"]
    logger.info("Dataset loaded: %d rows x %d columns, target='%s'", df.shape[0], df.shape[1], target)

    target_meta = _validate_target(df, target)
    task_type = target_meta["task_type"]
    dtype_audit = _audit_dtypes(df, target)
    auto_drop = _detect_trivial_columns(df, target)
    auto_drop_names = {d["column"] for d in auto_drop}
    col_summary = _build_col_summary(df, target, auto_drop_names)
    dtype_audit_cols = {d["column"] for d in dtype_audit}
    missing_patterns = _detect_missing_patterns(df, target, auto_drop_names | dtype_audit_cols)

    # LLM call 1: send schema/statistics only. Raw rows can contain identifiers,
    # health data or secrets and are never required for this semantic task.
    sample = "Raw row values withheld by data-minimisation policy."
    system_prompt = prompt_templates.get_template("system_prompt")

    logger.info("Inferring domain context and anomalies (1 LLM call)")
    domain_prompt = prompt_templates.format_template(
        "infer_domain_and_anomalies",
        columns=list(df.columns), sample=sample,
        target=target, task_type=task_type, col_summary=str(col_summary),
    )
    domain_response = validate_llm_response(llm_client, domain_prompt, system_prompt, DomainAndAnomaliesResponse)
    domain_context = {
        "domain": domain_response["domain"],
        "description": domain_response["description"],
        "constraints": domain_response["constraints"],
        "sensitive_columns": domain_response["sensitive_columns"],
    }
    semantic_anomalies = domain_response.get("anomalies", [])

    # LLM call 2: diagnosis
    logger.info("Generating diagnosis (1 LLM call)")
    analysis_prompt = prompt_templates.format_template(
        "analyze_dataset",
        col_summary=str(col_summary),
        shape=f"{df.shape[0]} rows x {df.shape[1]} columns",
        auto_drop=str(auto_drop) if auto_drop else "Aucune",
        target=target, task_type=task_type,
        domain_context=f"{domain_context['domain']} - {domain_context['description']}",
        dtype_audit=str(dtype_audit) if dtype_audit else "Aucun",
    )
    diagnosis = validate_llm_response(llm_client, analysis_prompt, system_prompt, DiagnosisResponse)

    df_info = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": f"{df.shape[0]} rows x {df.shape[1]} columns",
        "missing_values": {col: int(v) for col, v in df.isnull().sum().items()},
        "col_summary": col_summary,
        "auto_drop": auto_drop,
        "diagnosis": diagnosis,
        "semantic_anomalies": semantic_anomalies,
        "target_meta": target_meta,
        "missing_patterns": missing_patterns,
    }

    # Derive stable dataset name from original path
    raw = state["df_path"]
    if urlparse(raw).scheme in ("http", "https"):
        dataset_name = Path(urlparse(raw).path).stem
    else:
        dataset_name = Path(raw).stem

    return {
        "df_info": df_info,
        "original_df_path": state["df_path"],
        "dataset_name": dataset_name,
        "task_type": task_type,
        "target_meta": target_meta,
        "domain_context": domain_context,
        "dtype_audit": dtype_audit,
        "proposals": {"semantic_anomalies": semantic_anomalies},
        "steps_done": [{"step": "analyze", "result": diagnosis, "confidence": "high", "source": "deterministic+llm"}],
    }


SPLIT_TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42


def split_data(state: PipelineState) -> dict:
    """Split dataset into train/test.

    Strategy is controlled by state["split_config"]:
      - "auto"       : stratified for classification, random for regression (default)
      - "stratified" : always stratified on target
      - "temporal"   : sort by time_col, take last test_size rows as test
      - "group"      : GroupShuffleSplit using group_col (no group leaks between train/test)

    All subsequent fit operations use train only; transform applies to both.
    """
    from sklearn.model_selection import GroupShuffleSplit

    df = pd.read_csv(state["df_path"])
    target = state["target"]
    task_type = state.get("task_type", "regression")

    cfg = state.get("split_config") or {}
    test_size = float(cfg.get("test_size", SPLIT_TEST_SIZE))
    random_state = int(cfg.get("random_state", SPLIT_RANDOM_STATE))
    strategy = cfg.get("strategy", "auto")
    time_col = cfg.get("time_col")
    group_col = cfg.get("group_col")

    # Dataset hash for traceability
    dataset_hash = hashlib.md5(df.to_csv(index=False).encode()).hexdigest()
    logger.info("Dataset hash (MD5): %s", dataset_hash)

    actual_strategy = strategy
    if strategy == "auto":
        actual_strategy = "stratified" if task_type == "classification" else "random"

    if actual_strategy == "temporal":
        if not time_col or time_col not in df.columns:
            logger.warning("time_col '%s' not found, falling back to random split", time_col)
            actual_strategy = "random"
        else:
            df = df.sort_values(time_col).reset_index(drop=True)
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            logger.info("Temporal split on '%s': %d train, %d test", time_col, len(train_df), len(test_df))

    if actual_strategy == "group":
        if not group_col or group_col not in df.columns:
            logger.warning("group_col '%s' not found, falling back to random split", group_col)
            actual_strategy = "random"
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            logger.info("Group split on '%s': %d train, %d test", group_col, len(train_df), len(test_df))

    if actual_strategy in ("stratified", "random"):
        stratify = df[target] if actual_strategy == "stratified" else None
        try:
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=stratify,
            )
        except ValueError:
            logger.warning("Stratified split failed, falling back to random split")
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state,
            )
            actual_strategy = "random"

    # Save splits
    raw_path = state["df_path"]
    if urlparse(raw_path).scheme in ("http", "https"):
        stem = Path(urlparse(raw_path).path).stem
        out_dir = Path("data/outputs")
    else:
        p = Path(raw_path)
        stem = p.stem
        out_dir = p.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = str(out_dir / f"{stem}_train.csv")
    test_path = str(out_dir / f"{stem}_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Split [%s]: %d train, %d test (%.0f%%) -> %s, %s",
                actual_strategy, len(train_df), len(test_df), test_size * 100, train_path, test_path)

    return {
        "df_path": train_path,
        "train_path": train_path,
        "test_path": test_path,
        "steps_done": [{
            "step": "split_data",
            "train_size": len(train_df),
            "test_size": len(test_df),
            "strategy": actual_strategy,
            "test_size_ratio": test_size,
            "random_state": random_state,
            "dataset_hash": dataset_hash,
        }],
    }
