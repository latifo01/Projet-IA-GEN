"""OutlierAgent: W&B tracking, train-fitted bounds, D'Agostino-Pearson, bimodality check."""

import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from scipy import stats as scipy_stats
from src.utils.logger import get_logger
from src.agents.state import PipelineState
from src.agents.dependencies import llm_client, prompt_templates
from src.agents.schemas import OutlierResponse
from src.handlers.error_handler import validate_llm_response
from src.tracking import wandb_tracker as tracker

logger = get_logger(__name__)

DAGOSTINO_MIN_ROWS = 20
SKEWNESS_THRESHOLD = 1.0
HIGH_OUTLIER_PCT = 5.0
LOW_OUTLIER_PCT = 1.0
BIMODALITY_THRESHOLD = 0.555  # Sarle's bimodality coefficient threshold
IMBALANCE_GUARD_THRESHOLD = 0.85  # Majority ratio above which minority class is protected


def _bimodality_coefficient(series: pd.Series) -> float:
    """Sarle's bimodality coefficient: (skew^2 + 1) / kurtosis.

    Values > 0.555 suggest bimodal distribution.
    """
    clean = series.dropna()
    if len(clean) < 4:
        return 0.0
    skew = clean.skew()
    kurt = clean.kurtosis()
    # Excess kurtosis from pandas; need kurtosis + 3 for Sarle's formula
    k = kurt + 3
    if k == 0:
        return 0.0
    return float((skew ** 2 + 1) / k)


def _choose_method(series: pd.Series) -> tuple[str, str]:
    """Choose outlier detection method based on distribution.

    Uses D'Agostino-Pearson (works on any sample size >= 20).
    Falls back to skewness heuristic for small samples.
    If bimodal, returns 'bimodal' to signal IQR is inappropriate.

    Returns (method, reason) tuple for audit traceability.
    """
    clean = series.dropna()
    if len(clean) < 8:
        return "iqr", f"sample too small (n={len(clean)} < 8), defaulting to IQR"

    # Check bimodality first
    bc = _bimodality_coefficient(clean)
    if bc > BIMODALITY_THRESHOLD:
        logger.info("Column appears bimodal (BC=%.3f), outlier detection unreliable", bc)
        return "bimodal", f"bimodal distribution detected (BC={bc:.3f})"

    # D'Agostino-Pearson test (needs n >= 20)
    if len(clean) >= DAGOSTINO_MIN_ROWS:
        try:
            _, p_value = scipy_stats.normaltest(clean)
            if p_value > 0.05:
                return "zscore", f"D'Agostino-Pearson normal (p={p_value:.4f})"
            return "iqr", f"D'Agostino-Pearson non-normal (p={p_value:.4f})"
        except Exception:
            pass

    # Fallback: skewness heuristic
    skew = abs(clean.skew())
    if skew < SKEWNESS_THRESHOLD:
        return "zscore", f"low skewness ({skew:.2f}), approximately normal"
    return "iqr", f"high skewness ({skew:.2f})"


def _choose_treatment(pct_outliers: float, task_type: str, col: str,
                      semantic_cols: set, imbalance_ratio: float = 0.0) -> tuple[str, str, str]:
    """Deterministic treatment decision based on outlier percentage and context.

    Returns (action, confidence, reason).
    """
    # Semantic anomaly columns -> replace_with_nan (treat as bad data)
    if col in semantic_cols:
        return "replace_with_nan", "high", "semantic anomaly detected, replacing with NaN for re-imputation"

    # Imbalanced classification: never remove outliers (could destroy minority class)
    if task_type == "classification" and imbalance_ratio > IMBALANCE_GUARD_THRESHOLD:
        if pct_outliers < HIGH_OUTLIER_PCT:
            return "clip", "high", f"imbalanced dataset (ratio={imbalance_ratio:.2f}), clipping to preserve minority class"
        return "keep", "medium", f"{pct_outliers:.1f}% outliers in imbalanced dataset, keeping to preserve minority class"

    # Very few outliers in classification -> remove (minimal data loss)
    if pct_outliers < LOW_OUTLIER_PCT and task_type == "classification":
        return "remove", "high", f"<{LOW_OUTLIER_PCT}% outliers in classification, safe to remove"

    # Very few outliers in regression -> clip (preserve data points)
    if pct_outliers < LOW_OUTLIER_PCT and task_type == "regression":
        return "clip", "high", f"<{LOW_OUTLIER_PCT}% outliers in regression, clip to preserve data"

    # Moderate outliers -> clip
    if pct_outliers < HIGH_OUTLIER_PCT:
        return "clip", "medium", f"{pct_outliers:.1f}% outliers, clipping"

    # High outlier percentage -> keep (likely a distribution property, not errors)
    return "keep", "low", f"{pct_outliers:.1f}% outliers (high), likely distribution property"


def _detect_outliers_per_column(df: pd.DataFrame, target: str) -> list[dict]:
    """Outlier detection per numeric column with method selection."""
    report = []
    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target]
    for col in num_cols:
        method, method_reason = _choose_method(df[col])

        # Bimodal: IQR/zscore unreliable, report 0 outliers with 'bimodal' method
        if method == "bimodal":
            report.append({
                "column": col, "method": "bimodal", "method_reason": method_reason,
                "n_outliers": 0, "pct_outliers": 0.0,
                "lower_bound": None, "upper_bound": None,
                "min": round(float(df[col].min()), 4) if pd.notna(df[col].min()) else None,
                "max": round(float(df[col].max()), 4) if pd.notna(df[col].max()) else None,
            })
            continue

        if method == "iqr":
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            mean, std = df[col].mean(), df[col].std()
            lower, upper = mean - 3 * std, mean + 3 * std

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = int(outlier_mask.sum())
        pct = round(n_outliers / len(df) * 100, 1)

        report.append({
            "column": col, "method": method, "method_reason": method_reason,
            "n_outliers": n_outliers, "pct_outliers": pct,
            "lower_bound": round(float(lower), 4), "upper_bound": round(float(upper), 4),
            "min": round(float(df[col].min()), 4) if pd.notna(df[col].min()) else None,
            "max": round(float(df[col].max()), 4) if pd.notna(df[col].max()) else None,
        })
    return report


def propose_outlier_treatment(state: PipelineState) -> dict:
    """Detect outliers on train set only. Deterministic treatment first, LLM for ambiguous."""
    train_df = pd.read_csv(state["train_path"])
    target = state["target"]
    task_type = state.get("task_type", "regression")
    domain_context = state.get("domain_context", {})
    target_meta = state.get("target_meta", {})
    imbalance_ratio = target_meta.get("imbalance_ratio", 0.0)

    # Consume semantic_anomalies — exclude columns already sentinel-replaced in transformations
    semantic_anomalies = state.get("df_info", {}).get("semantic_anomalies", [])
    already_replaced = set()
    for step in state.get("steps_done", []):
        if step.get("step") == "transformations":
            already_replaced = {a["column"] for a in step.get("applied", []) if a["action"] == "replace_sentinel"}
            break
    semantic_cols = {a["column"] for a in semantic_anomalies
                     if a.get("type") in ("proxy_for_missing", "impossible_value")
                     } - already_replaced

    outlier_report = _detect_outliers_per_column(train_df, target)
    cols_with_outliers = [r for r in outlier_report if r["n_outliers"] > 0]
    # Bimodal columns: forced to 'keep' with high confidence
    bimodal_cols = [r for r in outlier_report if r["method"] == "bimodal"]

    if not cols_with_outliers and not bimodal_cols:
        logger.info("No outliers detected")
        return {"proposals": {"outlier_actions": [], "outlier_report": outlier_report}}

    # Deterministic treatment
    deterministic_actions = []
    ambiguous_cols = []

    # Bimodal -> always keep
    for r in bimodal_cols:
        deterministic_actions.append({
            "column": r["column"], "method": "bimodal", "action": "keep",
            "reason": "bimodal distribution detected, IQR/zscore unreliable",
            "confidence": "high", "source": "rule",
        })

    for r in cols_with_outliers:
        col = r["column"]
        action, confidence, reason = _choose_treatment(
            r["pct_outliers"], task_type, col, semantic_cols, imbalance_ratio,
        )

        if confidence == "high":
            deterministic_actions.append({
                "column": col, "method": r["method"], "action": action,
                "reason": reason, "confidence": confidence, "source": "rule",
            })
        elif confidence == "low":
            # Low confidence "keep" — flag for human review
            deterministic_actions.append({
                "column": col, "method": r["method"], "action": action,
                "reason": reason, "confidence": confidence, "source": "rule",
                "requires_review": True,
            })
        else:
            ambiguous_cols.append(r)

    # LLM for ambiguous outliers only
    llm_actions = []
    if ambiguous_cols:
        human_feedback = state.get("human_feedback", "")
        feedback_block = (
            f"\n\nFEEDBACK UTILISATEUR (a integrer dans ta reponse) :\n{human_feedback}"
            if human_feedback else ""
        )

        prompt = prompt_templates.format_template(
            "propose_outlier_treatment",
            outlier_report=str(ambiguous_cols),
            domain=domain_context.get("domain", "unknown"),
            constraints=str(domain_context.get("constraints", [])),
            target=target,
        ) + feedback_block
        system_prompt = prompt_templates.get_template("system_prompt")
        logger.info("LLM for %d ambiguous outlier columns", len(ambiguous_cols))
        proposals = validate_llm_response(llm_client, prompt, system_prompt, OutlierResponse)
        llm_actions = proposals.get("outlier_actions", [])

    all_actions = deterministic_actions + llm_actions
    logger.info("Outlier proposals: %d deterministic, %d LLM", len(deterministic_actions), len(llm_actions))

    return {
        "proposals": {"outlier_actions": all_actions, "outlier_report": outlier_report},
    }


def execute_outlier_treatment(state: PipelineState) -> dict:
    """Execute outlier treatments. Bounds fitted on train, applied to both train and test."""
    train_df = pd.read_csv(state["train_path"])
    test_df = pd.read_csv(state["test_path"])
    proposals = state["proposals"]
    actions = proposals.get("outlier_actions", [])
    outlier_report = proposals.get("outlier_report", [])
    applied = []
    confidence_entries = []

    for a in actions:
        col = a["column"]
        method = a["method"]
        action = a["action"]

        if col == state["target"]:
            continue
        if col not in train_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            continue

        if action == "keep" or method == "bimodal":
            applied.append({"column": col, "method": method, "action": "keep", "outliers_found": 0})
            confidence_entries.append({
                "step": "outlier", "column": col, "action": "keep",
                "confidence": a.get("confidence", "medium"), "source": a.get("source", "llm"),
            })
            continue

        try:
            # Bounds computed on TRAIN only
            if method == "iqr":
                q1, q3 = train_df[col].quantile(0.25), train_df[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            elif method == "zscore":
                mean, std = train_df[col].mean(), train_df[col].std()
                lower, upper = mean - 3 * std, mean + 3 * std
            else:
                logger.warning("Unknown method '%s' for '%s'", method, col)
                continue

            train_outlier_count = int(((train_df[col] < lower) | (train_df[col] > upper)).sum())
            test_clipped = False

            # Apply treatment to both train and test using train-fitted bounds
            for frame_name, frame in [("train", train_df), ("test", test_df)]:
                if col not in frame.columns:
                    continue
                outlier_mask = (frame[col] < lower) | (frame[col] > upper)
                # Add _was_outlier flag before modifying values
                if action in ("clip", "replace_with_nan"):
                    flag_col = f"{col}_was_outlier"
                    frame[flag_col] = outlier_mask.astype(int)
                if action == "clip":
                    frame[col] = frame[col].clip(lower=lower, upper=upper)
                elif action == "remove":
                    if frame_name == "train":
                        train_df = frame[~outlier_mask]
                    else:
                        # On test: clip instead of removing rows (tracked separately)
                        frame[col] = frame[col].clip(lower=lower, upper=upper)
                        test_clipped = True
                elif action == "replace_with_nan":
                    frame.loc[outlier_mask, col] = np.nan

            entry = {"column": col, "method": method, "action": action,
                     "outliers_found": train_outlier_count}
            if test_clipped and action == "remove":
                entry["test_action"] = "clip"
            applied.append(entry)
            confidence_entries.append({
                "step": "outlier", "column": col, "action": action,
                "confidence": a.get("confidence", "medium"), "source": a.get("source", "llm"),
            })
            logger.info("%s/%s on '%s': %d outliers (bounds fitted on train)", method, action, col, train_outlier_count)

        except Exception as e:
            logger.warning("Failed on '%s': %s", col, str(e))

    # Save train/test — derive names from dataset_name for stability across rejects
    raw_path = state["train_path"]
    original_name = state.get("dataset_name", "")
    if original_name:
        base_stem = Path(original_name).stem
        p = Path(raw_path)
        out_dir = p.parent
    elif urlparse(raw_path).scheme in ("http", "https"):
        base_stem = Path(urlparse(raw_path).path).stem
        out_dir = Path("data/outputs")
    else:
        p = Path(raw_path)
        base_stem = p.stem
        for suffix in ("_train_transformed", "_train"):
            if base_stem.endswith(suffix):
                base_stem = base_stem[: -len(suffix)]
                break
        out_dir = p.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = str(out_dir / f"{base_stem}_train_clean.csv")
    test_out = str(out_dir / f"{base_stem}_test_clean.csv")
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    logger.info("Clean datasets: %s, %s", train_out, test_out)

    # W&B: outlier proposals vs executed comparison table
    tracker.log_outlier_table(actions, applied)

    return {
        "df_path": train_out,
        "train_path": train_out,
        "test_path": test_out,
        "confidence_map": confidence_entries,
        "steps_done": [{
            "step": "outlier_treatment", "applied": applied,
            "outlier_report": outlier_report, "new_shape": str(train_df.shape),
        }],
    }
