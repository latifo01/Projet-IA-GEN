"""TransformationAgent : W&B tracking, train/test split-aware, deferred scaling."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, RobustScaler, OneHotEncoder
from src.utils.logger import get_logger
from src.agents.state import PipelineState
from src.agents.dependencies import llm_client, prompt_templates
from src.agents.schemas import TransformResponse, FeatureEngineeringResponse
from src.handlers.error_handler import validate_llm_response
from src.tracking import wandb_tracker as tracker
from src.agents.base_agent import _build_col_summary, _detect_trivial_columns

logger = get_logger(__name__)

MISSING_DROP_THRESHOLD = 0.5
MISSING_LOW_THRESHOLD = 0.05
HIGH_CARDINALITY_THRESHOLD = 50
LOW_CARDINALITY_THRESHOLD = 10
ORDINAL_NUMERIC_THRESHOLD = 10
LEAKAGE_CORRELATION_THRESHOLD = 0.95
HIGH_CORRELATION_THRESHOLD = 0.75
VERY_HIGH_CORRELATION_THRESHOLD = 0.90

# Scaling actions are deferred to post-outlier phase
_SCALING_ACTIONS = {"standard_scaling", "minmax_scaling", "robust_scaling", "log_transform", "sqrt_transform"}

_ACTION_PRIORITY = {
    "replace_sentinel": -1, "cast_numeric": -1,
    "impute_mean": 0, "impute_median": 0, "impute_mode": 0, "drop_rows": 0, "drop_column": 0,
    "one_hot_encoding": 1, "label_encoding": 1, "ordinal_encoding": 1, "frequency_encoding": 1, "extract_datetime": 1,
    "standard_scaling": 2, "minmax_scaling": 2, "robust_scaling": 2, "log_transform": 2, "sqrt_transform": 2,
}


def _dtype_audit_transforms(dtype_audit: list[dict]) -> list[dict]:
    """Generate deterministic transforms from BaseAgent's dtype audit."""
    transforms = []
    for entry in dtype_audit:
        col = entry["column"]
        suggested = entry["suggested_type"]
        if suggested == "datetime":
            transforms.append({
                "column": col, "action": "extract_datetime",
                "reason": f"dtype audit: {entry['reason']}",
                "confidence": "high", "source": "rule",
            })
        elif suggested in ("numeric", "int", "float"):
            transforms.append({
                "column": col, "action": "cast_numeric",
                "reason": f"dtype audit: {entry['reason']}",
                "confidence": "high", "source": "rule",
            })
    return transforms


def _semantic_anomaly_transforms(semantic_anomalies: list[dict], df: pd.DataFrame) -> list[dict]:
    """Generate deterministic replace_sentinel transforms from semantic anomalies.

    Runs BEFORE imputation so sentinel values become NaN and get properly imputed.
    """
    transforms = []
    for anomaly in semantic_anomalies:
        col = anomaly.get("column", "")
        anomaly_type = anomaly.get("type", "other")
        if col not in df.columns:
            continue
        if anomaly_type == "proxy_for_missing":
            sentinel_values = anomaly.get("sentinel_values", [])
            if not sentinel_values and pd.api.types.is_numeric_dtype(df[col]):
                # Heuristic: detect common sentinel patterns in numeric columns
                candidates = {-1, -999, 999, 9999, -9999}
                observed = set(df[col].dropna().unique())
                sentinel_values = sorted(candidates & observed)
            if sentinel_values:
                transforms.append({
                    "column": col, "action": "replace_sentinel",
                    "reason": f"proxy_for_missing: sentinel values {sentinel_values}",
                    "confidence": "high", "source": "rule",
                    "sentinel_values": sentinel_values,
                })
        elif anomaly_type == "impossible_value":
            sentinel_values = anomaly.get("sentinel_values", [])
            if sentinel_values:
                transforms.append({
                    "column": col, "action": "replace_sentinel",
                    "reason": f"impossible_value: replace {sentinel_values} with NaN",
                    "confidence": "high", "source": "rule",
                    "sentinel_values": sentinel_values,
                })
    return transforms


def _rule_based_decisions(df: pd.DataFrame, target: str, auto_drop: list[dict],
                          col_summary: dict | None = None) -> tuple[list[dict], list[str]]:
    """Deterministic rules for obvious transformations. Returns (transforms, ambiguous_cols)."""
    transforms = []
    resolved_cols = set()
    skip_cols = {d["column"] for d in auto_drop} | {target}
    if col_summary is None:
        col_summary = {}

    for col in df.columns:
        if col in skip_cols:
            continue

        dtype = str(df[col].dtype)
        pct_nan = df[col].isnull().mean()
        nunique = df[col].nunique(dropna=True)

        # >50% NaN -> drop
        if pct_nan > MISSING_DROP_THRESHOLD:
            transforms.append({
                "column": col, "action": "drop_column",
                "reason": f"{pct_nan*100:.0f}% NaN (>{MISSING_DROP_THRESHOLD*100:.0f}%)",
                "confidence": "high", "source": "rule",
            })
            resolved_cols.add(col)
            continue

        # Categorical rules
        if dtype in ("object", "category") or str(dtype) == "string":
            if pct_nan > 0:
                transforms.append({
                    "column": col, "action": "impute_mode",
                    "reason": f"categorical {pct_nan*100:.1f}% NaN",
                    "confidence": "high", "source": "rule",
                })
            if nunique == 2:
                transforms.append({
                    "column": col, "action": "ordinal_encoding",
                    "reason": "binary categorical (2 classes)",
                    "confidence": "high", "source": "rule",
                })
                resolved_cols.add(col)
            elif nunique <= LOW_CARDINALITY_THRESHOLD:
                transforms.append({
                    "column": col, "action": "one_hot_encoding",
                    "reason": f"low cardinality ({nunique})",
                    "confidence": "high", "source": "rule",
                })
                resolved_cols.add(col)
            elif nunique > HIGH_CARDINALITY_THRESHOLD:
                transforms.append({
                    "column": col, "action": "frequency_encoding",
                    "reason": f"high cardinality ({nunique})",
                    "confidence": "high", "source": "rule",
                })
                resolved_cols.add(col)
            else:
                continue  # 10-50 unique -> ambiguous

        # Boolean -> ordinal encode to 0/1
        elif str(df[col].dtype) == "bool":
            transforms.append({
                "column": col, "action": "ordinal_encoding",
                "reason": "boolean column -> int",
                "confidence": "high", "source": "rule",
            })
            resolved_cols.add(col)
            continue

        # Numeric rules
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Imputation
            if pct_nan > 0 and pct_nan <= MISSING_LOW_THRESHOLD:
                transforms.append({
                    "column": col, "action": "impute_median",
                    "reason": f"numeric {pct_nan*100:.1f}% NaN (<{MISSING_LOW_THRESHOLD*100:.0f}%)",
                    "confidence": "high", "source": "rule",
                })
            elif pct_nan > MISSING_LOW_THRESHOLD:
                continue  # ambiguous

            # Ordinal guard: only if value range is small relative to nunique
            value_range = float(df[col].astype(float).max() - df[col].astype(float).min()) if df[col].notna().any() else 0
            if nunique <= ORDINAL_NUMERIC_THRESHOLD and nunique >= 2 and value_range <= nunique * 2:
                transforms.append({
                    "column": col, "action": "ordinal_encoding",
                    "reason": f"numeric with {nunique} unique values, range={value_range:.0f} (<= {nunique*2})",
                    "confidence": "medium", "source": "rule",
                })
            else:
                # Scaling for continuous numeric (deferred to post-outlier phase)
                skew = abs(float(df[col].skew())) if df[col].notna().sum() > 2 else 0
                outlier_ratio = col_summary.get(col, {}).get("outlier_ratio", 0)
                if skew > 1.0 and outlier_ratio < 2.0:
                    # Skewed with few/no outliers: log-normal shape, log_transform preferred
                    transforms.append({
                        "column": col, "action": "log_transform",
                        "reason": f"skewed (skewness={skew:.2f}) with low outlier ratio ({outlier_ratio:.1f}%), log transform",
                        "confidence": "high", "source": "rule",
                    })
                elif skew > 1.0:
                    # Skewed with outliers: robust_scaling handles both
                    transforms.append({
                        "column": col, "action": "robust_scaling",
                        "reason": f"skewed (skewness={skew:.2f}) with outliers ({outlier_ratio:.1f}%), robust scaling",
                        "confidence": "high", "source": "rule",
                    })
                else:
                    transforms.append({
                        "column": col, "action": "standard_scaling",
                        "reason": "numeric, standard scaling",
                        "confidence": "high", "source": "rule",
                    })

            resolved_cols.add(col)

    ambiguous = [c for c in df.columns if c not in resolved_cols and c not in skip_cols]
    logger.info("Rule-based: %d transforms, %d ambiguous", len(transforms), len(ambiguous))
    return transforms, ambiguous


def _detect_leakage(df: pd.DataFrame, target: str) -> list[dict]:
    """Detect features with correlation >0.95 to target."""
    alerts = []
    if target not in df.columns or not pd.api.types.is_numeric_dtype(df[target]):
        return alerts
    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target]
    for col in num_cols:
        try:
            corr = abs(df[col].corr(df[target]))
            if corr > LEAKAGE_CORRELATION_THRESHOLD:
                alerts.append({"column": col, "correlation": round(float(corr), 4),
                               "reason": f"corr {corr:.4f} > {LEAKAGE_CORRELATION_THRESHOLD}"})
        except Exception:
            continue
    if alerts:
        logger.info("Leakage: %s", [a["column"] for a in alerts])
    return alerts


def _cramers_v(series_a: pd.Series, series_b: pd.Series) -> float:
    """Compute Cramér's V association measure between two categorical series."""
    from scipy.stats import chi2_contingency

    ct = pd.crosstab(series_a, series_b)
    if ct.size == 0:
        return 0.0
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    denom = n * (min(r, k) - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def _suggest_actions(corr_value: float, n_group: int, method: str) -> list[str]:
    """Suggest treatment methods based on correlation strength and group size."""
    actions = []
    if corr_value > VERY_HIGH_CORRELATION_THRESHOLD:
        actions.append("drop_least_informative: supprimer la variable avec le moins de variance ou la moins corrélée au target")
    if n_group >= 3:
        actions.append("pca: réduction de dimension sur le groupe de variables corrélées (>= 3)")
    if method == "pearson":
        actions.append("vif: supprimer itérativement les features avec VIF > 5")
        actions.append("feature_interaction: combiner les variables corrélées (ratio, produit, différence)")
    actions.append("keep: certains modèles (arbres, gradient boosting) tolèrent la colinéarité")
    return actions


def _detect_high_correlations(df: pd.DataFrame, target: str) -> list[dict]:
    """Detect inter-feature multicollinearity (numeric: Pearson, categorical: Cramér's V).

    Threshold: 0.75 = high, 0.90 = very_high.
    Returns list of correlation alerts with suggested treatment actions.
    """
    alerts = []
    skip = {target}

    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in skip]
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().abs()
        seen = set()
        for i, col_a in enumerate(num_cols):
            for col_b in num_cols[i + 1:]:
                pair_key = (col_a, col_b)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                corr_val = corr_matrix.loc[col_a, col_b]
                if pd.isna(corr_val) or corr_val < HIGH_CORRELATION_THRESHOLD:
                    continue
                severity = "very_high" if corr_val > VERY_HIGH_CORRELATION_THRESHOLD else "high"
                alerts.append({
                    "column_a": col_a, "column_b": col_b,
                    "correlation": round(float(corr_val), 4),
                    "method": "pearson", "severity": severity,
                    "suggested_actions": [],  # filled below
                })

    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in skip]
    # Include boolean columns
    cat_cols += [c for c in df.columns if str(df[c].dtype) == "bool" and c not in skip]
    cat_cols = list(dict.fromkeys(cat_cols))  # dedupe preserving order

    if len(cat_cols) >= 2:
        for i, col_a in enumerate(cat_cols):
            for col_b in cat_cols[i + 1:]:
                try:
                    valid = df[[col_a, col_b]].dropna()
                    if len(valid) < 10:
                        continue
                    v = _cramers_v(valid[col_a], valid[col_b])
                except Exception:
                    continue
                if v < HIGH_CORRELATION_THRESHOLD:
                    continue
                severity = "very_high" if v > VERY_HIGH_CORRELATION_THRESHOLD else "high"
                alerts.append({
                    "column_a": col_a, "column_b": col_b,
                    "correlation": round(float(v), 4),
                    "method": "cramers_v", "severity": severity,
                    "suggested_actions": [],
                })

    # Build adjacency for group size, then assign suggestions
    from collections import defaultdict
    adjacency: dict[str, set[str]] = defaultdict(set)
    for a in alerts:
        adjacency[a["column_a"]].add(a["column_b"])
        adjacency[a["column_b"]].add(a["column_a"])

    for a in alerts:
        group_size = len(adjacency[a["column_a"]] | adjacency[a["column_b"]]) + 1
        a["suggested_actions"] = _suggest_actions(a["correlation"], group_size, a["method"])

    if alerts:
        logger.info("Inter-feature correlations: %d pair(s) above %.2f [%d numeric, %d categorical]",
                     len(alerts), HIGH_CORRELATION_THRESHOLD,
                     sum(1 for a in alerts if a["method"] == "pearson"),
                     sum(1 for a in alerts if a["method"] == "cramers_v"))
    return alerts


# -- Node functions --


def propose_transformations(state: PipelineState) -> dict:
    """dtype_audit transforms + rule-based + LLM on ambiguous. Sorted by execution priority."""
    df = pd.read_csv(state["df_path"])
    target = state["target"]
    df_info = state["df_info"]
    domain_context = state.get("domain_context", {})

    auto_drop = df_info.get("auto_drop", [])

    # Fix 7: re-run trivial detection on train only — full-dataset detection may miss or over-flag
    train_trivial = _detect_trivial_columns(df, target)
    original_trivial_names = {d["column"] for d in auto_drop}
    new_trivial = [d for d in train_trivial if d["column"] not in original_trivial_names]
    if new_trivial:
        logger.info("Additional trivial columns in train (not in full-data analysis): %s",
                    [d["column"] for d in new_trivial])
        auto_drop = auto_drop + new_trivial

    auto_transforms = [{"column": d["column"], "action": "drop_column", "reason": d["reason"],
                        "confidence": "high", "source": "rule"} for d in auto_drop]

    dtype_transforms = _dtype_audit_transforms(state.get("dtype_audit", []))

    semantic_anomalies = df_info.get("semantic_anomalies", [])
    sentinel_transforms = _semantic_anomaly_transforms(semantic_anomalies, df)

    # Fix 1+2: recompute col_summary on train only — avoids test-data statistics
    # (skewness, outlier_ratio from full dataset would influence log/robust scaling choice)
    col_summary = _build_col_summary(df, target, {d["column"] for d in auto_drop})
    rule_transforms, ambiguous_cols = _rule_based_decisions(df, target, auto_drop, col_summary=col_summary)

    existing_cols = set(df.columns)

    llm_transforms = []
    if ambiguous_cols:
        ambiguous_info = {}
        for col in ambiguous_cols:
            info = {"type": str(df[col].dtype), "nunique": int(df[col].nunique())}
            pct_nan = df[col].isnull().mean()
            if pct_nan > 0:
                info["pct_nan"] = round(float(pct_nan) * 100, 1)
            if pd.api.types.is_numeric_dtype(df[col]):
                info["mean"] = round(float(df[col].mean()), 4)
                info["std"] = round(float(df[col].std()), 4)
            ambiguous_info[col] = info

        human_feedback = state.get("human_feedback", "")
        feedback_block = (
            f"\n\nFEEDBACK UTILISATEUR (a integrer dans ta reponse) :\n{human_feedback}"
            if human_feedback else ""
        )

        prompt = prompt_templates.format_template(
            "propose_transformations",
            diagnosis=str(df_info.get("diagnosis", {})),
            domain_context=f"{domain_context.get('domain', 'unknown')} - {domain_context.get('description', '')}",
            constraints=str(domain_context.get("constraints", [])),
            semantic_anomalies=str(df_info.get("semantic_anomalies", [])),
            ambiguous_columns=str(ambiguous_info),
            target=target,
        ) + feedback_block
        system_prompt = prompt_templates.get_template("system_prompt")
        logger.info("LLM for %d ambiguous columns", len(ambiguous_cols))
        llm_response = validate_llm_response(llm_client, prompt, system_prompt, TransformResponse)
        raw_llm_transforms = llm_response.get("transformations", [])

        # Validate LLM proposals: reject drop_column and unknown columns
        for t in raw_llm_transforms:
            col = t.get("column", "")
            action = t.get("action", "")
            if action == "drop_column":
                logger.warning(
                    "LLM proposed drop_column on '%s' — rejected (only rule-based drops are allowed)", col
                )
                continue
            if col not in existing_cols:
                logger.warning(
                    "LLM proposed action '%s' on unknown column '%s' — rejected", action, col
                )
                continue
            llm_transforms.append(t)

        n_rejected = len(raw_llm_transforms) - len(llm_transforms)
        if n_rejected:
            logger.info("LLM validation: %d proposal(s) rejected", n_rejected)

    cached_leakage = state.get("high_target_correlation_alerts")
    high_target_correlation_alerts = cached_leakage if cached_leakage is not None else _detect_leakage(df, target)

    cached_correlations = state.get("correlation_alerts")
    correlation_alerts = cached_correlations if cached_correlations is not None else _detect_high_correlations(df, target)

    feature_proposals = []
    try:
        fe_prompt = prompt_templates.format_template(
            "propose_feature_engineering",
            domain=domain_context.get("domain", "unknown"),
            columns=list(df.columns), target=target,
            task_type=state.get("task_type", "unknown"),
            diagnosis=str(df_info.get("diagnosis", {})),
        )
        system_prompt = prompt_templates.get_template("system_prompt")
        logger.info("LLM for feature engineering proposals")
        fe_response = validate_llm_response(llm_client, fe_prompt, system_prompt, FeatureEngineeringResponse)
        raw_fe = fe_response.get("features", [])

        # Validate FE proposals: all source_columns must exist in the DataFrame
        for fp in raw_fe:
            missing_src = [c for c in fp.get("source_columns", []) if c not in existing_cols]
            if missing_src:
                logger.warning(
                    "FE proposal '%s' references unknown columns %s — rejected", fp.get("name"), missing_src
                )
                continue
            feature_proposals.append(fp)

        n_fe_rejected = len(raw_fe) - len(feature_proposals)
        if n_fe_rejected:
            logger.info("FE validation: %d proposal(s) rejected", n_fe_rejected)
    except Exception as e:
        logger.info("FE proposals skipped: %s", str(e))

    all_transforms = auto_transforms + dtype_transforms + sentinel_transforms + rule_transforms + llm_transforms
    all_transforms.sort(key=lambda t: _ACTION_PRIORITY.get(t["action"], 99))

    logger.info("Proposals: %d auto-drop, %d dtype, %d sentinel, %d rule, %d LLM = %d total",
                len(auto_transforms), len(dtype_transforms), len(sentinel_transforms),
                len(rule_transforms), len(llm_transforms), len(all_transforms))

    return {
        "proposals": {
            "transformations": all_transforms,
            "feature_engineering": feature_proposals,
            "_fe_note": "Feature engineering proposals are informational only. NOT executed automatically.",
            "correlation_alerts": correlation_alerts,
        },
        "high_target_correlation_alerts": high_target_correlation_alerts,
        "correlation_alerts": correlation_alerts,
        "feature_proposals": feature_proposals,
    }


def execute_transformations(state: PipelineState) -> dict:
    """Execute transformations: fit on train, transform test. Scaling is deferred to post-outlier."""
    train_df = pd.read_csv(state["train_path"])
    test_df = pd.read_csv(state["test_path"])
    proposals = state["proposals"]
    transformations = proposals.get("transformations", [])
    applied = []
    confidence_entries = []
    artifacts = {}  # name -> fitted object (for persistence and test transform)
    deferred_scaling = []

    for t in transformations:
        col = t["column"]
        action = t["action"]

        if col == state["target"]:
            logger.info("Skipping target column '%s'", col)
            continue

        # Defer scaling actions to post-outlier phase
        if action in _SCALING_ACTIONS:
            deferred_scaling.append(t)
            continue

        if col not in train_df.columns:
            logger.info("Column '%s' not found, skipping", col)
            continue

        try:
            # Pre-imputation: replace sentinel values with NaN
            if action == "replace_sentinel":
                sentinel_values = t.get("sentinel_values", [])
                for sv in sentinel_values:
                    train_df[col] = train_df[col].replace(sv, np.nan)
                    if col in test_df.columns:
                        test_df[col] = test_df[col].replace(sv, np.nan)
            # Pre-imputation: cast numeric-as-string columns
            elif action == "cast_numeric":
                train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
                if col in test_df.columns:
                    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
            elif action == "impute_mean":
                if train_df[col].isnull().any():
                    flag_col = f"{col}_was_imputed"
                    train_df[flag_col] = train_df[col].isnull().astype(int)
                    test_df[flag_col] = test_df[col].isnull().astype(int) if col in test_df.columns else 0
                fill_value = float(train_df[col].mean())
                train_df[col] = train_df[col].fillna(fill_value)
                if col in test_df.columns:
                    test_df[col] = test_df[col].fillna(fill_value)
                artifacts[f"impute_mean_{col}"] = fill_value
            elif action == "impute_median":
                if train_df[col].isnull().any():
                    flag_col = f"{col}_was_imputed"
                    train_df[flag_col] = train_df[col].isnull().astype(int)
                    test_df[flag_col] = test_df[col].isnull().astype(int) if col in test_df.columns else 0
                fill_value = float(train_df[col].median())
                train_df[col] = train_df[col].fillna(fill_value)
                if col in test_df.columns:
                    test_df[col] = test_df[col].fillna(fill_value)
                artifacts[f"impute_median_{col}"] = fill_value
            elif action == "impute_mode":
                if train_df[col].isnull().any():
                    flag_col = f"{col}_was_imputed"
                    train_df[flag_col] = train_df[col].isnull().astype(int)
                    test_df[flag_col] = test_df[col].isnull().astype(int) if col in test_df.columns else 0
                fill_value = train_df[col].mode()[0]
                train_df[col] = train_df[col].fillna(fill_value)
                if col in test_df.columns:
                    test_df[col] = test_df[col].fillna(fill_value)
                artifacts[f"impute_mode_{col}"] = fill_value
            elif action == "drop_column":
                train_df = train_df.drop(columns=[col])
                if col in test_df.columns:
                    test_df = test_df.drop(columns=[col])
            elif action == "drop_rows":
                train_df = train_df.dropna(subset=[col])
                # On test, impute with train median instead of dropping rows
                if col in test_df.columns and test_df[col].isnull().any():
                    fallback = float(train_df[col].median())
                    test_df[col] = test_df[col].fillna(fallback)
                    logger.info("Test set: imputed '%s' with train median instead of dropping", col)
            elif action == "one_hot_encoding":
                # Fit categories from train only; unknown test categories → zeros (handle_unknown='ignore')
                enc = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
                arr_train = enc.fit_transform(train_df[[col]].astype(str))
                feature_names = [f"{col}_{cat}" for cat in enc.categories_[0][1:]]
                dummies_train = pd.DataFrame(arr_train, columns=feature_names, index=train_df.index, dtype=int)
                train_df = pd.concat([train_df.drop(columns=[col]), dummies_train], axis=1)
                if col in test_df.columns:
                    arr_test = enc.transform(test_df[[col]].astype(str))
                    dummies_test = pd.DataFrame(arr_test, columns=feature_names, index=test_df.index, dtype=int)
                    test_df = pd.concat([test_df.drop(columns=[col]), dummies_test], axis=1)
                artifacts[f"ohe_encoder_{col}"] = enc
            elif action in ("label_encoding", "ordinal_encoding"):
                str_train = train_df[[col]].astype(str)
                # For boolean columns, enforce deterministic category order
                if train_df[col].dtype == "bool" or set(str_train[col].unique()) <= {"True", "False"}:
                    enc = OrdinalEncoder(
                        categories=[["False", "True"]],
                        handle_unknown="use_encoded_value", unknown_value=-1,
                    )
                else:
                    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                train_df[col] = enc.fit_transform(str_train)
                if col in test_df.columns:
                    test_df[col] = enc.transform(test_df[[col]].astype(str))
                artifacts[f"ordinal_encoder_{col}"] = enc
            elif action == "frequency_encoding":
                freq = train_df[col].value_counts(normalize=True).to_dict()
                train_df[col] = train_df[col].map(freq).fillna(0)
                if col in test_df.columns:
                    test_df[col] = test_df[col].map(freq).fillna(0)
                artifacts[f"freq_map_{col}"] = freq
            elif action == "extract_datetime":
                for frame in (train_df, test_df):
                    if col in frame.columns:
                        dt = pd.to_datetime(frame[col], errors="coerce")
                        frame[f"{col}_year"] = dt.dt.year
                        frame[f"{col}_month"] = dt.dt.month
                        frame[f"{col}_day"] = dt.dt.day
                        frame[f"{col}_dayofweek"] = dt.dt.dayofweek
                train_df = train_df.drop(columns=[col])
                if col in test_df.columns:
                    test_df = test_df.drop(columns=[col])
            else:
                logger.warning("Unknown action '%s' for '%s', skipping", action, col)
                continue

            applied.append({"column": col, "action": action})
            confidence_entries.append({
                "step": "transformation", "column": col, "action": action,
                "confidence": t.get("confidence", "medium"), "source": t.get("source", "llm"),
            })
            logger.info("Applied %s on '%s' (fit on train, transform on test)", action, col)

        except Exception as e:
            logger.warning("Failed %s on '%s': %s", action, col, str(e))

    # Save artifacts (scalers, encoders)
    artifacts_dir = Path("data/outputs/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        artifact_path = artifacts_dir / f"{name}.joblib"
        joblib.dump(obj, artifact_path)
    if artifacts:
        logger.info("Saved %d artifacts to %s", len(artifacts), artifacts_dir)

    # Save transformed train/test
    raw_path = state["df_path"]
    if urlparse(raw_path).scheme in ("http", "https"):
        stem = Path(urlparse(raw_path).path).stem
        out_dir = Path("data/outputs")
    else:
        p = Path(raw_path)
        stem = p.stem.replace("_train", "")
        out_dir = p.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = str(out_dir / f"{stem}_train_transformed.csv")
    test_out = str(out_dir / f"{stem}_test_transformed.csv")
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    logger.info("Saved train to %s, test to %s", train_out, test_out)

    if deferred_scaling:
        logger.info("Deferred %d scaling actions to post-outlier phase", len(deferred_scaling))

    # W&B: proposals vs executed comparison table + transformed dataset
    tracker.log_comparison_table(transformations, applied, table_name="transform_comparison")
    tracker.log_datasets(train_out, test_out, stage="transformed")

    # Only serializable values in state (sklearn objects already saved to disk)
    serializable_artifacts = {k: v for k, v in artifacts.items() if not hasattr(v, "fit_transform")}

    return {
        "df_path": train_out,
        "train_path": train_out,
        "test_path": test_out,
        "fitted_artifacts": serializable_artifacts,
        "deferred_scaling": deferred_scaling,
        "confidence_map": confidence_entries,
        "steps_done": [{
            "step": "transformations", "applied": applied,
            "new_shape": str(train_df.shape), "artifacts_saved": list(artifacts.keys()),
            "deferred_count": len(deferred_scaling),
        }],
    }


def apply_scaling(state: PipelineState) -> dict:
    """Apply deferred scaling after outlier treatment. Fit on train, transform test."""
    deferred = state.get("deferred_scaling", [])
    if not deferred:
        logger.info("No deferred scaling to apply")
        return {}

    train_df = pd.read_csv(state["train_path"])
    test_df = pd.read_csv(state["test_path"])
    applied = []
    confidence_entries = []
    artifacts = {}

    for t in deferred:
        col = t["column"]
        action = t["action"]

        if col not in train_df.columns:
            logger.info("Column '%s' not found for scaling, skipping", col)
            continue

        try:
            if action == "standard_scaling":
                scaler = StandardScaler()
                train_df[col] = scaler.fit_transform(train_df[[col]])
                if col in test_df.columns:
                    test_df[col] = scaler.transform(test_df[[col]])
                artifacts[f"standard_scaler_{col}"] = scaler
            elif action == "minmax_scaling":
                scaler = MinMaxScaler()
                train_df[col] = scaler.fit_transform(train_df[[col]])
                if col in test_df.columns:
                    test_df[col] = scaler.transform(test_df[[col]])
                artifacts[f"minmax_scaler_{col}"] = scaler
            elif action == "robust_scaling":
                scaler = RobustScaler()
                train_df[col] = scaler.fit_transform(train_df[[col]])
                if col in test_df.columns:
                    test_df[col] = scaler.transform(test_df[[col]])
                artifacts[f"robust_scaler_{col}"] = scaler
            elif action == "log_transform":
                col_min = float(train_df[col].min())
                if col_min < 0:
                    shift = abs(col_min) + 1
                    train_df[col] = np.log1p(train_df[col] + shift)
                    if col in test_df.columns:
                        test_df[col] = np.log1p(test_df[col] + shift)
                    artifacts[f"log_shift_{col}"] = shift
                else:
                    train_df[col] = np.log1p(train_df[col])
                    if col in test_df.columns:
                        test_df[col] = np.log1p(test_df[col])
            elif action == "sqrt_transform":
                has_negative = bool(train_df[col].min() < 0)
                train_df[col] = np.sqrt(train_df[col].clip(lower=0))
                if col in test_df.columns:
                    test_df[col] = np.sqrt(test_df[col].clip(lower=0))
                if has_negative:
                    artifacts[f"sqrt_clip_{col}"] = True
            else:
                continue

            applied.append({"column": col, "action": action})
            confidence_entries.append({
                "step": "scaling", "column": col, "action": action,
                "confidence": t.get("confidence", "medium"), "source": t.get("source", "rule"),
            })
            logger.info("Applied deferred %s on '%s' (fit on train)", action, col)

        except Exception as e:
            logger.warning("Failed scaling %s on '%s': %s", action, col, str(e))

    # Save scaler artifacts
    artifacts_dir = Path("data/outputs/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        if name.startswith(("standard_scaler_", "minmax_scaler_", "robust_scaler_", "log_shift_", "sqrt_clip_")):
            joblib.dump(obj, artifacts_dir / f"{name}.joblib")

    # Save final scaled datasets - derive names from original dataset stem
    raw_path = state["train_path"]
    p = Path(raw_path)
    out_dir = p.parent
    # Use original dataset name if available, otherwise strip known suffixes
    original_name = state.get("dataset_name", "")
    if original_name:
        base_stem = Path(original_name).stem
    else:
        base_stem = p.stem
        for suffix in ("_train_clean", "_train_transformed", "_train"):
            if base_stem.endswith(suffix):
                base_stem = base_stem[: -len(suffix)]
                break
    train_out = str(out_dir / f"{base_stem}_train_final.csv")
    test_out = str(out_dir / f"{base_stem}_test_final.csv")
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    logger.info("Final datasets: %s, %s", train_out, test_out)

    # W&B: fitted artifacts + final datasets
    tracker.log_fitted_artifacts()
    tracker.log_datasets(train_out, test_out, stage="final")

    return {
        "df_path": train_out,
        "train_path": train_out,
        "test_path": test_out,
        "confidence_map": confidence_entries,
        "steps_done": [{
            "step": "scaling", "applied": applied, "new_shape": str(train_df.shape),
        }],
    }
