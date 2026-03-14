"""ReportAgent : W&B tracking, constraint validation, baseline evaluation."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.llm.utils import parse_json_response
from src.utils.logger import get_logger
from src.agents.state import PipelineState
from src.agents.dependencies import llm_client, prompt_templates
from src.tracking import wandb_tracker as tracker

logger = get_logger(__name__)


def _check_constraints(state: PipelineState) -> list[dict]:
    """Check inferred domain constraints against the final train data.

    Supports simple numeric constraints like 'X >= 0', 'X > 0', 'X <= N', 'X < N'.
    Returns a list of violation dicts.
    """
    import re

    constraints = state.get("domain_context", {}).get("constraints", [])
    train_path = state.get("train_path", "")
    if not constraints or not train_path:
        return []

    try:
        df = pd.read_csv(train_path)
    except Exception:
        return []

    # Parse constraint patterns: "column op value" or natural language hints
    ops = {
        ">=": lambda s, v: (s < v).sum(),
        "<=": lambda s, v: (s > v).sum(),
        ">": lambda s, v: (s <= v).sum(),
        "<": lambda s, v: (s >= v).sum(),
    }
    pattern = re.compile(
        r"(\w+)\s*(>=|<=|>|<)\s*(-?\d+\.?\d*)"
    )

    violations = []
    parsed_count = 0
    for c in constraints:
        m = pattern.search(str(c))
        if not m:
            logger.warning("Constraint not parseable (skipped): %s", c)
            continue
        parsed_count += 1
        col_name, op, val_str = m.group(1), m.group(2), float(m.group(3))
        # Find column (case-insensitive match)
        matched = [col for col in df.columns if col.lower() == col_name.lower()]
        if not matched or not pd.api.types.is_numeric_dtype(df[matched[0]]):
            continue
        col = matched[0]
        n_violations = int(ops[op](df[col].dropna(), val_str))
        if n_violations > 0:
            violations.append({
                "constraint": str(c),
                "column": col,
                "n_violations": n_violations,
                "pct": round(n_violations / max(len(df), 1) * 100, 2),
            })

    if violations:
        logger.warning("Constraint violations: %d constraint(s) violated", len(violations))
    return violations


def _compute_baseline_score(state: PipelineState) -> dict:
    """Train a dummy vs simple model on preprocessed data. Measures empirical preprocessing impact.

    Returns dict with baseline_score, model_score, delta, metric_name.
    """
    train_path = state.get("train_path", "")
    test_path = state.get("test_path", "")
    target = state.get("target", "")
    task_type = state.get("task_type", "regression")

    if not train_path or not test_path or not target:
        return {"status": "skipped", "reason": "missing train/test paths or target"}

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if target not in train_df.columns or target not in test_df.columns:
            return {"status": "skipped", "reason": "target column not in final data"}

        y_train = train_df[target]
        y_test = test_df[target]
        X_train = train_df.drop(columns=[target])
        X_test = test_df.drop(columns=[target])

        # Keep only numeric columns (encoding should already be done)
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        if not num_cols:
            return {"status": "skipped", "reason": "no numeric features after preprocessing"}

        X_train = X_train[num_cols]
        X_test = X_test[[c for c in num_cols if c in X_test.columns]]
        # Align columns
        for c in num_cols:
            if c not in X_test.columns:
                X_test[c] = 0
        X_test = X_test[num_cols]

        # Warn on residual NaN before fallback fill
        train_nan = int(X_train.isnull().sum().sum())
        test_nan = int(X_test.isnull().sum().sum())
        if train_nan or test_nan:
            logger.warning(
                "Baseline eval: %d NaN in train, %d NaN in test after pipeline — "
                "filling with 0 for model fitting (may bias score)",
                train_nan, test_nan,
            )
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        if task_type == "classification":
            from sklearn.dummy import DummyClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score

            dummy = DummyClassifier(strategy="most_frequent", random_state=42)
            dummy.fit(X_train, y_train)
            baseline = float(accuracy_score(y_test, dummy.predict(X_test)))

            model = LogisticRegression(max_iter=500, random_state=42, solver="lbfgs")
            model.fit(X_train, y_train)
            score = float(accuracy_score(y_test, model.predict(X_test)))

            return {
                "status": "ok", "metric": "accuracy",
                "baseline_score": round(baseline, 4),
                "model_score": round(score, 4),
                "delta": round(score - baseline, 4),
            }
        else:
            from sklearn.dummy import DummyRegressor
            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score

            dummy = DummyRegressor(strategy="mean")
            dummy.fit(X_train, y_train)
            baseline = float(r2_score(y_test, dummy.predict(X_test)))

            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train, y_train)
            score = float(r2_score(y_test, model.predict(X_test)))

            return {
                "status": "ok", "metric": "r2",
                "baseline_score": round(baseline, 4),
                "model_score": round(score, 4),
                "delta": round(score - baseline, 4),
            }

    except Exception as e:
        logger.warning("Baseline evaluation failed: %s", str(e))
        return {"status": "error", "reason": str(e)}


def _compute_quality_score(state: PipelineState) -> dict:
    """Compute a decomposed quality score on the train set.

    Returns dict: {overall, completeness, type_consistency, leakage_risk, outlier_coverage, drop_ratio}.
    """
    df_info = state.get("df_info", {})
    steps_done = state.get("steps_done", [])
    original_path = state.get("original_df_path", "")

    # Completeness (30 pts)
    completeness = 30.0
    rows_lost = 0
    cols_lost = 0
    completeness_gain = 0.0
    if original_path:
        try:
            original_df = pd.read_csv(original_path)
            final_train = pd.read_csv(state.get("train_path", state.get("df_path", "")))
            # Rows lost = original train rows - final train rows (account for split)
            split_step = next((s for s in steps_done if s["step"] == "split_data"), {})
            original_train_size = split_step.get("train_size", len(original_df))
            rows_lost = max(0, original_train_size - len(final_train))
            cols_lost = max(0, len(original_df.columns) - len(final_train.columns))
            final_missing = int(final_train.isnull().sum().sum())
            final_cells = max(int(final_train.size), 1)
            # Fix 5: compare NaN rates on shared columns only — OHE adds zero-filled columns
            # that would otherwise artificially deflate final_pct
            shared_cols = [c for c in original_df.columns if c in final_train.columns]
            if shared_cols:
                original_pct = original_df[shared_cols].isnull().values.mean()
                final_pct = final_train[shared_cols].isnull().values.mean()
            else:
                original_pct = float(original_df.isnull().sum().sum()) / max(int(original_df.size), 1)
                final_pct = float(final_missing / final_cells)
            completeness_gain = max(0.0, original_pct - final_pct)
            completeness = 30.0 - min(final_pct * 100, 30.0)
        except Exception:
            pass
    else:
        total_missing = sum(df_info.get("missing_values", {}).values())
        total_cells = 1
        shape_str = df_info.get("shape", "0 rows x 0 columns")
        try:
            parts = shape_str.split("rows x")
            total_cells = max(int(parts[0].strip()) * int(parts[1].strip().split()[0]), 1)
        except (IndexError, ValueError):
            pass
        completeness = 30.0 - min((total_missing / total_cells) * 100, 30)
        completeness_gain = 0
        rows_lost = 0
        cols_lost = 0

    # Drop ratio (20 pts)
    drop_penalty = 0.0
    transform_step = next((s for s in steps_done if s["step"] == "transformations"), {})
    applied = transform_step.get("applied", [])
    auto_drop_cols = {d["column"] for d in df_info.get("auto_drop", [])}
    llm_drops = sum(1 for a in applied if a.get("action") == "drop_column" and a.get("column") not in auto_drop_cols)
    original_cols = len(df_info.get("columns", []))
    if original_cols > 0 and llm_drops > 0:
        llm_drop_ratio = llm_drops / original_cols
        if llm_drop_ratio > 0.3:
            drop_penalty = 20
        elif llm_drop_ratio > 0.15:
            drop_penalty = 10
        elif llm_drop_ratio > 0.05:
            drop_penalty = 5

    # Type consistency (15 pts)
    dtype_audit = state.get("dtype_audit", [])
    # Subtract columns that were corrected by transformations (cast_numeric, extract_datetime)
    corrected_cols = {a.get("column") for a in applied
                      if a.get("action") in ("cast_numeric", "extract_datetime")}
    uncorrected = [d for d in dtype_audit if d.get("column") not in corrected_cols]
    type_penalty = min(len(uncorrected) * 3, 15)

    # Leakage risk (15 pts)
    leakage_alerts = state.get("high_target_correlation_alerts", [])
    leakage_penalty = min(len(leakage_alerts) * 5, 15)

    # Outlier coverage (20 pts)
    outlier_step = next((s for s in steps_done if s["step"] == "outlier_treatment"), {})
    outlier_applied = outlier_step.get("applied", [])
    outlier_report = outlier_step.get("outlier_report", [])
    cols_with_outliers = [r for r in outlier_report if r.get("n_outliers", 0) > 0]
    if not cols_with_outliers:
        # No outliers detected — full score
        outlier_penalty = 0.0
    elif not outlier_applied:
        # Outliers exist but nothing was treated
        outlier_penalty = 5.0
    else:
        treated_cols = {a["column"] for a in outlier_applied if a.get("action") != "keep"}
        total_outlier_cols = len(cols_with_outliers)
        coverage = len(treated_cols) / max(total_outlier_cols, 1)
        # Penalty scales from 5 (0% coverage) to 0 (100% coverage)
        outlier_penalty = round(5.0 * (1.0 - coverage), 1)

    # Sub-scores
    comp_score = round(float(completeness), 1)           # /30
    drop_score = round(float(20 - drop_penalty), 1)      # /20
    type_score = round(float(15 - type_penalty), 1)      # /15
    leak_score = round(float(15 - leakage_penalty), 1)   # /15
    outlier_score = round(float(20 - outlier_penalty), 1) # /20

    overall = max(round(comp_score + drop_score + type_score + leak_score + outlier_score, 1), 0)

    return {
        "overall": float(overall),
        "completeness": comp_score,
        "completeness_gain": round(float(completeness_gain * 100), 1) if completeness_gain else 0,
        "rows_lost": int(rows_lost),
        "cols_lost": int(cols_lost),
        "type_consistency": type_score,
        "leakage_risk": "high" if leakage_alerts else "low",
        "leakage_count": int(len(leakage_alerts)),
        "drop_penalty": round(float(drop_penalty), 1),
        "llm_drops": int(llm_drops),
        "outlier_coverage": outlier_score,
    }


def _analyze_confidence_map(confidence_map: list[dict], all_columns: list[str] | None = None) -> dict:
    """Analyze the accumulated confidence map."""
    if not confidence_map:
        return {"total_decisions": 0, "deterministic_ratio": 0, "low_confidence_columns": [],
                "no_decision_columns": list(all_columns) if all_columns else []}

    total = len(confidence_map)
    rule_count = sum(1 for c in confidence_map if c.get("source") == "rule")
    llm_count = total - rule_count
    low_confidence = [c["column"] for c in confidence_map if c.get("confidence") == "low"]
    medium_confidence = [c["column"] for c in confidence_map if c.get("confidence") == "medium"]

    # Detect columns with no decisions at all
    decided_cols = {c["column"] for c in confidence_map if "column" in c}
    no_decision = [col for col in (all_columns or []) if col not in decided_cols]

    return {
        "total_decisions": total,
        "deterministic_count": rule_count,
        "llm_count": llm_count,
        "deterministic_ratio": round(rule_count / total * 100, 1) if total > 0 else 0,
        "low_confidence_columns": list(set(low_confidence)),
        "medium_confidence_columns": list(set(medium_confidence)),
        "no_decision_columns": no_decision,
    }


def generate_report(state: PipelineState) -> dict:
    """Generate final JSON report (1 LLM call)."""
    steps_done = state.get("steps_done", [])
    df_info = state.get("df_info", {})
    domain_context = state.get("domain_context", {})
    target_meta = state.get("target_meta", {})
    feature_proposals = state.get("feature_proposals", [])

    # Deterministic computations
    quality_score = _compute_quality_score(state)
    logger.info("Quality score: %.1f/100", quality_score["overall"])

    baseline_eval = _compute_baseline_score(state)
    if baseline_eval.get("status") == "ok":
        logger.info("Baseline evaluation: %s=%s, model=%s, delta=%s",
                     baseline_eval["metric"], baseline_eval["baseline_score"],
                     baseline_eval["model_score"], baseline_eval["delta"])
    else:
        logger.info("Baseline evaluation: %s", baseline_eval.get("reason", "skipped"))

    constraint_violations = _check_constraints(state)
    if constraint_violations:
        logger.info("Domain constraints: %d violation(s) found", len(constraint_violations))

    confidence_map = state.get("confidence_map", [])
    # Exclude target and auto-dropped columns from no_decision detection
    auto_drop_names = {d["column"] for d in df_info.get("auto_drop", [])}
    target = state.get("target", "")
    actionable_columns = [c for c in df_info.get("columns", []) if c != target and c not in auto_drop_names]
    confidence_analysis = _analyze_confidence_map(confidence_map, all_columns=actionable_columns)
    logger.info("Confidence: %d decisions, %.1f%% deterministic",
                confidence_analysis["total_decisions"], confidence_analysis["deterministic_ratio"])

    # Extract step data
    diagnosis = df_info.get("diagnosis", {})
    split_step = next((s for s in steps_done if s["step"] == "split_data"), {})
    transformations = next((s for s in steps_done if s["step"] == "transformations"), {})
    outlier_treatment = next((s for s in steps_done if s["step"] == "outlier_treatment"), {})
    scaling_step = next((s for s in steps_done if s["step"] == "scaling"), {})
    leakage_alerts = state.get("high_target_correlation_alerts", [])
    correlation_alerts = state.get("correlation_alerts", [])

    # Original vs final shape
    original_shape = df_info.get("shape", "unknown")
    final_shape = scaling_step.get("new_shape", outlier_treatment.get("new_shape", transformations.get("new_shape", "unknown")))

    # Single LLM call: narrative + recommendations
    system_prompt = prompt_templates.get_template("system_prompt")
    report_prompt = prompt_templates.format_template(
        "generate_report",
        diagnosis=str(diagnosis),
        domain_context=f"{domain_context.get('domain', 'unknown')} - {domain_context.get('description', '')}",
        target_meta=str(target_meta),
        transformations=str(transformations),
        outlier_treatment=str(outlier_treatment),
        leakage_alerts=str(leakage_alerts),
        feature_proposals=str(feature_proposals),
        original_shape=original_shape,
        final_shape=final_shape,
        quality_score=str(quality_score),
        confidence_analysis=str(confidence_analysis),
        constraint_violations=str(constraint_violations),
        baseline_evaluation=str(baseline_eval),
        correlation_alerts=str(correlation_alerts),
    )

    logger.info("Generating report (1 LLM call)")
    response = llm_client.complete(report_prompt, system_prompt=system_prompt)
    report = parse_json_response(response)

    # Inject deterministic fields
    report["prompt_templates_version"] = prompt_templates.get_version()
    report["quality_score"] = quality_score
    report["confidence_analysis"] = confidence_analysis
    report["confidence_map"] = confidence_map
    report["feature_engineering_suggestions"] = feature_proposals
    report["train_test_split"] = {
        "train_size": split_step.get("train_size", "unknown"),
        "test_size": split_step.get("test_size", "unknown"),
        "stratified": split_step.get("stratified", False),
        "train_path": state.get("train_path", ""),
        "test_path": state.get("test_path", ""),
    }
    report["baseline_evaluation"] = baseline_eval
    report["constraint_violations"] = constraint_violations
    report["known_limitations"] = [
        "Outlier detection is univariate only (IQR/Z-score per column). "
        "Multivariate outliers (e.g. Mahalanobis, Isolation Forest) are not detected.",
    ]
    report["correlation_alerts"] = correlation_alerts

    # Critical warnings: leakage + constraint violations
    critical_warnings = []
    if leakage_alerts:
        for alert in leakage_alerts:
            critical_warnings.append({
                "type": "high_target_correlation",
                "column": alert["column"],
                "detail": alert.get("reason", f"correlation {alert.get('correlation', '?')} with target"),
            })
        logger.warning("High target correlation: %d feature(s) with corr > 0.95 to target: %s",
                       len(leakage_alerts), [a["column"] for a in leakage_alerts])
    if constraint_violations:
        for v in constraint_violations:
            critical_warnings.append({
                "type": "constraint_violation",
                "column": v["column"],
                "detail": f"{v['constraint']}: {v['n_violations']} violations ({v['pct']}%)",
            })
    report["critical_warnings"] = critical_warnings

    # Save
    output_path = Path("data/outputs/report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Report saved to %s", output_path)

    # W&B: log metrics, artifacts, finish run
    tracker.log_metrics({
        "quality_overall": quality_score["overall"],
        "completeness": quality_score["completeness"],
        "type_consistency": quality_score["type_consistency"],
        "outlier_coverage": quality_score["outlier_coverage"],
        "deterministic_ratio": confidence_analysis["deterministic_ratio"],
    }, step="report")
    if baseline_eval.get("status") == "ok":
        tracker.log_metrics({
            "baseline_score": baseline_eval["baseline_score"],
            "model_score": baseline_eval["model_score"],
            "delta": baseline_eval["delta"],
        }, step="baseline")
    tracker.log_summary("quality_score", quality_score["overall"])
    tracker.log_summary("constraint_violations", len(constraint_violations))
    tracker.log_summary("correlation_alerts", len(correlation_alerts))
    tracker.log_datasets(state.get("train_path", ""), state.get("test_path", ""), stage="final")
    tracker.log_artifact("report", "report", [str(output_path)])
    tracker.finish()

    return {
        "report": report,
        "quality_score": quality_score,
        "steps_done": [{"step": "report", "output_path": str(output_path)}],
    }
