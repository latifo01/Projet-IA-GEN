"""State definition shared across all agents in the LangGraph pipeline."""

from typing import Annotated
from typing_extensions import TypedDict
import operator


class PipelineState(TypedDict):
    """State passed between agents in the preprocessing pipeline."""

    # Input
    df_path: str
    original_df_path: str       # preserved original path, never overwritten
    dataset_name: str            # original dataset stem (e.g. "titanic"), set once by analyze
    target: str

    # Split configuration (optional, has defaults)
    # strategy: "auto" | "stratified" | "temporal" | "group"
    # auto = stratified for classification, random for regression
    split_config: dict

    # Train/test split paths
    train_path: str
    test_path: str

    # BaseAgent outputs
    df_info: dict
    task_type: str              # "classification" or "regression"
    target_meta: dict           # {pct_nan, n_classes, imbalance_ratio, task_type}
    domain_context: dict        # inferred domain, constraints, semantic anomalies
    dtype_audit: list[dict]     # mistyped columns detected

    # Proposals from current agent (for human review)
    proposals: dict

    # Human decision: "approve", "reject"
    human_decision: str

    # Human feedback on reject (optional reason/instructions for re-proposal)
    human_feedback: str

    # Accumulated history of completed steps (with confidence levels)
    steps_done: Annotated[list[dict], operator.add]

    # TransformationAgent outputs
    high_target_correlation_alerts: list[dict]  # features with correlation > 0.95 to target (potential leakage or redundancy)
    correlation_alerts: list[dict]  # inter-feature multicollinearity (numeric + categorical)
    feature_proposals: list[dict]  # informational FE proposals from LLM

    # Fitted artifacts for train->test consistency (scalers, encoders, freq_maps)
    fitted_artifacts: dict

    # Deferred scaling transforms (applied after outlier treatment)
    deferred_scaling: list[dict]

    # ReportAgent outputs
    quality_score: dict         # decomposed: {overall, completeness, leakage_risk, ...}
    confidence_map: Annotated[list[dict], operator.add]  # accumulated per-decision confidence
    report: dict                # final JSON report
