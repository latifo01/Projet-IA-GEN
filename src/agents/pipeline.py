"""Pipeline : human feedback on reject, exponential backoff, vectorized audit."""

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from src.agents.state import PipelineState
from src.agents.base_agent import analyze, split_data
from src.agents.transformation_agent import propose_transformations, execute_transformations, apply_scaling
from src.agents.outlier_agent import propose_outlier_treatment, execute_outlier_treatment
from src.agents.report_agent import generate_report
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_decision(raw_decision) -> tuple[str, str]:
    """Parse human decision: accept str or dict {decision, feedback}."""
    if isinstance(raw_decision, dict):
        return raw_decision.get("decision", "reject"), raw_decision.get("feedback", "")
    return str(raw_decision), ""


def human_review_domain(state: PipelineState) -> dict:
    """Interrupt for human validation of inferred domain context."""
    raw = interrupt({
        "type": "domain_review",
        "domain_context": state.get("domain_context", {}),
        "semantic_anomalies": state.get("df_info", {}).get("semantic_anomalies", []),
        "message": "Review the inferred domain context and semantic anomalies. "
                   "Respond 'approve', 'reject', or {'decision': 'reject', 'feedback': '...'}.",
    })
    decision, feedback = _parse_decision(raw)
    logger.info("Human decision on domain context: %s", decision)
    return {"human_decision": decision, "human_feedback": feedback}


def human_review_transformations(state: PipelineState) -> dict:
    """Interrupt for human validation of transformation proposals."""
    proposals = state.get("proposals", {})
    correlation_alerts = state.get("correlation_alerts", [])
    high_corr = state.get("high_target_correlation_alerts", [])
    raw = interrupt({
        "type": "transformation_review",
        "proposals": proposals,
        "high_target_correlation_alerts": high_corr,
        "correlation_alerts": correlation_alerts,
        "message": "Review the proposed transformations. "
                   "Respond 'approve', 'reject', or {'decision': 'reject', 'feedback': '...'}.",
    })
    decision, feedback = _parse_decision(raw)

    # Force attention on high target correlation (potential leakage or redundant feature)
    if high_corr and decision == "approve":
        leakage_decision = interrupt({
            "type": "high_target_correlation_warning",
            "high_target_correlation_alerts": high_corr,
            "message": (
                f"WARNING: {len(high_corr)} feature(s) with correlation > 0.95 to target detected. "
                "May indicate data leakage or a redundant feature. "
                "Confirm 'approve' to keep them or 'reject' to drop them."
            ),
        })
        if leakage_decision == "reject":
            # Add drop_column for correlated columns to proposals
            current_transforms = proposals.get("transformations", [])
            for alert in high_corr:
                current_transforms.append({
                    "column": alert["column"],
                    "action": "drop_column",
                    "reason": f"high target correlation ({alert['correlation']:.4f}) — potential leakage",
                    "confidence": "high",
                    "source": "rule",
                })
            return {"human_decision": "approve", "proposals": {**proposals, "transformations": current_transforms}}

    # Log inter-feature multicollinearity (informational, no interrupt)
    if correlation_alerts:
        very_high = [a for a in correlation_alerts if a.get("severity") == "very_high"]
        high = [a for a in correlation_alerts if a.get("severity") == "high"]
        logger.info("Multicollinearity: %d pair(s) detected (%d very_high, %d high)",
                     len(correlation_alerts), len(very_high), len(high))
        for a in correlation_alerts:
            method_label = "Pearson" if a["method"] == "pearson" else "Cramér's V"
            logger.info("  %s <-> %s: %.3f (%s, %s)",
                        a["column_a"], a["column_b"], a["correlation"], method_label, a["severity"])

    logger.info("Human decision on transformations: %s", decision)
    return {"human_decision": decision, "human_feedback": feedback}


def human_review_outliers(state: PipelineState) -> dict:
    """Interrupt for human validation of outlier treatment proposals."""
    proposals = state.get("proposals", {})
    outlier_actions = proposals.get("outlier_actions", [])
    flagged = [a["column"] for a in outlier_actions if a.get("requires_review")]
    message = (
        "Review the proposed outlier treatments. "
        "Respond 'approve', 'reject', or {'decision': 'reject', 'feedback': '...'}."
    )
    if flagged:
        message += f" ATTENTION: {len(flagged)} proposal(s) flagged for review: {flagged}"
    raw = interrupt({
        "type": "outlier_review",
        "proposals": proposals,
        "requires_review": flagged,
        "message": message,
    })
    decision, feedback = _parse_decision(raw)
    logger.info("Human decision on outliers: %s", decision)
    return {"human_decision": decision, "human_feedback": feedback}


def route_after_domain_review(state: PipelineState) -> str:
    if state.get("human_decision") == "approve":
        return "propose_transformations"
    return "analyze"


def route_after_transform_review(state: PipelineState) -> str:
    if state.get("human_decision") == "approve":
        return "execute_transformations"
    return "propose_transformations"


def route_after_outlier_review(state: PipelineState) -> str:
    if state.get("human_decision") == "approve":
        return "execute_outlier_treatment"
    return "propose_outlier_treatment"


def build_pipeline(checkpointer=None):
    """Build and return the compiled LangGraph preprocessing pipeline.

    Graph: analyze -> split_data -> [domain review] -> propose_transforms -> [transform review]
           -> execute_transforms -> propose_outliers -> [outlier review]
           -> execute_outliers -> apply_scaling -> generate_report

    All fit operations use train only. Test is transformed with train-fitted params.

    Architecture note: there is no reject counter. Humans can reject any review step
    indefinitely, looping back without forced stop. This is intentional — the pipeline
    trusts the human to converge. If automated safeguards are needed, add a
    max_retries counter in the routing functions.

    Args:
        checkpointer: LangGraph checkpointer (required for interrupt/human-in-the-loop).
    """
    graph = StateGraph(PipelineState)

    # Nodes
    graph.add_node("analyze", analyze)
    graph.add_node("split_data", split_data)
    graph.add_node("human_review_domain", human_review_domain)
    graph.add_node("propose_transformations", propose_transformations)
    graph.add_node("human_review_transformations", human_review_transformations)
    graph.add_node("execute_transformations", execute_transformations)
    graph.add_node("propose_outlier_treatment", propose_outlier_treatment)
    graph.add_node("human_review_outliers", human_review_outliers)
    graph.add_node("execute_outlier_treatment", execute_outlier_treatment)
    graph.add_node("apply_scaling", apply_scaling)
    graph.add_node("generate_report", generate_report)

    # Edges
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "split_data")
    graph.add_edge("split_data", "human_review_domain")
    graph.add_conditional_edges(
        "human_review_domain",
        route_after_domain_review,
        {"propose_transformations": "propose_transformations", "analyze": "analyze"},
    )
    graph.add_edge("propose_transformations", "human_review_transformations")
    graph.add_conditional_edges(
        "human_review_transformations",
        route_after_transform_review,
        {"execute_transformations": "execute_transformations", "propose_transformations": "propose_transformations"},
    )
    graph.add_edge("execute_transformations", "propose_outlier_treatment")
    graph.add_edge("propose_outlier_treatment", "human_review_outliers")
    graph.add_conditional_edges(
        "human_review_outliers",
        route_after_outlier_review,
        {"execute_outlier_treatment": "execute_outlier_treatment", "propose_outlier_treatment": "propose_outlier_treatment"},
    )
    graph.add_edge("execute_outlier_treatment", "apply_scaling")
    graph.add_edge("apply_scaling", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile(checkpointer=checkpointer)
