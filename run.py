"""Point d'entree pour lancer le pipeline de preprocessing complet.

Usage :
  uv run python run.py <csv_path> <target> [options]

Options :
  --test-size FLOAT      Proportion du jeu de test (defaut: 0.2)
  --random-state INT     Graine aleatoire (defaut: 42)
  --strategy STR         auto | stratified | temporal | group (defaut: auto)
  --time-col STR         Colonne temporelle pour strategy=temporal
  --group-col STR        Colonne de groupe pour strategy=group
"""

import argparse
import json
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.agents.pipeline import build_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _display_domain_review(interrupt_value):
    """Display domain context for human review."""
    domain = interrupt_value.get("domain_context", {})
    anomalies = interrupt_value.get("semantic_anomalies", [])

    print("\nDomain context infere")
    print(f"  Domaine : {domain.get('domain', '?')}")
    print(f"  Description : {domain.get('description', '?')}")
    if domain.get("constraints"):
        print("  Contraintes metier :")
        for c in domain["constraints"]:
            print(f"    - {c}")
    if domain.get("sensitive_columns"):
        print(f"  Colonnes sensibles : {domain['sensitive_columns']}")
    if anomalies:
        print(f"  Anomalies semantiques ({len(anomalies)}) :")
        for a in anomalies:
            print(f"    - {a['column']}: {a['issue']} (severity: {a['severity']})")


def _display_transform_review(interrupt_value):
    """Display transformation proposals for human review."""
    proposals = interrupt_value.get("proposals", {})
    leakage = interrupt_value.get("leakage_alerts", [])

    transforms = proposals.get("transformations", [])
    print(f"\nPropositions de transformations ({len(transforms)})")
    for t in transforms:
        source = t.get("source", "llm")
        confidence = t.get("confidence", "?")
        print(f"  [{source}/{confidence}] {t['column']}: {t['action']} -- {t.get('reason', '')}")

    fe = proposals.get("feature_engineering", [])
    if fe:
        print(f"\nFeature engineering (suggestions textuelles, {len(fe)})")
        for f in fe:
            print(f"  {f['name']}: {f['formula']} -- {f['reason']}")

    if leakage:
        print(f"\nAlertes de leakage ({len(leakage)})")
        for l in leakage:
            print(f"  {l['column']}: correlation={l['correlation']} -- {l['reason']}")


def _display_outlier_review(interrupt_value):
    """Display outlier treatment proposals for human review."""
    proposals = interrupt_value.get("proposals", {})

    report = proposals.get("outlier_report", [])
    if report:
        print(f"\nRapport outliers ({len(report)} colonnes)")
        for r in report:
            print(f"  {r['column']}: {r['n_outliers']} outliers ({r['pct_outliers']}%), "
                  f"method={r['method']}, bounds=[{r['lower_bound']}, {r['upper_bound']}]")

    actions = proposals.get("outlier_actions", [])
    if actions:
        print(f"\nTraitement propose ({len(actions)})")
        for a in actions:
            print(f"  {a['column']}: {a['method']}/{a['action']} -- {a.get('reason', '')}")


def _get_decision() -> str:
    """Get human decision from stdin."""
    decision = input("\nApprouver ? (approve/reject) : ").strip().lower()
    if decision not in ("approve", "reject"):
        decision = "approve"
    return decision


def run(df_path: str, target: str, split_config: dict | None = None):
    """Lance le pipeline complet avec human-in-the-loop."""

    checkpoint_dir = Path("data/cache")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(checkpoint_dir / "checkpoints.db"), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    pipeline = build_pipeline(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "session-1"}}

    # Step 1: Analysis + Domain inference -> pause for domain review
    logger.info("Lancement du pipeline sur '%s' (target: '%s')", df_path, target)
    initial_state = {"df_path": df_path, "target": target, "split_config": split_config or {}}
    pipeline.invoke(initial_state, config=config)

    snapshot = pipeline.get_state(config)
    interrupt_value = snapshot.tasks[0].interrupts[0].value
    _display_domain_review(interrupt_value)
    decision = _get_decision()

    # Step 2: Resume -> Transformation proposals -> pause for transform review
    pipeline.invoke(Command(resume=decision), config=config)

    snapshot = pipeline.get_state(config)
    interrupt_value = snapshot.tasks[0].interrupts[0].value
    _display_transform_review(interrupt_value)
    decision = _get_decision()

    # Step 2b: Resume -> possible leakage interrupt if approved + leakage detected
    pipeline.invoke(Command(resume=decision), config=config)

    snapshot = pipeline.get_state(config)
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        interrupt_value = snapshot.tasks[0].interrupts[0].value
        if interrupt_value.get("type") == "leakage_warning":
            print("\nALERTE LEAKAGE")
            for alert in interrupt_value.get("leakage_alerts", []):
                print(f"  {alert['column']}: correlation={alert['correlation']} -- {alert['reason']}")
            print(interrupt_value.get("message", ""))
            leakage_decision = _get_decision()
            pipeline.invoke(Command(resume=leakage_decision), config=config)

            snapshot = pipeline.get_state(config)

    # Check if we're at outlier review or need another resume
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        interrupt_value = snapshot.tasks[0].interrupts[0].value
        if interrupt_value.get("type") == "outlier_review":
            _display_outlier_review(interrupt_value)
            decision = _get_decision()

            # Step 4: Resume -> Execute outliers + Generate report
            state = pipeline.invoke(Command(resume=decision), config=config)
        else:
            state = snapshot.values
    else:
        state = snapshot.values

    # Display final report
    report = state.get("report", {})
    quality_score = state.get("quality_score", {})
    overall = quality_score.get("overall", 0) if isinstance(quality_score, dict) else quality_score
    print(f"\nRapport final (quality score: {overall}/100)")
    if isinstance(quality_score, dict):
        print(f"  Completeness: {quality_score.get('completeness', '?')}/30")
        print(f"  Type consistency: {quality_score.get('type_consistency', '?')}/15")
        print(f"  Leakage risk: {quality_score.get('leakage_risk', '?')}")
        print(f"  Outlier coverage: {quality_score.get('outlier_coverage', '?')}/20")
        print(f"  LLM drops penalty: {quality_score.get('drop_penalty', 0)}")
    split_info = report.get("train_test_split", {})
    if split_info:
        print(f"  Train/test split: {split_info.get('train_size', '?')} / {split_info.get('test_size', '?')}"
              f" (stratified: {split_info.get('stratified', False)})")
        print(f"  Train path: {split_info.get('train_path', '?')}")
        print(f"  Test path: {split_info.get('test_path', '?')}")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    logger.info("Pipeline termine. Rapport sauvegarde dans data/outputs/report.json")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de preprocessing ML pilote par GPT avec human-in-the-loop."
    )
    parser.add_argument("df_path", help="Chemin local ou URL du fichier CSV")
    parser.add_argument("target", help="Nom de la colonne cible")
    parser.add_argument("--test-size", type=float, default=0.2, metavar="FLOAT",
                        help="Proportion du jeu de test, ex: 0.2 (defaut)")
    parser.add_argument("--random-state", type=int, default=42, metavar="INT",
                        help="Graine aleatoire pour la reproductibilite (defaut: 42)")
    parser.add_argument("--strategy", default="auto",
                        choices=["auto", "stratified", "temporal", "group"],
                        help="Strategie de split : auto | stratified | temporal | group (defaut: auto)")
    parser.add_argument("--time-col", default=None, metavar="COL",
                        help="Colonne temporelle pour --strategy temporal")
    parser.add_argument("--group-col", default=None, metavar="COL",
                        help="Colonne de groupe pour --strategy group")

    args = parser.parse_args()

    split_cfg = {
        "test_size": args.test_size,
        "random_state": args.random_state,
        "strategy": args.strategy,
        "time_col": args.time_col,
        "group_col": args.group_col,
    }

    run(args.df_path, args.target, split_config=split_cfg)
