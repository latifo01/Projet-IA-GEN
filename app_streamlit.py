"""Interface Streamlit pour le pipeline de preprocessing.

Chaque session Streamlit obtient un thread_id unique. Le SqliteSaver persiste
l'etat entre les reruns Streamlit (chaque clic de bouton est un rerun).

Lancement :
  uv run streamlit run app_streamlit.py
"""

import json
import sqlite3
import uuid
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.agents.pipeline import build_pipeline

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

_DB_PATH = Path("data/cache/checkpoints.db")


def _get_pipeline():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return build_pipeline(checkpointer=checkpointer)


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _extract_interrupt(snapshot):
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        return snapshot.tasks[0].interrupts[0].value
    return None


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "session_id": None,
        "stage": "start",      # start | domain | transform | leakage | outlier | done
        "interrupt": None,
        "report": None,
        "quality_score": None,
        "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Display helpers (no decorative separators per code_instructions.md)
# ---------------------------------------------------------------------------

def _show_domain(interrupt: dict):
    domain = interrupt.get("domain_context", {})
    st.subheader("🧠 Domaine inféré")
    st.write(f"**Domaine :** {domain.get('domain', '?')}")
    st.write(f"**Description :** {domain.get('description', '?')}")
    if domain.get("constraints"):
        st.write("**Contraintes métier :**")
        for c in domain["constraints"]:
            st.write(f"- {c}")
    if domain.get("sensitive_columns"):
        st.write(f"**Colonnes sensibles :** {domain['sensitive_columns']}")
    anomalies = interrupt.get("semantic_anomalies", [])
    if anomalies:
        st.subheader(f"⚠️ Anomalies sémantiques ({len(anomalies)})")
        rows = [{"Colonne": a["column"], "Problème": a["issue"], "Sévérité": a["severity"]}
                for a in anomalies]
        st.dataframe(rows, use_container_width=True)


def _show_transforms(interrupt: dict):
    proposals = interrupt.get("proposals", {})
    transforms = proposals.get("transformations", [])
    if transforms:
        st.subheader(f"✨ Transformations proposées ({len(transforms)})")
        rows = [{"Colonne": t["column"], "Action": t["action"],
                 "Source": t.get("source", "llm"), "Confiance": t.get("confidence", "?"),
                 "Raison": t.get("reason", "")}
                for t in transforms]
        st.dataframe(rows, use_container_width=True)

    fe = proposals.get("feature_engineering", [])
    if fe:
        st.subheader(f"💡 Feature engineering ({len(fe)} suggestions)")
        rows = [{"Nom": f["name"], "Formule": f["formula"], "Raison": f["reason"]}
                for f in fe]
        st.dataframe(rows, use_container_width=True)

    leakage = interrupt.get("leakage_alerts", [])
    if leakage:
        st.warning(f"🚨 {len(leakage)} alerte(s) de leakage détectée(s)")
        rows = [{"Colonne": l["column"], "Corrélation": l["correlation"], "Raison": l["reason"]}
                for l in leakage]
        st.dataframe(rows, use_container_width=True)


def _show_leakage(interrupt: dict):
    st.warning("🚨 " + interrupt.get("message", "Leakage détecté"))
    leakage = interrupt.get("leakage_alerts", [])
    rows = [{"Colonne": l["column"], "Corrélation": l["correlation"], "Raison": l["reason"]}
            for l in leakage]
    st.dataframe(rows, use_container_width=True)


def _show_outliers(interrupt: dict):
    proposals = interrupt.get("proposals", {})
    report = proposals.get("outlier_report", [])
    if report:
        st.subheader(f"📊 Détection outliers ({len(report)} colonnes)")
        rows = [{"Colonne": r["column"], "N outliers": r["n_outliers"],
                 "Pct": r["pct_outliers"], "Méthode": r["method"],
                 "Borne inf": r["lower_bound"], "Borne sup": r["upper_bound"]}
                for r in report]
        st.dataframe(rows, use_container_width=True)

    actions = proposals.get("outlier_actions", [])
    if actions:
        st.subheader(f"🛠️ Traitements proposés ({len(actions)})")
        rows = [{"Colonne": a["column"], "Méthode": a["method"], "Action": a["action"],
                 "Confiance": a.get("confidence", "?"), "Raison": a.get("reason", "")}
                for a in actions]
        st.dataframe(rows, use_container_width=True)

    flagged = interrupt.get("requires_review", [])
    if flagged:
        st.warning(f"⚠️ Colonnes à examiner attentivement : {flagged}")


def _show_report(report: dict, quality_score: dict):
    overall = quality_score.get("overall", 0)
    st.metric("Taux de qualité global", f"{overall}/100")

    col1, col2, col3 = st.columns(3)
    col1.metric("Complétude", f"{quality_score.get('completeness', '?')}/30")
    col2.metric("Types valides", f"{quality_score.get('type_consistency', '?')}/15")
    col3.metric("Outliers maîtrisés", f"{quality_score.get('outlier_coverage', '?')}/20")

    baseline = report.get("baseline_evaluation", {})
    if baseline.get("status") == "ok":
        st.subheader("📈 Évaluation Baseline")
        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Dummy", baseline["baseline_score"])
        col2.metric("Modèle Linéaire", baseline["model_score"])
        col3.metric("Delta de gain", f"+{baseline['delta']}" if baseline["delta"] >= 0 else str(baseline["delta"]))

    warnings = report.get("critical_warnings", [])
    if warnings:
        st.subheader("🔥 Alertes critiques")
        for w in warnings:
            st.error(f"[{w['type']}] {w.get('column', '')} : {w['detail']}")

    split = report.get("train_test_split", {})
    if split:
        st.subheader("✂️ Split Train/Test")
        st.write(f"Train : {split.get('train_size', '?')} lignes | Test : {split.get('test_size', '?')} lignes")
        st.write(f"Fichier train : `{split.get('train_path', '?')}`")
        st.write(f"Fichier test : `{split.get('test_path', '?')}`")

    if report.get("prompt_templates_version"):
        st.caption(f"Prompts version : {report['prompt_templates_version']}")

    st.subheader("📄 Rapport complet (JSON)")
    st.json(report)


# ---------------------------------------------------------------------------
# Decision widget (approve / reject + feedback)
# ---------------------------------------------------------------------------

def _decision_widget(key_prefix: str) -> tuple[str | None, str]:
    """Return (decision, feedback). decision is None if user has not clicked yet."""
    feedback = st.text_area("Feedback (optionnel, en cas de rejet)", key=f"{key_prefix}_feedback")
    col1, col2 = st.columns(2)
    if col1.button("Approuver", key=f"{key_prefix}_approve", type="primary"):
        return "approve", ""
    if col2.button("Rejeter", key=f"{key_prefix}_reject"):
        return "reject", feedback
    return None, ""


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

st.set_page_config(page_title="GenAI Preprocessing Pipeline", layout="wide", page_icon="🚀")
st.title("🚀 GenAI Preprocessing Pipeline")

_init_state()

if st.session_state.error:
    st.error(st.session_state.error)
    if st.button("Recommencer"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    st.stop()

# -- Etape 1 : Formulaire de lancement --
if st.session_state.stage == "start":
    with st.form("start_form"):
        df_path = st.text_input("Chemin local ou URL du fichier CSV",
                                placeholder="data/titanic.csv")
        target = st.text_input("Colonne cible", placeholder="survived")

        st.write("**Configuration du split (optionnel)**")
        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("Test size", 0.05, 0.5, 0.2, step=0.05)
        random_state = c2.number_input("Random state", value=42, step=1)
        strategy = c3.selectbox("Strategie", ["auto", "stratified", "temporal", "group"])
        time_col = st.text_input("Colonne temporelle (strategy=temporal)", "")
        group_col = st.text_input("Colonne de groupe (strategy=group)", "")

        submitted = st.form_submit_button("Lancer le pipeline")

    if submitted:
        if not df_path or not target:
            st.error("Le chemin CSV et la colonne cible sont obligatoires.")
        else:
            split_config = {
                "test_size": test_size,
                "random_state": int(random_state),
                "strategy": strategy,
                "time_col": time_col or None,
                "group_col": group_col or None,
            }
            session_id = str(uuid.uuid4())
            st.session_state.session_id = session_id
            pipeline = _get_pipeline()
            config = _config(session_id)
            with st.spinner("Analyse en cours..."):
                try:
                    pipeline.invoke(
                        {"df_path": df_path, "target": target, "split_config": split_config},
                        config=config,
                    )
                except Exception as e:
                    st.session_state.error = str(e)
                    st.rerun()
            snapshot = pipeline.get_state(config)
            interrupt = _extract_interrupt(snapshot)
            if interrupt:
                st.session_state.interrupt = interrupt
                st.session_state.stage = "domain"
            st.rerun()

# -- Etape 2 : Revue du domaine --
elif st.session_state.stage == "domain":
    st.subheader("🧠 Revue du contexte métier")
    _show_domain(st.session_state.interrupt)
    decision, feedback = _decision_widget("domain")
    if decision:
        pipeline = _get_pipeline()
        config = _config(st.session_state.session_id)
        resume_val = decision if not feedback else {"decision": decision, "feedback": feedback}
        with st.spinner("Reprise du pipeline..."):
            pipeline.invoke(Command(resume=resume_val), config=config)
        snapshot = pipeline.get_state(config)
        interrupt = _extract_interrupt(snapshot)
        if interrupt:
            st.session_state.interrupt = interrupt
            next_type = interrupt.get("type", "")
            st.session_state.stage = "domain" if decision == "reject" else "transform"
        else:
            st.session_state.stage = "done"
            st.session_state.report = snapshot.values.get("report")
            st.session_state.quality_score = snapshot.values.get("quality_score")
        st.rerun()

# -- Etape 3 : Revue des transformations --
elif st.session_state.stage == "transform":
    st.subheader("✨ Revue des transformations proposées")
    _show_transforms(st.session_state.interrupt)
    decision, feedback = _decision_widget("transform")
    if decision:
        pipeline = _get_pipeline()
        config = _config(st.session_state.session_id)
        resume_val = decision if not feedback else {"decision": decision, "feedback": feedback}
        with st.spinner("Application des transformations..."):
            pipeline.invoke(Command(resume=resume_val), config=config)
        snapshot = pipeline.get_state(config)
        interrupt = _extract_interrupt(snapshot)
        if interrupt:
            st.session_state.interrupt = interrupt
            next_type = interrupt.get("type", "")
            if next_type == "leakage_warning":
                st.session_state.stage = "leakage"
            elif next_type == "outlier_review":
                st.session_state.stage = "outlier"
            else:
                st.session_state.stage = "transform"
        else:
            st.session_state.stage = "done"
            st.session_state.report = snapshot.values.get("report")
            st.session_state.quality_score = snapshot.values.get("quality_score")
        st.rerun()

# -- Etape 3b : Alerte leakage --
elif st.session_state.stage == "leakage":
    st.subheader("🚨 Alerte Leakage")
    _show_leakage(st.session_state.interrupt)
    decision, _ = _decision_widget("leakage")
    if decision:
        pipeline = _get_pipeline()
        config = _config(st.session_state.session_id)
        with st.spinner("Traitement du leakage..."):
            pipeline.invoke(Command(resume=decision), config=config)
        snapshot = pipeline.get_state(config)
        interrupt = _extract_interrupt(snapshot)
        if interrupt:
            st.session_state.interrupt = interrupt
            next_type = interrupt.get("type", "")
            st.session_state.stage = "outlier" if next_type == "outlier_review" else "transform"
        else:
            st.session_state.stage = "done"
            st.session_state.report = snapshot.values.get("report")
            st.session_state.quality_score = snapshot.values.get("quality_score")
        st.rerun()

# -- Etape 4 : Revue des outliers --
elif st.session_state.stage == "outlier":
    st.subheader("📊 Revue du traitement des outliers")
    _show_outliers(st.session_state.interrupt)
    decision, feedback = _decision_widget("outlier")
    if decision:
        pipeline = _get_pipeline()
        config = _config(st.session_state.session_id)
        resume_val = decision if not feedback else {"decision": decision, "feedback": feedback}
        with st.spinner("Traitement des outliers et generation du rapport..."):
            result = pipeline.invoke(Command(resume=resume_val), config=config)
        snapshot = pipeline.get_state(config)
        interrupt = _extract_interrupt(snapshot)
        if interrupt:
            st.session_state.interrupt = interrupt
            st.session_state.stage = "outlier"
        else:
            st.session_state.stage = "done"
            report = result.get("report") if isinstance(result, dict) else None
            st.session_state.report = report or snapshot.values.get("report")
            st.session_state.quality_score = snapshot.values.get("quality_score")
        st.rerun()

# -- Etape 5 : Rapport final --
elif st.session_state.stage == "done":
    st.subheader("✅ Pipeline Terminé")
    if st.session_state.report:
        _show_report(st.session_state.report, st.session_state.quality_score or {})
        report_json = json.dumps(st.session_state.report, ensure_ascii=False, indent=2)
        st.download_button(
            label="Télécharger le rapport JSON",
            data=report_json,
            file_name="report.json",
            mime="application/json",
            type="primary",
        )
    if st.button("Nouveau pipeline"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
