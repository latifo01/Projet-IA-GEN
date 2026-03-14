"""API REST FastAPI pour le pipeline de preprocessing.

Endpoints :
  POST /pipeline/start          Demarre un pipeline, retourne l'interrupt domain_review
  POST /pipeline/resume         Reprend avec une decision humaine
  GET  /pipeline/{session_id}/state   Etat courant (interrupt en attente ou rapport final)

Chaque session est identifiee par un UUID utilise comme thread_id LangGraph.
Le SqliteSaver persiste les checkpoints entre requetes (et entre redemarrages).

Lancement :
  uv run uvicorn src.api.app:app --reload
"""

import sqlite3
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agents.pipeline import build_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pipeline singleton (shared across requests)
# ---------------------------------------------------------------------------

_DB_PATH = Path("data/cache/checkpoints.db")
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
_checkpointer = SqliteSaver(_conn)
_pipeline = build_pipeline(checkpointer=_checkpointer)


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _extract_interrupt(snapshot) -> dict | None:
    """Return the first pending interrupt value, or None if pipeline is done."""
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        return snapshot.tasks[0].interrupts[0].value
    return None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GenAI Preprocessing Pipeline",
    description="Pipeline de preprocessing ML pilote par GPT avec human-in-the-loop.",
    version="1.0.0",
)


# -- Request / Response models --

class SplitConfig(BaseModel):
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    random_state: int = 42
    strategy: str = Field("auto", pattern="^(auto|stratified|temporal|group)$")
    time_col: str | None = None
    group_col: str | None = None


class StartRequest(BaseModel):
    df_path: str = Field(..., description="Chemin local ou URL du fichier CSV")
    target: str = Field(..., description="Nom de la colonne cible")
    split_config: SplitConfig = Field(default_factory=SplitConfig)


class ResumeRequest(BaseModel):
    session_id: str
    decision: str = Field(..., pattern="^(approve|reject)$")
    feedback: str | None = Field(None, description="Commentaire facultatif en cas de rejet")


class InterruptResponse(BaseModel):
    session_id: str
    status: str          # "waiting_for_review" | "completed"
    interrupt: dict | None = None
    report: dict | None = None


# -- Endpoints --

@app.post("/pipeline/start", response_model=InterruptResponse)
def start_pipeline(req: StartRequest):
    """Demarre un nouveau pipeline. Retourne le premier point d'interruption (domain_review)."""
    session_id = str(uuid.uuid4())
    config = _config(session_id)

    initial_state = {
        "df_path": req.df_path,
        "target": req.target,
        "split_config": req.split_config.model_dump(),
    }

    try:
        _pipeline.invoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    snapshot = _pipeline.get_state(config)
    interrupt = _extract_interrupt(snapshot)

    if interrupt:
        return InterruptResponse(session_id=session_id, status="waiting_for_review", interrupt=interrupt)

    # Pipeline finished without interrupt (unlikely on first call)
    return InterruptResponse(
        session_id=session_id,
        status="completed",
        report=snapshot.values.get("report"),
    )


@app.post("/pipeline/resume", response_model=InterruptResponse)
def resume_pipeline(req: ResumeRequest):
    """Reprend le pipeline avec la decision humaine (approve/reject + feedback optionnel)."""
    config = _config(req.session_id)

    snapshot = _pipeline.get_state(config)
    if not snapshot.tasks:
        raise HTTPException(status_code=404, detail="Session introuvable ou pipeline deja termine.")

    decision: str | dict = req.decision
    if req.feedback and req.decision == "reject":
        decision = {"decision": "reject", "feedback": req.feedback}

    try:
        result = _pipeline.invoke(Command(resume=decision), config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    snapshot = _pipeline.get_state(config)
    interrupt = _extract_interrupt(snapshot)

    if interrupt:
        return InterruptResponse(
            session_id=req.session_id,
            status="waiting_for_review",
            interrupt=interrupt,
        )

    report = result.get("report") if isinstance(result, dict) else snapshot.values.get("report")
    return InterruptResponse(session_id=req.session_id, status="completed", report=report)


@app.get("/pipeline/{session_id}/state", response_model=InterruptResponse)
def get_pipeline_state(session_id: str):
    """Retourne l'etat courant d'une session (interrupt en attente ou rapport final)."""
    config = _config(session_id)
    snapshot = _pipeline.get_state(config)

    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Session introuvable.")

    interrupt = _extract_interrupt(snapshot)
    if interrupt:
        return InterruptResponse(session_id=session_id, status="waiting_for_review", interrupt=interrupt)

    return InterruptResponse(
        session_id=session_id,
        status="completed",
        report=snapshot.values.get("report"),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
