# Human-in-the-Loop Data Preparation

A reproducible Python pipeline for reviewing and preparing tabular datasets before
model development. Deterministic code performs splitting, fitting and validation;
an optional LLM is limited to contextual suggestions that a person must approve.

This repository is an engineering case study, not an autonomous data-cleaning
product. It does not establish that a dataset is unbiased, lawful or suitable for
an operational model.

## What the system owns

The pipeline accepts a CSV and target column, then:

1. validates the source and profiles the schema;
2. creates a random, stratified, temporal or group-aware train/test split;
3. pauses for review of inferred context and possible leakage;
4. proposes and applies train-fitted imputers and encoders;
5. proposes and applies outlier treatment after another review;
6. scales the resulting matrices and writes an auditable report.

```mermaid
flowchart LR
  A["Validated CSV source"] --> B["Profile schema"]
  B --> C["Split before fit"]
  C --> D["Review context"]
  D --> E["Propose transformations"]
  E --> F["Review transformations"]
  F --> G["Fit on train; apply to test"]
  G --> H["Review outliers"]
  H --> I["Report and artifacts"]
```

The graph and its review routes are defined in
[`src/agents/pipeline.py`](src/agents/pipeline.py). A rejection returns to the
relevant proposal stage instead of silently continuing.

## Invariants enforced in code

- **Split before fit:** imputation values, encoders, outlier boundaries and
  scalers are learned from the training partition only.
- **Human approval at ambiguous boundaries:** domain context, transformations
  and outlier actions are explicit LangGraph interrupts.
- **Restricted data sources:** the API reads only configured local roots and
  explicitly allow-listed HTTPS hosts; redirects, private IP targets and files
  over the configured size limit are rejected.
- **Data minimization:** prompts receive schema and aggregate statistics, not
  raw rows.
- **Offline deterministic path:** the LLM client is created lazily and can be
  replaced with a fake through `configure_llm_client`; collecting or running
  deterministic tests does not require provider credentials.
- **Opt-in persistence:** the SQLite LLM response cache is disabled unless
  `LLM_CACHE_ENABLED=1` is set.

The security boundary is implemented in
[`src/security/input_sources.py`](src/security/input_sources.py), and the
dependency-injection seam is in
[`src/agents/dependencies.py`](src/agents/dependencies.py).

## Engineering decisions

| Decision | Alternative rejected | Reason and trade-off |
| --- | --- | --- |
| Split before transformation | Clean the full dataset and split afterward | Prevents leakage, at the cost of maintaining separate train-fitted state. |
| Deterministic rules for mechanical operations | Ask an LLM to choose every operation | Rules are testable and reproducible; the LLM is reserved for ambiguous context. |
| Explicit review interrupts | Apply every high-confidence proposal automatically | Review slows batch execution but preserves accountability for destructive choices. |
| Injectable, lazy LLM client | Construct the provider client at import time | Keeps tests and rules usable without credentials and separates provider contracts. |
| Allow-listed input sources | Accept arbitrary paths and URLs | Reduces SSRF and local-file exposure; operators must configure legitimate sources. |
| Pydantic response contracts | Parse free-form model prose | Invalid outputs fail visibly or retry; schema evolution requires deliberate changes. |

LangGraph is used because the workflow contains resumable approval points and
back-edges. A linear script would be simpler for a fully automatic pipeline, but
would obscure review state and rejection routing.

## Run locally

Requirements: Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q
```

The deterministic path needs no API key. To enable contextual suggestions, copy
`.env.example` to `.env` and set:

```env
OPENAI_API_KEY=...
LLM_PROVIDER=openai
LLM_MODEL=gpt-5-mini
```

### Streamlit review interface

```bash
uv run streamlit run app_streamlit.py
```

### Command line

```bash
uv run python run.py data/example.csv target
uv run python run.py data/events.csv status --strategy temporal --time-col timestamp
uv run python run.py data/patients.csv outcome --strategy group --group-col patient_id
```

### Local API

```bash
uv run uvicorn src.api.app:app --reload
```

By default the API reads CSV files only from `data/` and refuses URLs.
`DATASET_ALLOWED_ROOTS` adds local roots and `DATASET_ALLOWED_HOSTS` adds exact
HTTPS hosts. `MAX_DATASET_BYTES` defaults to 25 MiB.

## How to review the implementation

| Question | Start here |
| --- | --- |
| Where are graph transitions and review loops defined? | `src/agents/pipeline.py` |
| Where is split-before-fit enforced? | `src/agents/base_agent.py`, `src/agents/transformation_agent.py` |
| How are LLM outputs constrained? | `src/agents/schemas.py`, `src/llm/gpt_client.py` |
| How are credentials decoupled from tests? | `src/agents/dependencies.py`, `tests/test_security.py` |
| How are file and URL inputs restricted? | `src/security/input_sources.py`, `src/api/app.py` |
| Where are prompts versioned? | `config/prompt_templates.yaml` |

## Repository map

```text
config/                    Versioned model, logging and prompt configuration
src/agents/                State, rules, proposals and LangGraph orchestration
src/api/                   FastAPI integration surface
src/llm/                   Provider client, retries and optional cache
src/security/              Local path and remote-source validation
src/prompt_engineering/    Prompt templates and structured parsing
tests/                     Statistical, transformation and security contracts
app_streamlit.py           Interactive review UI
run.py                     CLI entry point
```

## Verification and evidence boundary

Run the complete local suite with:

```bash
uv run pytest -q
```

The hardened reference run passes 84 tests covering deterministic rules,
split strategies, transformation state, outlier behavior, fake-client injection
and input-source controls. These tests validate software contracts; they do not
prove that a suggested transformation is appropriate for an unseen business
domain. Live-provider quality, cost and latency require a separate gated
evaluation with a declared dataset and prompt version.

## Known limitations

- Human reviewers can reject indefinitely; the graph intentionally has no
  automatic retry ceiling.
- Aggregate metadata can still be sensitive in small or sparse datasets.
- Statistical outlier rules can remove rare but valid cases.
- The baseline report is a pipeline check, not a model-selection study.
- A production service still needs identity, authorization, encrypted storage,
  retention policy, observability and dataset-specific governance.

## License

See the repository license and the licenses of any datasets processed with it.
