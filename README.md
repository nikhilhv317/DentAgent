# dentaVIS — Agentic AI for Dental Practices

**Local-first AI agents that draft grounded, cited clinical notes, screen dental images and run the clinic's workflow, with a clinician approving every action.**

Everything runs on the clinic's own machine (Ollama / transformers), so patient
data never leaves it. Nine agents share one registry and one output contract:

- **Clinical Assistant & Voice-to-SOAP:** SOAP notes from typed or live-dictated findings (Whisper + Silero, CPU), every sentence cited to its source, with a fine-tuned LoRA model.
- **Imaging:** a dental-image gate, then YOLO lesion detection.
- **Visit Pipeline:** imaging → SOAP → report, pausing for clinician review at each step (LangGraph, durable).
- **Clinic Copilot:** a tool-calling chat over the records, where writes come back as proposals for approval.
- **Patient Intelligence, Report, Smart Scheduling (no-show risk), Model Retraining.**

Grounding, citation and safety checks run on every draft, and an evaluation suite measures them (see [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)).

## Stack

```
HTML/JS (dentaVIS UI) → HTTP/JSON → FastAPI → Agents → Ollama (local)
                                       │         │
                                       │         └─ scikit-learn (no-show model)
                                     SQLite         LangGraph (scheduling flow)
```

Division of labour in the scheduling agent — the important idea:
- **scikit-learn LogisticRegression** produces the risk NUMBER (reproducible, explainable).
- **qwen3:4b via Ollama** produces only the WORDS (reason line, reminder, summary).
- **LangGraph** sequences it and PAUSES at approval via `interrupt()`, checkpointed
  to SQLite so a paused run survives a restart and resumes on approve.

## One-time setup

Install Ollama (https://ollama.com), then:

```bat
ollama pull qwen3:4b
setup.bat
```

`setup.bat` installs packages, prepares the appointment history, and trains the model.

## Run

```bat
run.bat
```

Opens http://127.0.0.1:8000. Or manually: `uvicorn app:app`.

- **AI Summary**: pick a patient → Generate brief.
- **Appointments**: opens today's scored schedule; tick the reminders to send → Approve.

A green **Local model** tag means the LLM answered; **Demo (offline)** means Ollama
wasn't reachable and a deterministic fallback ran (the app never hard-fails).

## Files

| Path | Role |
|------|------|
| `app.py` | FastAPI — serves UI + both agents' APIs + morning job |
| `db.py` | SQLite schema, seed, appointment/log helpers |
| `contracts.py` | Shared agent contract (Insights→Recs→Actions) |
| `llm.py` | Ollama client for Patient Intelligence (+ demo fallback) |
| `agents/base.py` | BaseAgent lifecycle |
| `agents/patient_intelligence.py` | Patient Intelligence agent |
| `agents/scheduling_agent.py` | Smart Scheduling agent (LangGraph graph + runner) |
| `agents/noshow_features.py` | One feature definition — shared train ↔ serve |
| `agents/copilot.py` | Clinic Copilot — tool-calling agent (model picks tools; writes are proposals only) |
| `agents/__init__.py` | Registry + action handlers (Patient Intelligence) |
| `ml/generate_data.py` | Appointment history + today's schedule |
| `ml/train.py` | Trains the model → `models/noshow.pkl` |
| `models/noshow.pkl` | Trained logistic-regression pipeline (regenerate locally) |
| `static/index.html` | dentaVIS UI (AI Summary + Appointments) |
| `setup.bat` / `run.bat` | Windows one-time setup / daily launch |

## Model notes

- Target AUC ~0.73 — deliberately realistic, not a fake 0.95. No-show is partly
  random; a flagged patient is right ~half the time, which is why the only action
  on a prediction is a soft reminder, never an auto-cancel.
- Always `python ml\train.py` on your own machine so `noshow.pkl` matches your
  installed scikit-learn version.
- The scheduling model uses behavioural features only (prior attendance, lead time,
  type, confirmation, hour/day) — no demographics, by design.

## Visit Pipeline (event-driven, durable, human-in-the-loop)

`agents/visit_pipeline.py` chains the agents for one visit and pauses for a
clinician at each decision (**Visit Pipeline** page):

```
imaging finding approved ─▶ gather imaging ─▶ Clinical agent drafts SOAP ─▶ ⏸ review
   ─▶ Report agent drafts report ─▶ ⏸ review ─▶ finalize + add visit to history
```

- **Event trigger:** approving an imaging finding on the Scan page starts it. A
  second finding approved while the SOAP note awaits review triggers a redraft.
- **Pauses:** LangGraph `interrupt()`; the clinician approves (optionally edited),
  redrafts or rejects, passed back via `Command(resume=...)`. Who decided is logged.
- **Durable:** state is checkpointed to `pipeline_checkpoints.db` (next to
  `dpms.db`). Paused runs survive restarts; a run that was mid-step when the
  server stopped continues from its last checkpoint on startup.
- **Safety check:** every draft is compared with the record before review —
  each recorded allergy must be mentioned and not negated, each condition
  mentioned. (Testing caught qwen3:1.7b turning a history note "Latex-free
  gloves" into "patient is latex-free" for a latex-allergic patient.)
- **Grounding check** (`agents/grounding.py`): every clinical finding in a draft
  must trace to a source. The SOAP Objective may only use *today's* complaint
  and imaging; Subjective/Assessment may also use visit history; the report may
  also use the clinician-approved SOAP note. It flags invented exams
  ("examination reveals…" with no exam entered), unsourced findings, findings
  copied from a past visit as if current, and unsourced tooth numbers / mm
  measurements. Hedged mentions ("possible infection") aren't flagged. It's a
  vocabulary-based check: it narrows what the reviewer must verify, it doesn't
  replace the reviewer. The SOAP part lives in the Clinical Assistant
  (`clinical_assistant.ground_soap`), so `/api/clinical/draft` returns the same
  `unsupported` list (plus a "Grounding check" insight) on every draft; an
  examination may only be described when the clinician typed the findings.
  The report part lives in the Report agent (`report_agent.ground_report`), so
  the **Create Report** page shows the same warning before you finalize. Its
  sources are the record, imaging and clinician findings; visit history is not
  in its prompt, so a finding found only there is flagged as from a past visit.

## Clinic Copilot (tool-calling agent)

The **Copilot** page is a chat where the *model* decides which steps to take,
unlike the other agents whose steps are fixed in code. It loops agent → tools →
agent (LangGraph, max 6 rounds) over:

- **Read tools** (run freely): `find_patients`, `get_patient_record`,
  `get_patient_notes`, `get_imaging_findings`, `get_schedule`.
- **Write tools** (proposals only): `propose_followup`, `propose_chart_note`,
  `propose_patient_message`. They appear as **Approve** buttons; nothing is
  written or sent until a human approves, and the log records who approved.

Guards: a patient_id is only accepted after a lookup returned it (small models
guess ids), an answer that claims an action with nothing behind it gets a
visible warning, and the **grounding check** (`agents/grounding.py`, same as
the visit pipeline) flags clinical findings in the answer that appear in no
tool result or chat message — e.g. it caught "active infection" claimed from a
year-old abscess record. Negations ("no caries on record") are allowed. Each reply shows its tool calls and the model's reasoning.

Model: `qwen3:1.7b` with thinking on (set `COPILOT_MODEL` to change). With
thinking off it describes tool calls instead of making them. The local
`qwen3:4b` tag is the thinking-only 2507 build and leaks its reasoning into
answers — use a non-thinking build such as `qwen3:4b-instruct-2507` if you want
a bigger model. On CPU expect roughly 40–100 s per question.

## Adding the next agent (e.g. Imaging)

Same shape: a module in `agents/` returning the shared contract, plus its screen.
Imaging needs a vision-capable local model (e.g. a Qwen-VL or llama3.2-vision tag)
since it reads an image rather than a record.

## Strategy

Where the project stands against the market, and a prioritised roadmap: [docs/STRATEGY.md](docs/STRATEGY.md).

## Evaluations

`python evals/run.py` runs 44 offline checker cases (also in CI on every push).
`python evals/run.py --suite live` measures the drafting agents' grounding-failure
and safety-violation rates and the Copilot's tool choice on the local model.
Cases, known gaps and the current baseline: [evals/README.md](evals/README.md).

## MCP server

`mcp_server.py` exposes the agent registry and the Copilot's tools over the
Model Context Protocol (stdio), so other AI assistants can use dentaVIS:
`list_agents`, `run_agent` (admin-only and file-upload agents excluded), the
read tools (`find_patients`, `get_patient_record`, `get_patient_notes`,
`get_imaging_findings`, `search_patient_history`, `get_schedule`) and the
`propose_*` tools, which only create actions a clinician approves in dentaVIS.

Claude Code:

```bat
claude mcp add dentavis -- D:\medrag_transfer\.venv\Scripts\python.exe D:\Agents\mcp_server.py
```

Claude Desktop (`claude_desktop_config.json`):

```json
{"mcpServers": {"dentavis": {
  "command": "D:\medrag_transfer\.venv\Scripts\python.exe",
  "args": ["D:\Agents\mcp_server.py"],
  "env": {"DPMS_DB_PATH": "D:\Agents\dpms.db"}}}}
```

**Privacy:** the server itself is local, but whatever it returns goes to the
client that called it. A cloud-hosted assistant therefore receives patient data,
which gives up the "nothing leaves the clinic" property. Use de-identified data,
or a client running a local model, for real patients.

## Voice-to-SOAP (live)

**Create Report → 🎙 Dictate SOAP note** streams the microphone to the server
over a WebSocket (`/ws/dictation`; 16 kHz PCM from an AudioWorklet). **Silero
VAD** cuts the speech at each pause, and **Whisper small** transcribes that
utterance on the **CPU** while the next one is being spoken. You see the text
after every pause. On **Stop**, only the last sentence is left: the transcript
arrives about 1 s later and the grounded, cited SOAP note a few seconds after
that. Measured with a 16 s dictation: 0.8 s to transcript and 6.8 s to the
finished note after Stop, against ~14 s just to transcribe with the old
record-then-upload flow.

- Everything is local and CPU-only (no GPU libraries for speech). The token is
  sent as the first WebSocket message, not in the URL. Clinician or admin role,
  own clinic's patients only.
- Whisper, Silero and the drafting model are loaded at app start in the
  background (`DPMS_WARMUP=0` to skip), so the first dictation doesn't wait for
  model loads.
- The dictation is split at "on examination…" into complaint and findings. A
  dental vocabulary prompt and a context-only fix ("distal carries" → "caries")
  improve terms.
- `POST /api/clinical/voice` (upload a recording) still works as a fallback.

## Per-patient memory

`agents/memory.py` embeds every visit, chart note and finalized report with
the local `nomic-embed-text` model and searches them by meaning. The index is
built lazily the first time a patient is searched. Used by Patient
Intelligence, the SOAP note prompt (for long histories) and the Copilot's
`search_patient_history` tool. Falls back to keyword matching if the embedding
model isn't available.

## Tracing and health

Every agent run, model call (with token counts), Copilot tool call, pipeline
step and approval (with who approved) is recorded as an OpenTelemetry span in
the local database. **Settings → System health & agent traces**, or
`GET /api/health` and `GET /api/traces`. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to
also export to a collector.

## Data exchange

| What | Endpoint |
|---|---|
| Export a chart as FHIR R4 Bundle (validated) | `GET /api/patients/{id}/fhir` (FHIR button on Patients) |
| Import FHIR Patient / Bundle | `POST /api/interop/fhir` |
| HL7 v2 ADT^A04/A08 registration feed (returns ACK) | `POST /api/interop/hl7` (raw message body) |
| Practice-management CSV export | `POST /api/interop/csv/patients`, `/csv/appointments` |
| DICOM image → imaging agent | `POST /api/imaging/dicom` |

Imports reuse an existing chart only when **both** name and phone match.

## Roles, encryption and clinic separation

- **Roles:**
  - **admin:** everything, including staff, data imports and retraining.
  - **clinician:** clinical drafts, approvals, report sign-off and deleting charts.
  - **front_desk:** schedule, patients and booking.
  Role and active status are checked on every request.
- **Encryption at rest:** contact details and clinical text (allergies,
  conditions, visits, notes, reports, scan images) are Fernet-encrypted in
  SQLite. The key is `DPMS_ENC_KEY`, or `dpms.key` next to the database (created
  on first run). **Back the key up separately; without it the encrypted fields
  can't be recovered.** Existing databases are encrypted in place on first
  start.
- **Clinic separation:** staff only see patients of their own clinic.
- **`.env`:** loaded at startup. A persistent `JWT_SECRET` is written there if
  missing, so logins survive restarts.

## Fine-tuned model (LoRA)

`ml/lora/` builds a grounded training set from the app's own prompts, trains a LoRA adapter on Qwen3-1.7B, and packages it for Ollama as
`dentavis-clinical`:

```bat
python ml\lora\make_data.py
python ml\lora\train.py --epochs 2
python ml\lora\train.py --merge
python ml\lora\to_ollama.py
python evals\run.py --suite drafting --model dentavis-clinical
```

Results (`evals/README.md`): on the drafting evals the fine-tuned model has 0%
invented content, 0% safety violations, 0% citation failures and 0/8
clinical-relevance misses. The base qwen3:1.7b scores 19%, 6%, 75% and 1/8. The
SOAP and report agents use it automatically when `ml/lora/out/merged` exists
(it runs through transformers, because Ollama 0.40 on Windows can't import
safetensors); otherwise they use qwen3:1.7b.
