# DentAgent — Agentic Dental Practice System
### Full workflow & AI documentation

A local-first, agentic dental practice management prototype. One web app, one
Python backend, one database, four working AI agents — every model runs on the
clinic's own machine, so patient data never leaves it. This document covers the
entire system end to end: the architecture, every AI/ML component, each agent,
the clinical workflow, the data model, the API, and how to run it.

---

## Table of contents

1. What the product is
2. The stack (one whole product)
3. Core design principles
4. The AI/ML components (all of them)
5. The agents in detail
6. The end-to-end clinical workflow
7. Screen-by-screen reference
8. The data model
9. The API reference
10. Project file structure
11. How to run it
12. Speed / latency design
13. Safety & clinical framing
14. Honest limitations & next steps

---

## 1. What the product is

**DentAgent** is a dental clinic application that puts four AI agents behind an
ordinary clinic UI. A front-desk user triages the day's appointments for
no-show risk; then, per patient, the clinician screens dental images for disease
and generates a short clinical report. The distinguishing idea is that every AI
step **proposes**, and a human **approves** before anything is written to the
record — the agents remove the drudgery (scoring, drafting, detecting), not the
clinical judgement.

Everything runs locally:

- **No-show prediction** → a scikit-learn logistic-regression model.
- **Dental image screening** → a two-stage PyTorch vision pipeline (a "gate"
  classifier + a YOLO lesion detector).
- **Text generation** (report summaries) → a tiny local LLM via Ollama.

No cloud API, no API key, no patient data leaving the machine.

---

## 2. The stack (one whole product)

```
 HTML/JS (DentAgent single-page UI)
        │  HTTP / JSON
        ▼
 FastAPI  (app.py)  ── serves the UI and every agent API
        │
        ├── Agents (Python)
        │     • Smart Scheduling   → scikit-learn + LangGraph
        │     • Imaging & Diagnostic → PyTorch (EfficientNetV2 + YOLO) + LangGraph
        │     • Report             → tiny LLM (Ollama)
        │     • Patient Intelligence / Clinical SOAP (in codebase, off the main path)
        │
        ├── Local models
        │     • Ollama server (qwen3:1.7b)          — text
        │     • gate_best.pt (EfficientNetV2-S)     — "is this dental?"
        │     • detector_best.pt (YOLO, 8 classes)  — lesion detection
        │     • models/noshow.pkl (LogisticRegression) — no-show risk
        │
        └── SQLite (dpms.db)  — patients, visits, appointments, decisions,
                                notes, reports, agent actions & log
```

- **Frontend:** a single self-contained `static/index.html` (the DentAgent shell —
  login, sidebar, all screens). No build step, no framework. Voice input has been
  fully removed.
- **Backend:** FastAPI + Uvicorn. One process serves the page and all APIs.
- **Orchestration:** LangGraph expresses the scheduling and imaging agents as
  typed graphs (nodes + edges).
- **Scheduling:** APScheduler runs a "morning job" so the day's schedule is scored
  before anyone arrives.
- **Database:** SQLite — file-based, zero-config; fine for one clinic, and the
  schema carries over to Postgres for a multi-terminal deployment.

---

## 3. Core design principles

**a. One shared agent contract.** Every agent returns the same envelope —
**Insights → Recommendations → Actions**. Insights are read-only observations,
Recommendations are advisory next steps, and Actions are side effects that only
run after a human clicks **Approve**. Because all agents speak this shape, the
frontend renders any agent with one component and new agents drop in without
touching the API or the UI renderer.

**b. Human-in-the-loop by construction.** Agents only ever *propose*. A proposed
action is persisted with status `proposed`; nothing happens until
`POST /api/actions/{id}/approve` fires the matching handler, which then executes
and writes to the log/record. This is enforced structurally, not by convention —
the right shape for anything clinical.

**c. Local, private inference.** Text generation goes through Ollama on
`localhost`; the vision models are the clinic's own `.pt` files; the no-show
model is a local pickle. Nothing is sent to a third party.

**d. The right tool for each job.** The *intelligence* lives in purpose-built
models (logistic regression for risk, trained CNNs for images), not the LLM. The
LLM does *language* only (short summaries). This keeps predictions reproducible
and explainable and keeps the LLM's job small and fast.

**e. Speed first.** Deterministic models are instant; the LLM is a tiny model
with its "thinking" pass disabled and kept warm in memory; vision runs batched.

**f. Persistence closes the loop.** Approved imaging findings and finalized
reports are written to the patient's record, so a later step (or a later visit)
can read what happened earlier. Appointment outcomes (showed / no-show) are
captured — the signal a future model would retrain on.

---

## 4. The AI/ML components (all of them)

The system contains **four** distinct AI/ML technologies. This section documents
each one fully.

### 4.1 No-show prediction — scikit-learn Logistic Regression

**Purpose:** estimate the probability a booked patient will attend, so the front
desk knows where to spend reminder effort.

**Why logistic regression (not linear):** attendance is a yes/no outcome. Linear
regression can predict impossible values (<0 or >1); logistic regression squashes
the linear combination through a sigmoid so the output is a genuine probability
in 0–1. It is also interpretable — each feature has a coefficient — so the agent
can explain *why* a patient is high-risk.

**Features (behavioural only — never demographic, by design):**

| Feature | Meaning |
|---|---|
| `prior_appts` | how many past appointments the patient has |
| `prior_noshow_rate` | their historical no-show rate (base rate 0.25 for new patients) |
| `lead_days` | days between booking and the appointment (longer → riskier) |
| `appt_hour` | hour of day (early morning dips) |
| `dow` | day of week (Monday dips) |
| `confirmed` | whether a reminder was acknowledged |
| `is_new` | new vs returning patient |
| `appt_type` | checkup, cleaning, filling, root_canal, extraction, emergency, followup |

Prior attendance dominates; emergencies/root canals attend well, routine
cleanings are easy to skip. Demographics are deliberately excluded — a risk score
that penalised patients by age/sex/etc. would be both unfair and a liability, and
behaviour predicts better anyway.

**Pipeline:** `ColumnTransformer` (one-hot the categorical `appt_type`, standard-
scale the numerics) → `LogisticRegression(class_weight="balanced")` to handle the
~18–25% no-show imbalance.

**Training discipline:**
- **Time-based split** (train on the past 80%, test on the future 20%) — the only
  honest estimate of tomorrow's performance.
- **Leak-free features:** synthetic history is generated in chronological order so
  `prior_noshow_rate` only ever counts appointments *before* the current one.
- **Honest metrics:** judged on **AUC**, precision/recall on the no-show class,
  and calibration (**Brier score**) — never plain accuracy, because "predict
  everyone shows" scores ~80% and is useless.

**Measured result:** **AUC ≈ 0.73**, no-show rate ≈ 25% — deliberately realistic.
Real no-show models sit around here; a model that scored 0.95 would be a sign the
synthetic data was made unrealistically easy and would collapse on real data. At
this AUC, a "high-risk" flag is right roughly half the time — good for
*prioritising attention*, which is why the only action on a prediction is a soft
reminder, never an auto-cancel.

**Training data:** `ml/generate_data.py` synthesises ~300 patients and ~3,700
appointments with a hidden "reliability" trait plus noise, labelled from a true
logistic function of the features. `ml/train.py` fits and pickles the pipeline to
`models/noshow.pkl`. **Today's schedule** is 12 real roster patients (ids 1–12)
with hand-set feature profiles chosen to span the green/yellow/red risk range.

### 4.2 Dental image screening — two-stage PyTorch vision pipeline

Two trained models run in sequence. (These are the clinic's own models, supplied
as `.pt` files; the app wraps them.)

**Stage 1 — Gate (EfficientNetV2-S, via `timm`):** answers "is this a dental /
oral image?" It's a 224×224 classifier with classes `dental` (index 0) and
`non_dental` (index 1). If P(dental) is below the tuned threshold **0.080** the
image is **rejected** and the detector never runs — this prevents nonsense
"findings" on non-dental photos. The low threshold favours letting real dental
images through (a wrongly rejected image is a case the detector never sees).

**Stage 2 — Detector (YOLO, end-to-end / NMS-free):** on dental images, detects
disease at deploy confidence **0.10** (a high-sensitivity screening operating
point). Eight classes:

`0 early_caries · 1 advanced_caries · 2 gingivitis · 3 hypodontia · 4 plaque ·
5 discoloration · 6 ulcer · 7 abnormal`

`abnormal` is an image-level "diseased but not one of the specific classes" flag
(a full-frame box). Healthy images have no boxes. `hypodontia` is displayed as
**"Edentulism"** in the UI while the model still outputs index 3.

**Verdict logic (applied by the agent):**

| Situation | Verdict |
|---|---|
| gate says non-dental | **Non-dental — rejected** (detector skipped) |
| dental, 0 detections | **Healthy — no findings** |
| only `abnormal` box(es) | **Abnormal — needs review** |
| any lesion box (0–6) | **Disease Detected** |

**Batched, full-speed inference:** for multiple images, the gate runs one batched
forward pass over all of them, then the detector runs one batched pass over only
the images that clear the gate (chunked, default 8 per call, to bound memory).
So N images cost ~2 batched calls, not 2N.

### 4.3 Text generation — tiny local LLM (Ollama)

**Model:** **`qwen3:1.7b`** by default — a ~1.4 GB model in the Qwen3 family.
Chosen for speed at a size that still writes a coherent 3–4 line clinical summary.
(`llama3.2:1b` is an even smaller/faster alternative; set `OLLAMA_MODEL` to
switch.)

**How it's used well:**
- **`think=False`** — Qwen3 otherwise writes a long hidden chain-of-thought before
  answering; disabling it is the single biggest latency win for short structured
  text.
- **`keep_alive=30m`** — the model stays resident in RAM between calls, so only the
  first request of a session pays the load cost.
- **Schema-constrained JSON** — the expected JSON shape is passed to Ollama's
  `format` parameter, so the model is forced to emit valid, correctly-shaped JSON
  at the token level (the usual failure mode of small models, eliminated).
- **Graceful fallback** — if Ollama isn't running, the agent returns a
  deterministic templated result tagged **"Demo (offline)"**, so the app never
  hard-fails. A real model response is tagged **"Local model"**.

**Where the LLM runs:** the Report agent's 3–4 line summary. (The Patient
Intelligence brief and the Clinical SOAP note also use this LLM layer; they remain
in the codebase but are off the main pipeline — see §5.)

### 4.4 Agent orchestration — LangGraph

The Scheduling and Imaging agents are expressed as **LangGraph** graphs: a typed
state dict flows through nodes (plain Python functions) connected by edges, with
conditional branching (e.g. the imaging gate routes non-dental images straight to
the verdict, skipping the detector). LangGraph gives a clean, extendable way to
sequence multi-step work and is where new steps or branches are added later.

---

## 5. The agents in detail

### 5.1 Smart Scheduling agent  (`agents/scheduling_agent.py`)

- **Job:** score today's schedule for no-show risk and prepare reminder decisions.
- **Trigger:** an APScheduler "morning job" (the deck's *Daily Schedule Job*),
  and on-demand whenever the Appointments screen opens.
- **Graph:** `load_schedule → featurize → predict → explain`.
  - *load_schedule* — pull today's appointments (attended IS NULL) from SQLite.
  - *featurize* — build the model's feature row per appointment (shared
    definition in `agents/noshow_features.py`, so training and serving never
    diverge).
  - *predict* — the sklearn model outputs `P(attend)` per appointment; band into
    Low/Med/High; sum the probabilities into "expect ~N of M today (±range)".
  - *explain* — a one-line reason from the model's own coefficients (no LLM, so
    the table is instant).
- **Reasoning core:** the logistic-regression model (§4.1). Not the LLM.
- **UI:** risk chips coloured by attend probability — **green ≥70% · yellow
  30–70% · red <30%**. The clinician sets each row to **Approve / Hold / Reject**
  (a selection, not an instant action) and hits **Submit** to finalize the batch
  in one fast call. Decisions land in the **Finalized** tab, grouped by date.
- **Outcome capture:** in Finalized, each decision can be marked **Showed /
  No-show**, recording the real result to the appointment — the signal a future
  model retrains on.

### 5.2 Imaging & Diagnostic agent  (`agents/imaging_agent.py`, `vision/pipeline.py`)

- **Job:** screen a patient's dental images and flag lesions for review.
- **Graph:** `gate → (dental?) → detect → verdict → explain`; non-dental routes
  straight to verdict.
- **Reasoning core:** the two-stage vision pipeline (§4.2). Not the LLM.
- **Output:** the shared contract — Insights (detected conditions with confidence,
  or "not a dental image"), Recommendations (per-condition next steps), Actions
  ("Add findings to record", "Flag for specialist review").
- **Persistence:** approving "Add findings" writes the finding to the patient's
  record (`patient_notes`, kind `imaging`) — which then flows into the report.
- **Speed:** batched over selected images; the models load once then stay warm.

### 5.3 Report agent  (`agents/report_agent.py`)

- **Job:** produce a concise report at the end of a visit.
- **Inputs:** the patient's record + history, the latest approved imaging finding
  (pulled automatically), and the clinician's typed findings.
- **Reasoning core:** the tiny LLM (§4.3) writes a **3–4 line AI summary**,
  schema-constrained, `think=False`, with a templated fallback.
- **Flow:** *generate* creates a draft report (imaging + clinical + AI summary);
  *finalize* marks it finalized and files it under Reports.

### 5.4 Patient Intelligence & Clinical SOAP  (`agents/patient_intelligence.py`, `agents/clinical_assistant.py`)

Two additional agents that share the LLM layer:
- **Patient Intelligence** — drafts a clinical *brief* (summary, risk flags,
  recommendations) from a patient's record and visits.
- **Clinical Assistant** — drafts a full **SOAP note** (Subjective / Objective /
  Assessment / Plan), pulling the latest imaging finding into the Objective.

Both are complete and tested and follow the same contract, but the current
pipeline uses the shorter Report summary instead (your "3–4 lines, tiny model,
fast" requirement). They remain in the codebase if the longer forms are wanted
again.

---

## 6. The end-to-end clinical workflow

```
  LOGIN
    │
    ▼
  DASHBOARD  (today's counts, orientation)
    │
    ▼
  APPOINTMENTS  ── Smart Scheduling agent
    • morning job scores every booking for no-show risk
    • green / yellow / red chips + "expect ~N of M today"
    • set Approve / Hold / Reject per row → Submit
    • Finalized tab: decisions by date + Showed/No-show outcome
    │  (approved patients form the queue for the rest of the visit)
    ▼
  SCAN  ── Imaging & Diagnostic agent
    • shows the APPROVED-TODAY queue → pick a patient (becomes "active patient")
    • upload one or more images → batched gate + detector
    • verdict + detected conditions → Approve findings (attach to the patient)
    │  → "Continue to report"
    ▼
  CREATE REPORT  ── Report agent
    • imaging findings pulled in automatically (read-only)
    • clinician types clinical findings
    • Generate → 3–4 line AI summary (tiny local model)
    • Finalize → filed
    ▼
  REPORTS
    • list of finalized reports → View: AI summary + imaging + clinical findings
```

The **active patient** is the spine: chosen once (by clicking a name in
Appointments or opening a chart in Patients / the Scan queue) and carried through
Scan → Report without re-selecting. A banner under the top bar always shows who is
active.

**The connected loop:** an imaging finding approved in Scan is written to the
patient's record and then appears, pre-filled, in Create Report — so what the
detector found flows into the written report without re-typing. Finalized reports
and captured outcomes persist, so the record reflects what happened.

---

## 7. Screen-by-screen reference

- **Login** — a cosmetic gate that reveals the app (no real authentication in the
  prototype).
- **Dashboard** — quick counts (roster size, agents live, today's appointment
  count pulled live, model mode) and an orientation note.
- **Appointments** (Front desk) — two tabs:
  - *Today* — the scored schedule, decision buttons, Submit; patient names are
    clickable to open their chart.
  - *Finalized* — decisions grouped by date with Showed/No-show outcome capture.
- **Scan** (Clinical) — the approved-today queue; pick a patient, upload/analyze
  images, approve findings.
- **Create Report** (Clinical) — imaging (auto) + clinical findings → AI summary →
  finalize.
- **Reports** (Records) — finalized list → View detail.
- **Patients** (Admin) — the real roster from the database; open a chart to start
  a visit.
- **Users / Settings** (Admin) — static forms in this prototype.

---

## 8. The data model (SQLite, `db.py`)

| Table | What it holds |
|---|---|
| `patients` | the unified roster (id, mrn, name, age, sex, phone, allergies, conditions) — 12 seeded patients |
| `visits` | past clinical visits per patient (history the agents read) |
| `appointments` | schedule rows; today = `attended NULL`, history = 0/1; carries the no-show features |
| `decisions` | finalized scheduling decisions (approve/hold/reject) per appointment, by date |
| `patient_notes` | **persisted record** — approved imaging findings (`kind=imaging`) and saved notes (`soap`/`summary`) |
| `reports` | draft/finalized reports (imaging, clinical, AI summary, status, timestamps) |
| `agent_actions` | proposed actions awaiting approval (with payload) |
| `agent_log` | executed actions, timestamped (the activity trail) |

**Unified identity:** today's appointment `patient_id` equals `patients.id`, so
"open the chart" from a schedule row lands on a real patient with history.

---

## 9. The API reference (`app.py`)

**Patients**
- `GET /api/patients` — roster list
- `GET /api/patients/{id}` — patient + visit history
- `GET /api/patients/{id}/context` — latest imaging finding + saved notes

**Patient Intelligence**
- `POST /api/agents/patient_intelligence/run` — `{subject_id}` → brief

**Scheduling**
- `GET /api/scheduling/today` — today's scored schedule + predictions
- `POST /api/scheduling/decisions` — batch `{decisions:[…]}` finalize
- `POST /api/scheduling/decision` — single decision (legacy)
- `GET /api/scheduling/finalized` — finalized decisions (+ attendance outcome)
- `GET /api/scheduling/approved-today` — the approved queue for Scan
- `POST /api/appointments/{appt_id}/outcome` — `{attended:0|1}` outcome capture

**Imaging**
- `POST /api/imaging/analyze_batch` — multipart `files[]` + `patient_id` → per-image results
- `POST /api/imaging/analyze` — single image

**Clinical / Reports**
- `POST /api/clinical/draft` — `{patient_id, complaint, findings}` → SOAP note
- `POST /api/reports/generate` — `{patient_id, clinical}` → draft report + AI summary
- `POST /api/reports/{id}/finalize` — finalize
- `GET /api/reports?status=finalized` — list
- `GET /api/reports/{id}` — view

**Shared**
- `POST /api/actions/{id}/approve` — execute a proposed action (persists to record)
- `GET /api/agent-log` — recent activity

---

## 10. Project file structure

```
D:\Agents\
├── app.py                       FastAPI — serves UI + all agent APIs + morning job
├── db.py                        SQLite schema, seed roster, all data helpers
├── contracts.py                 shared agent contract (Insights→Recs→Actions)
├── llm.py                       Ollama client (qwen3:1.7b, think=False, JSON, fallback)
├── requirements.txt
├── agents\
│   ├── __init__.py              agent registry + action handlers (persist on approve)
│   ├── base.py                  BaseAgent lifecycle
│   ├── patient_intelligence.py  brief agent (LLM)
│   ├── scheduling_agent.py      no-show agent (sklearn + LangGraph)
│   ├── noshow_features.py       one feature definition — shared train ↔ serve
│   ├── imaging_agent.py         two-stage vision agent (LangGraph)
│   ├── clinical_assistant.py    SOAP note agent (LLM)
│   └── report_agent.py          3–4 line report summary (tiny LLM)
├── vision\
│   ├── pipeline.py              loads/runs the .pt gate + detector (batched)
│   ├── gate_best.pt             EfficientNetV2 gate  (you supply)
│   ├── gate_meta.json           threshold/classes    (you supply)
│   └── detector_best.pt         YOLO detector        (you supply)
├── ml\
│   ├── generate_data.py         synthetic history + today's roster schedule
│   └── train.py                 trains → models\noshow.pkl
├── models\
│   └── noshow.pkl               trained logistic-regression pipeline
└── static\
    └── index.html               the whole DentAgent UI (no voice)
```

---

## 11. How to run it (Windows, conda)

```cmd
conda activate agents
cd /d D:\Agents

:: one-time
pip install -r requirements.txt
pip install timm ultralytics            :: for the imaging agent (torch pulled in)
ollama pull qwen3:1.7b                   :: tiny text model

:: place the vision models in vision\  (gate_best.pt, gate_meta.json,
:: detector_best.pt) OR set GATE_WEIGHTS / GATE_META / DETECTOR_WEIGHTS

:: build data + model (re-run after any schema/data change)
del dpms.db
python ml\generate_data.py
python ml\train.py

:: run
uvicorn app:app
```

Open **http://127.0.0.1:8000**.

**Tags to know:** a green **"Local model"** tag means the LLM answered; **"Demo
(offline)"** means Ollama wasn't reachable and the templated fallback ran. The
imaging agent returns a clear message if the `.pt` models or vision libraries are
missing. Re-run `python ml\train.py` on your own machine so `noshow.pkl` matches
your installed scikit-learn version.

---

## 12. Speed / latency design

- **Deterministic parts are instant:** no-show scoring (sklearn) and the risk
  chips have no model latency.
- **The LLM is tiny and warm:** `qwen3:1.7b`, `think=False`, `keep_alive=30m`.
  Only the first summary of a session pays the model-load cost; the rest are a few
  seconds. Set `OLLAMA_MODEL=llama3.2:1b` for even less.
- **Vision is batched:** one gate pass + one detector pass over a chunk of images,
  not per image; models load once.
- **The table never waits on the LLM:** schedule risk and image verdicts come from
  the fast models; the LLM only runs when you generate a report.

---

## 13. Safety & clinical framing

- **Screening, not diagnosis.** The vision detector runs at a high-sensitivity
  threshold (deliberately over-flagging); the no-show model is ~0.73 AUC. Both are
  tuned to *direct attention*, and every output is confirmed by a clinician.
- **Human approval on every write.** Agents propose; a person approves; only then
  is anything saved. Actions on a wrong prediction are soft (a reminder), never
  punitive (no auto-cancel, no auto-diagnosis).
- **Fair by design.** No-show features are behavioural, never demographic.
- **Privacy.** All inference is local; patient data stays on the machine.

---

## 14. Honest limitations & next steps

- **Prototype auth.** Login is cosmetic; there is no real authentication or
  per-user access control.
- **Static admin pages.** Patients list is real (from the DB); New Patient, Users,
  and Settings are static forms not yet wired to the database.
- **No retraining loop yet.** Outcomes (showed / no-show) and notes persist, but
  nothing yet *retrains* the no-show model on accumulated real outcomes — that's a
  genuine ML pipeline (collect labels → periodically refit → swap the pickle) and
  the natural next phase.
- **INT8 vision accuracy unverified.** If quantised models are deployed to edge
  hardware, their accuracy vs FP32 should be checked (plaque/abnormal are the
  weak, mutually-confused classes).
- **Reminders don't actually send.** Approving a reminder logs it; wiring it to an
  SMS/WhatsApp provider is a follow-up.
- **Single-clinic scale.** SQLite suits one clinic; a multi-terminal deployment
  moves to Postgres (schema carries over).

---

*Local-first · agentic · human-in-the-loop. Four AI models — logistic regression,
a two-stage CNN vision pipeline, and a tiny LLM — behind one clinic UI, each one
proposing while a clinician decides.*
