# DentAgent: where it stands, and where to take it

*Assessment as of November 2026 (commit `d6db51d`, which added the agent registry). Market notes come from general knowledge up to mid-2026; check them before relying on them.*

**In short:** the engineering patterns are ahead of where a prototype usually is. Specifically: human approval before any action, durable workflows, and checks on what the model writes. The product is behind the market. There is no voice input, no connections to other clinic systems, no evaluation data, and no clinical validation, and the models are small and slow. Being fully local is the strongest strategic card, and it isn't being used as a selling point yet.

## What's already strong (keep and promote)

| Capability | Why it matters to the market |
|---|---|
| **Fully local (Ollama, SQLite)**, so patient data never leaves the clinic | Most dental AI products run in the cloud. Privacy laws (India's DPDP Act, HIPAA in the US) and clinics with poor internet make on-premises software a real point of difference. |
| **The AI only proposes; a human approves**, with the approver's name logged | This is now the standard expectation for clinical AI. Regulators and buyers ask about it first. |
| **Durable visit pipeline** (LangGraph checkpoints, pause/resume, triggered by events) | Agents built into the workflow, not chatbots, are where agentic software is heading. Few dental tools chain scan → note → report like this. |
| **Safety and grounding checks** on every draft, in all four drafting agents | Invented findings are the main barrier to AI-written clinical notes. A check you can point to is a selling point. |
| **Tool-calling Copilot** with guarded writes and checked patient IDs | This is the current agent design: the model picks its tools, and code enforces the limits. |
| **Agent registry** (`GET /api/agents`, `POST /api/agents/{key}/run`) | One list of all 8 agents, with parameter checks and admin-only controls. It's the base for exposing them to outside tools (see MCP below). |
| **Explainable no-show model** (logistic regression, no demographic data) | This is the right call. Not every problem needs an LLM. |

## Gaps versus the market

### Competitors

Dental AI spending has gathered in four areas:

- **X-ray detection:** Overjet, Pearl and VideaHealth have FDA clearances for it. Competing head-on with them is a losing fight unless you have regulatory clearance.
- **AI scribes:** voice-to-clinical-note tools are the fastest-adopted category across healthcare.
- **Billing and insurance claims automation:** clinics feel the return on investment directly here.
- **Front-desk voice agents:** answering calls, booking, and sending recall and reminder messages.

DentAgent touches all four lightly but leads in none.

### Technical gaps

- **Model quality and speed:** qwen3:1.7b on a CPU takes 30–100 seconds per answer and makes mistakes, as testing showed. A clinic won't wait that long.
- **No evaluations:** there's no fixed test set and no measured grounding-failure or safety-violation rate. Without numbers, you can't show any clinic or regulator that the system is safe.
- **Grounding matches words:** it can't catch claims about *when* something happened. "Abscess" written as current passes if the word appears anywhere in the record.
- **No connections to other systems:** no FHIR or HL7 for health records, no DICOM for X-rays, no link to existing practice-management software or insurers. Clinics won't re-enter their data.
- **Tests aren't in the repo,** and there's no CI to run them on each change.
- **Production readiness:** SQLite, a single admin role and no multi-clinic setup. There's no monitoring of agent runs and no encryption of stored data.
- **Prompt injection:** text from the public booking page and patient notes goes into the model's prompts unfiltered.

## Methods worth adopting, in priority order

### Do now (biggest gain for the effort)

1. **Evaluation set:** 50–100 real-style cases with expected outputs. Measure the grounding-failure rate, safety violations and whether the Copilot picks the right tool. Run it in CI on every change. Everything else depends on this. *Started: [evals/](../evals/README.md), with 72 cases. First baseline: about 31% of drafts invent a finding, about 19% have a real safety violation, and the Copilot passes 11/12.*
2. **Citations:** make the model attach a source ID to each sentence (for example `imaging#3` or `visit 2025-07-01`), then check the claim against that source with a small local entailment model. This fixes the "when did it happen" gap and gives reviewers one-click verification.
3. **A faster, better model:** a GPU, or a quantised 7–8B instruct model through llama.cpp or vLLM. Send simple tasks to the 1.7B model and harder ones to the bigger one.

### Next (product differentiation)

4. **Voice-to-SOAP:** run Whisper locally to turn a recording into a draft SOAP note, with the existing grounding check. This joins the hottest category while keeping data on-site, which is a clear pitch.
5. **Memory per patient:** search over each patient's past notes (for example with `nomic-embed-text`), so the agents see full history instead of the last few visits.
6. **MCP server:** expose the agent registry and the Copilot's tools through MCP, the standard way AI tools plug into each other. Other agents and assistants could then use DentAgent without custom integration.
7. **Tracing:** use OpenTelemetry or Langfuse to record each agent step, tool call and approval. This also strengthens the audit trail for compliance.

### Later (to sell it)

8. **Interoperability:** FHIR or HL7 export, DICOM import, and a connector to one existing practice-management system.
9. **Fine-tuning:** a LoRA adapter trained on de-identified dental notes, which helps the small models most.
10. **Regulatory path for the vision model:** a clinical validation study, then CDSCO (India) or FDA 510(k) clearance, before claiming it detects anything.
11. **Hardening:** Postgres, role-based access, encryption of stored data, separate data per clinic, and filtering of untrusted text before it reaches the model.

## Status: November 2026

The assessment above is kept as written. Here's what has been done about each item since, with measured
numbers from [evals/](../evals/README.md):

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Evaluation set | ✅ Done | 92 cases (62 offline checker cases, 16 drafting, 14 Copilot). The offline suite and unit tests run in CI on every push. |
| 2 | Citations | ✅ Done, deterministic | Every draft sentence carries a source tag. Each claim is checked against the source it cites, and the Objective may only cite today's sources. A timing check flags past findings stated as current. The NLI model wasn't installed: word-level checks against the cited source cover the gap that was listed. |
| 3 | Faster, better model | ✅ Routing done | The GPU already runs qwen3:1.7b at ~3 s per draft (not 30–100 s). Tiers `fast` / `strong` / `lora`; measured qwen2.5:3b worse on grounding, so routing stays on 1.7B. |
| 4 | Voice-to-SOAP | ✅ Done | Local faster-whisper (CPU int8; GPU when CUDA 12 libraries are present). Dictation → complaint and findings → grounded SOAP. |
| 5 | Memory per patient | ✅ Done | `nomic-embed-text` semantic search over visits, notes and reports. Used by Patient Intelligence, SOAP context and a Copilot tool. |
| 6 | MCP server | ✅ Done | `mcp_server.py` (stdio): the registry, the Copilot's read tools, and proposal-only write tools. |
| 7 | Tracing | ✅ Done | OpenTelemetry spans stored locally, with optional OTLP export; `/api/traces`, `/api/health`, Settings page. |
| 8 | Interoperability | ✅ Done (generic) | FHIR R4 export/import (validated), HL7 v2 ADT^A04/A08, DICOM import, CSV patients/appointments. No connector for a named commercial PMS, because there's no vendor API access. |
| 9 | Fine-tuning | ✅ Done (synthetic data) | LoRA r=16 on Qwen3-1.7B, 2,000 synthetic grounded examples, 30 min on the laptop GPU. Drafting evals against the base model: invented content 0% vs 19%, safety 0% vs 6%, citation failures 0% vs 75%, relevance misses 0/8 vs 1/8; also correct on held-out allergy/condition combinations. Now the default for SOAP notes and reports when the local build is present (otherwise the base model is used). It was trained on synthetic data, not real notes, and its plans are still partly templated. |
| 10 | Regulatory path | 📄 Planned | [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md): study design, sample size, reader study, CDSCO/510(k) route. A study and clearance can't be produced in code. |
| 11 | Hardening | ✅ Done, except Postgres | Roles (admin / clinician / front desk), field-level encryption of clinical text, clinic separation, untrusted-text filtering, `.env` with a persistent JWT secret. The Postgres port was deferred by decision; SQLite is kept. |

Technical gaps above, now:
- **Model speed:** solved on this hardware (GPU).
- **No evaluations:** solved.
- **Grounding only matched words:** citations and the timing check added.
- **No connections to other systems:** added.
- **No tests or CI:** added.
- **Monitoring and encryption:** added.
- **Prompt injection:** filtered and fenced, with a Copilot injection eval case.

Still open:
- Postgres.
- A real PMS vendor connector.
- Clinical validation of the vision model.
- Training on real de-identified notes (the adapter learned from synthetic cases; its plans are still partly templated).

## Suggested strategy

Don't sell it as "AI that reads X-rays." Sell it as **a private AI assistant for the whole visit, where nothing leaves the clinic**: voice → grounded SOAP → report → recall, with every AI claim cited and approved by a clinician. Start with privacy-sensitive or low-connectivity clinics, such as Indian private practices and dental chains. That's where cloud competitors are weakest and the local-first design is strongest.

The first step that unlocks the rest is the evaluation set (item 1). Without measured accuracy, nothing else can be shown to a clinic or regulator.
