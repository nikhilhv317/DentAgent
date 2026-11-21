"""
FastAPI app: serves the frontend and both agents' APIs.

Run:
    pip install -r requirements.txt
    ollama pull qwen3:1.7b
    uvicorn app:app --reload
Then open http://127.0.0.1:8000

Agents:
    Patient Intelligence  -> POST /api/agents/patient_intelligence/run   (on demand)
    Smart Scheduling      -> GET  /api/scheduling/today                  (morning job + on demand)
                             POST /api/scheduling/approve
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Body, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime
import os


def _load_env():
    """Read .env (if python-dotenv is installed) BEFORE db/auth read their settings,
    and make sure a strong JWT_SECRET exists — without one, auth.py falls back to a
    random per-process secret and every restart logs everyone out."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass
    secret = os.getenv("JWT_SECRET", "")
    if len(secret) < 32 or secret.startswith("change-me"):
        import secrets
        secret = secrets.token_urlsafe(48)
        with open(env_path, "a", encoding="utf-8") as f:
            f.write(f"\n# generated on first run — keeps logins valid across restarts\nJWT_SECRET={secret}\n")
        os.environ["JWT_SECRET"] = secret


_load_env()

import db  # noqa: E402  (after _load_env so DPMS_DB_PATH etc. from .env apply)
import agents
from auth import authenticate, create_token, require_auth, require_admin, require_clinician

app = FastAPI(title="DPMS Agentic AI")
db.init_db()
db.seed_staff()   # admin from ADMIN_PASSWORD, else random (printed); rotates old admin123

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ---- clinic separation --------------------------------------------------------
# Every chart belongs to one clinic (patients.clinic_id); every staff login to one
# clinic (staff.clinic_id, checked fresh on each request in auth.require_auth).
# Staff only see and touch their own clinic's patients — anything else is a 404,
# so ids from another clinic can't even be probed.

def _clinic(user) -> int:
    return int(user.get("clinic") or 1)


def _own_patient(patient_id, user):
    p = db.get_patient(int(patient_id)) if patient_id is not None else None
    if p is None or int(p.get("clinic_id") or 1) != _clinic(user):
        raise HTTPException(404, "patient not found")
    return p


def _clinic_patient_ids(user) -> set:
    return {p["id"] for p in db.list_patients(_clinic(user))}



@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/book")
def booking_page():
    return FileResponse(os.path.join(STATIC_DIR, "book.html"))


# ---- public booking endpoint (no auth — patients use this) -----------------
# One call, and it never returns a patient_id or MRN: an anonymous caller must
# not be able to look patients up by phone, or book against someone else's
# chart by guessing sequential ids.

PUBLIC_APPT_TYPES = {"checkup", "cleaning", "filling", "root_canal",
                     "extraction", "emergency", "followup"}


class PublicBookReq(BaseModel):
    name: str
    phone: str
    appt_date: str
    appt_time: str
    appt_type: str = "checkup"


@app.post("/api/public/book")
def api_public_book(req: PublicBookReq):
    """Find-or-create the patient and book. An existing chart is reused only
    when BOTH phone and name match — families often share one phone, and a
    phone-only match would put a different person's visit on that chart.
    Anything less creates a new record for staff to merge via the duplicate
    check."""
    name = req.name.strip()
    if not name or len(db.normalize_phone(req.phone)) < 10:
        raise HTTPException(400, "Please enter your name and a 10-digit phone number.")
    if req.appt_type not in PUBLIC_APPT_TYPES:
        raise HTTPException(400, "Unknown appointment type.")
    try:
        if datetime.strptime(req.appt_date, "%Y-%m-%d").date() < date.today():
            raise HTTPException(400, "Please choose a date from today onwards.")
    except ValueError:
        raise HTTPException(400, "Invalid date.")
    candidates = db.find_duplicate_candidates(name, req.phone)
    exact = next((c for c in candidates if c["match_reason"] == "phone"
                  and c["name"].strip().lower() == name.lower()), None)
    pid = exact["id"] if exact else db.create_patient(name, None, "", req.phone, "", "")[0]
    _, was_dup = db.create_appointment(pid, req.appt_date, req.appt_time, req.appt_type)
    return {"ok": True, "duplicate": was_dup}


# ---- auth ------------------------------------------------------------------

class LoginReq(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def api_login(req: LoginReq):
    user = authenticate(req.username, req.password)
    if not user:
        raise HTTPException(401, "Invalid username or password")
    token = create_token(user["username"], user["display_name"], user.get("role") or "clinician",
                         user.get("clinic_id") or 1)
    return {"token": token, "display_name": user["display_name"],
            "username": user["username"], "role": user.get("role") or "clinician"}


@app.get("/api/auth/me")
def api_me(user=Depends(require_auth)):
    return user


# ---- staff management (admin) ----------------------------------------------

class StaffReq(BaseModel):
    username: str
    display_name: str
    password: str
    role: str = "clinician"     # admin | clinician | front_desk


@app.get("/api/staff")
def api_list_staff(user=Depends(require_auth)):
    return db.list_staff()


@app.post("/api/staff")
def api_create_staff(req: StaffReq, user=Depends(require_admin)):
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    try:
        db.create_staff(req.username, req.display_name, req.password, req.role, user.get("clinic", 1))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


class PasswordReq(BaseModel):
    new_password: str


@app.put("/api/staff/{username}/password")
def api_change_password(username: str, req: PasswordReq,
                         user=Depends(require_auth)):
    # staff may change their own password; only admin may change someone else's
    if username != user["sub"] and user.get("role") != "admin":
        raise HTTPException(403, "You can only change your own password")
    if len(req.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    db.change_password(username, req.new_password)
    return {"ok": True}


class RoleReq(BaseModel):
    role: str


@app.put("/api/staff/{staff_id}/role")
def api_set_role(staff_id: int, req: RoleReq, user=Depends(require_admin)):
    try:
        db.set_staff_role(staff_id, req.role)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.delete("/api/staff/{staff_id}")
def api_deactivate_staff(staff_id: int, user=Depends(require_admin)):
    db.deactivate_staff(staff_id)
    return {"ok": True}


# ---- Patient Intelligence ---------------------------------------------------

@app.get("/api/patients")
def api_patients(user=Depends(require_auth)):
    return db.list_patients(_clinic(user))


# Must be declared before /api/patients/{patient_id}, or FastAPI matches
# "find" as a patient_id and returns 422.
@app.get("/api/patients/find")
def api_find_patient(name: str = "", phone: str = "", user=Depends(require_auth)):
    """Look up existing patients who might be the same person (staff-side
    duplicate check before 'New Patient' creates a second record for someone
    who already has a chart)."""
    ids = _clinic_patient_ids(user)
    return [c for c in db.find_duplicate_candidates(name, phone) if c["id"] in ids]


@app.get("/api/patients/{patient_id}")
def api_patient(patient_id: int, user=Depends(require_auth)):
    return _own_patient(patient_id, user)


@app.get("/api/patients/{patient_id}/context")
def api_patient_context(patient_id: int, user=Depends(require_auth)):
    """What staff should see when they open a patient's record: the latest
    imaging finding, saved notes, and past visit history — the Scan -> Note
    loop plus the clinical history, made visible."""
    patient = _own_patient(patient_id, user)
    findings = db.get_imaging_findings(patient_id)
    combined = db.imaging_summary(patient_id) if findings else None
    return {"latest_imaging": combined,
            "imaging_count": len(findings),
            "notes": db.get_notes(patient_id),
            "visits": patient.get("visits", [])}


class PatientReq(BaseModel):
    name: str
    age: int = None
    sex: str = ""
    phone: str = ""
    allergies: str = ""
    conditions: str = ""


@app.post("/api/patients")
def api_create_patient(req: PatientReq, user=Depends(require_auth)):
    pid, mrn = db.create_patient(req.name, req.age, req.sex, req.phone, req.allergies, req.conditions,
                                 _clinic(user))
    return {"id": pid, "mrn": mrn}


@app.put("/api/patients/{patient_id}")
def api_update_patient(patient_id: int, req: PatientReq, user=Depends(require_auth)):
    _own_patient(patient_id, user)
    db.update_patient(patient_id, req.model_dump())
    return {"ok": True}


@app.delete("/api/patients/{patient_id}")
def api_delete_patient(patient_id: int, user=Depends(require_clinician)):
    _own_patient(patient_id, user)
    db.deactivate_patient(patient_id)
    return {"ok": True}


class ApptReq(BaseModel):
    patient_id: int
    appt_date: str    # YYYY-MM-DD
    appt_time: str    # HH:MM
    appt_type: str = "checkup"


@app.post("/api/appointments")
def api_create_appointment(req: ApptReq, user=Depends(require_auth)):
    _own_patient(req.patient_id, user)
    aid, was_dup = db.create_appointment(req.patient_id, req.appt_date, req.appt_time, req.appt_type)
    return {"id": aid, "duplicate": was_dup}


class OutcomeReq(BaseModel):
    attended: int   # 1 = showed, 0 = no-show


@app.post("/api/appointments/{appt_id}/outcome")
def api_outcome(appt_id: int, req: OutcomeReq, user=Depends(require_auth)):
    """Outcome capture: record whether a patient actually showed. This is the
    real-world signal a future model retrains on."""
    conn = db.connect()
    row = conn.execute("SELECT patient_id FROM appointments WHERE id=?", (appt_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(404, "appointment not found")
    _own_patient(row["patient_id"], user)
    db.set_attended(appt_id, 1 if req.attended else 0)
    db.add_log("scheduling", f"appt:{appt_id}",
               f"Outcome recorded: {'showed' if req.attended else 'no-show'}")
    return {"ok": True}


@app.get("/api/agents")
def api_list_agents(user=Depends(require_auth)):
    return agents.list_agents()


@app.post("/api/agents/{agent_key}/run")
def api_run_agent(agent_key: str, params: dict = Body(default={}), user=Depends(require_auth)):
    """Run any registered agent with its keyword params as the JSON body, e.g.
    {"subject_id": 1} for patient_intelligence or {"patient_id": 1, "clinical": "..."}
    for report. File-input agents (imaging) use their multipart endpoint."""
    spec = agents.get_agent(agent_key)
    if spec is None:
        raise HTTPException(404, f"unknown agent '{agent_key}'")
    if spec.admin_only and user.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    if spec.clinical and user.get("role") not in ("admin", "clinician"):
        raise HTTPException(403, "Needs role: clinician or admin")
    if spec.input == "files":
        raise HTTPException(400, f"'{agent_key}' takes file uploads — use /api/imaging/analyze_batch")
    unknown = set(params) - set(spec.params)
    if unknown:
        raise HTTPException(400, f"unknown params {sorted(unknown)}; '{agent_key}' accepts {list(spec.params)}")
    for k in ("patient_id", "subject_id"):
        if params.get(k) is not None:
            _own_patient(params[k], user)
    try:
        return agents.run(agent_key, **params)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/actions/{action_id}/approve")
def api_approve(action_id: int, user=Depends(require_clinician)):
    try:
        message = agents.execute_action(action_id, approved_by=user.get("sub"))
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"ok": True, "message": message}


# ---- Clinic Copilot (tool-calling agent) ------------------------------------

class ChatTurn(BaseModel):
    role: str          # user | assistant
    content: str


class CopilotReq(BaseModel):
    message: str
    history: list[ChatTurn] = []


@app.post("/api/copilot/chat")
def api_copilot_chat(req: CopilotReq, user=Depends(require_auth)):
    """One chat turn. The model picks tools itself; any write comes back as a
    proposed action to approve via /api/actions/{id}/approve."""
    if not req.message.strip():
        raise HTTPException(400, "message is empty")
    return agents.run("copilot", message=req.message,
                      history=[t.model_dump() for t in req.history])


# ---- Visit Pipeline (event-driven, pauses for human review) -----------------

class PipelineStartReq(BaseModel):
    patient_id: int
    complaint: str = ""


class SoapEdit(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: list[str]


class PipelineDecision(BaseModel):
    stage: str                     # soap | report — must match what the run awaits
    action: str                    # approve | redraft | reject
    soap: Optional[SoapEdit] = None      # edited note (stage=soap, approve)
    summary: Optional[str] = None        # edited summary (stage=report, approve)
    complaint: Optional[str] = None      # updated chief complaint (stage=soap, redraft)


@app.get("/api/pipeline/runs")
def api_pipeline_runs(user=Depends(require_auth)):
    from agents import visit_pipeline
    ids = _clinic_patient_ids(user)
    return [r for r in visit_pipeline.list_runs() if r["patient_id"] in ids]


@app.post("/api/pipeline/start")
def api_pipeline_start(req: PipelineStartReq, user=Depends(require_clinician)):
    _own_patient(req.patient_id, user)
    try:
        return agents.run("visit_pipeline", patient_id=req.patient_id, complaint=req.complaint,
                          trigger=f"started by {user['sub']}")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/pipeline/{run_id}/resume")
def api_pipeline_resume(run_id: int, req: PipelineDecision, user=Depends(require_clinician)):
    from agents import visit_pipeline
    run = db.get_pipeline_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    _own_patient(run["patient_id"], user)
    decision = req.model_dump(exclude_none=True)
    decision["by"] = user["sub"]          # who signed off, recorded in the agent log
    try:
        return visit_pipeline.resume(run_id, decision)
    except ValueError as e:
        raise HTTPException(409, str(e))


# ---- Smart Scheduling (LangGraph agent) -------------------------------------
# Imported lazily so the rest of the app still boots if the scheduling deps
# (langgraph / sklearn) or the trained model aren't present yet.

def _scheduling():
    from agents import scheduling_agent
    return scheduling_agent


@app.get("/api/scheduling/today")
def api_scheduling_today(user=Depends(require_auth)):
    """Score today's schedule (fast — sklearn only). No parked run to go stale."""
    res = agents.run("scheduling", day=date.today().isoformat())
    ids = _clinic_patient_ids(user)
    res["predictions"] = [p for p in res["predictions"] if p["patient_id"] in ids]
    return res


class DecisionReq(BaseModel):
    appt_id: int
    patient: str
    appt_type: str
    appt_hour: int
    p_attend: float
    decision: str            # approve | hold | reject


@app.post("/api/scheduling/decision")
def api_scheduling_decision(req: DecisionReq, user=Depends(require_auth)):
    if req.decision not in ("approve", "hold", "reject"):
        raise HTTPException(400, "decision must be approve, hold or reject")
    reminder = None
    if req.decision == "approve":
        reminder = _scheduling().generate_reminder(req.patient, req.appt_hour)
    db.save_decision(req.appt_id, req.patient, req.appt_type, req.appt_hour,
                     req.p_attend, req.decision, reminder)
    verb = {"approve": "Reminder approved", "hold": "Held", "reject": "Rejected"}[req.decision]
    db.add_log("scheduling", f"appt:{req.appt_id}", f"{verb} — {req.patient} ({req.appt_hour}:00)")
    return {"ok": True, "decision": req.decision, "reminder": reminder}


class DecisionItem(BaseModel):
    appt_id: int
    patient: str
    appt_type: str
    appt_hour: int
    p_attend: float
    decision: str


class DecisionsReq(BaseModel):
    decisions: list[DecisionItem]


@app.post("/api/scheduling/decisions")
def api_scheduling_decisions(req: DecisionsReq, user=Depends(require_auth)):
    """Finalize a batch of decisions in one fast call — no LLM, no delay.
    Approved rows get an instant templated reminder; wording can be upgraded later."""
    sched = _scheduling()
    count = 0
    for it in req.decisions:
        if it.decision not in ("approve", "hold", "reject"):
            continue
        reminder = sched.templated_reminder(it.patient, it.appt_hour) if it.decision == "approve" else None
        db.save_decision(it.appt_id, it.patient, it.appt_type, it.appt_hour,
                         it.p_attend, it.decision, reminder)
        verb = {"approve": "Reminder approved", "hold": "Held", "reject": "Rejected"}[it.decision]
        db.add_log("scheduling", f"appt:{it.appt_id}", f"{verb} — {it.patient} ({it.appt_hour}:00)")
        count += 1
    return {"ok": True, "count": count}


@app.get("/api/scheduling/finalized")
def api_scheduling_finalized(user=Depends(require_auth)):
    ids = _clinic_patient_ids(user)
    return [d for d in db.list_finalized() if d.get("patient_id") in ids]


@app.get("/api/agent-log")
def api_log(user=Depends(require_auth)):
    return db.get_log()


# ---- Report pipeline: approved queue -> generate -> finalize -> view --------

@app.get("/api/scheduling/approved-today")
def api_approved_today(user=Depends(require_auth)):
    ids = _clinic_patient_ids(user)
    return [q for q in db.approved_today() if q["patient_id"] in ids]


class ReportReq(BaseModel):
    patient_id: int
    clinical: str = ""


class ScanImageReq(BaseModel):
    patient_id: int
    filename: str = ""
    data_uri: str
    verdict: str = ""
    detections: str = None    # JSON list of {cls,name,display,conf,box}


@app.post("/api/scan-images")
def api_save_scan_image(req: ScanImageReq, user=Depends(require_auth)):
    """Store a scanned image (data URI) against a patient so reports can embed it."""
    if req.patient_id is not None:
        _own_patient(req.patient_id, user)
    db.save_scan_image(req.patient_id, req.filename, req.data_uri, req.verdict, req.detections)
    return {"ok": True}


@app.get("/api/patients/{patient_id}/scan-images")
def api_get_scan_images(patient_id: int, user=Depends(require_auth)):
    _own_patient(patient_id, user)
    return db.get_scan_images(patient_id)


@app.post("/api/reports/generate")
def api_report_generate(req: ReportReq, user=Depends(require_clinician)):
    _own_patient(req.patient_id, user)
    try:
        return agents.run("report", patient_id=req.patient_id, clinical=req.clinical)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/api/reports/{report_id}/finalize")
def api_report_finalize(report_id: int, user=Depends(require_clinician)):
    r0 = db.get_report(report_id)
    if r0 is None:
        raise HTTPException(404, "report not found")
    _own_patient(r0["patient_id"], user)
    from agents import report_agent
    report_agent.finalize(report_id)   # also records the encounter in visit history
    return {"ok": True}


@app.get("/api/reports")
def api_reports(status: str = "all", user=Depends(require_auth)):
    """Default 'all' so freshly-generated drafts are visible, not silently hidden."""
    ids = _clinic_patient_ids(user)
    return [r for r in db.list_reports(None if status == "all" else status) if r["patient_id"] in ids]


@app.get("/api/reports/{report_id}")
def api_report(report_id: int, user=Depends(require_auth)):
    r = db.get_report(report_id)
    if r is None:
        raise HTTPException(404, "report not found")
    _own_patient(r["patient_id"], user)
    return r


# ---- Clinical Assistant (SOAP notes) ----------------------------------------

class ClinicalReq(BaseModel):
    patient_id: int
    complaint: str = ""
    findings: str = ""


@app.post("/api/clinical/draft")
def api_clinical_draft(req: ClinicalReq, user=Depends(require_clinician)):
    _own_patient(req.patient_id, user)
    try:
        return agents.run("clinical_assistant", patient_id=req.patient_id,
                          complaint=req.complaint, findings=req.findings)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/api/clinical/voice")
async def api_clinical_voice(file: UploadFile = File(...), patient_id: int = Form(...),
                             user=Depends(require_clinician)):
    """Dictation -> local Whisper transcript -> grounded SOAP draft. Audio is
    written to a temp file, transcribed on this machine, and deleted."""
    if patient_id is not None:
        _own_patient(patient_id, user)
    import tempfile
    from agents.voice import VoiceUnavailable

    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()
    try:
        return agents.run("voice_soap", audio_path=tmp.name, patient_id=patient_id)
    except VoiceUnavailable as e:
        raise HTTPException(503, f"Voice transcription not ready: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.websocket("/ws/dictation")
async def ws_dictation(ws: WebSocket):
    """Live dictation. Protocol:
      client -> {"type": "auth", "token": "...", "patient_id": N}       (first message)
      client -> binary frames: 16 kHz mono int16 PCM from the microphone
      client -> {"type": "stop"}
      server -> ready | speech{speaking} | partial{text, transcript} | transcript | note | error
    Silero VAD cuts utterances at pauses; each is transcribed on the CPU while the next
    is spoken (agents/voice_stream.py). The token travels in a message, not the URL,
    so it never lands in access logs. Clinician/admin only, own clinic's patients."""
    import asyncio
    import json as _json
    from agents.voice_stream import DictationSession
    from auth import decode_token

    await ws.accept()
    try:
        hello = _json.loads(await asyncio.wait_for(ws.receive_text(), timeout=10))
        payload = decode_token(hello.get("token", ""))
        staff = db.get_staff_by_username(payload.get("sub", ""))
        patient = db.get_patient(int(hello.get("patient_id") or 0))
    except Exception:
        await ws.send_json({"type": "error", "detail": "not authenticated"})
        await ws.close(code=4401)
        return
    if not staff or staff.get("role") not in ("admin", "clinician"):
        await ws.send_json({"type": "error", "detail": "Needs role: clinician or admin"})
        await ws.close(code=4403)
        return
    if patient is None or int(patient.get("clinic_id") or 1) != int(staff.get("clinic_id") or 1):
        await ws.send_json({"type": "error", "detail": "patient not found"})
        await ws.close(code=4404)
        return

    loop = asyncio.get_running_loop()
    out: asyncio.Queue = asyncio.Queue()
    session = DictationSession(lambda e: loop.call_soon_threadsafe(out.put_nowait, e))

    async def sender():                     # the only coroutine that writes to the socket
        while (e := await out.get()) is not None:
            await ws.send_json(e)

    send_task = asyncio.create_task(sender())
    out.put_nowait({"type": "ready"})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                session.jobs.put(None)
                return
            if msg.get("bytes"):
                await loop.run_in_executor(None, session.feed, msg["bytes"])
            elif msg.get("text") and _json.loads(msg["text"]).get("type") == "stop":
                break
        t = await loop.run_in_executor(None, session.finish, None)
        out.put_nowait({"type": "transcript", **t})          # shown before the note is drafted
        if not t["transcript"]:
            out.put_nowait({"type": "error", "detail": "no speech was recognised"})
        else:
            from agents import voice
            complaint, findings = voice.split_dictation(t["transcript"])
            note = await loop.run_in_executor(None, lambda: agents.run(
                "clinical_assistant", patient_id=patient["id"], complaint=complaint,
                findings=findings, findings_from_exam=True))
            out.put_nowait({"type": "note", "note": note, "complaint_from_dictation": complaint,
                            "findings_from_dictation": findings})
    except Exception as e:
        out.put_nowait({"type": "error", "detail": str(e)[:200]})
    finally:
        out.put_nowait(None)
        try:
            await send_task
            await ws.close()
        except Exception:
            pass


# ---- Imaging & Diagnostic (vision agent) ------------------------------------

@app.post("/api/imaging/analyze")
async def api_imaging_analyze(file: UploadFile = File(...), user=Depends(require_auth)):
    """Run the two-stage vision pipeline on an uploaded image. Returns the
    shared contract + verdict + detections, or a clear message if the models
    or vision deps aren't present."""
    import tempfile
    from agents import imaging_agent
    from vision.pipeline import PipelineUnavailable

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    data = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    try:
        return imaging_agent.analyze(tmp.name)
    except PipelineUnavailable as e:
        raise HTTPException(503, f"Imaging pipeline not ready: {e}")
    except Exception as e:
        # most commonly an unreadable / non-image upload
        raise HTTPException(400, f"Could not process image: {e}")
    finally:
        os.unlink(tmp.name)


@app.post("/api/imaging/analyze_batch")
async def api_imaging_batch(files: list[UploadFile] = File(...), patient_id: int = Form(None),
                            user=Depends(require_auth)):
    """Analyze many images in one batched, full-speed pass. If patient_id is given,
    findings attach to that patient's chart on approval."""
    if patient_id is not None:
        _own_patient(patient_id, user)
    import tempfile
    from vision.pipeline import PipelineUnavailable

    paths, names = [], []
    for f in files:
        suffix = os.path.splitext(f.filename or "")[1] or ".jpg"
        data = await f.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.close()
        paths.append(tmp.name)
        names.append(f.filename)
    try:
        results = agents.run("imaging", image_paths=paths, patient_id=patient_id)
        for r, n in zip(results, names):
            r["filename"] = n
        return {"results": results}
    except PipelineUnavailable as e:
        raise HTTPException(503, f"Imaging pipeline not ready: {e}")
    except Exception as e:
        raise HTTPException(400, f"Could not process images: {e}")
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass


@app.post("/api/imaging/analyze_video")
async def api_imaging_video(file: UploadFile = File(...), patient_id: int = Form(None),
                            user=Depends(require_auth)):
    """Sample frames from an uploaded video and run the same two-stage vision
    pipeline as the image endpoints. See imaging_agent.analyze_video for why
    the aggregation is per-class-per-frame, not per detection box."""
    if patient_id is not None:
        _own_patient(patient_id, user)
    import tempfile
    from agents import imaging_agent
    from vision.pipeline import PipelineUnavailable

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    data = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    try:
        result = imaging_agent.analyze_video(tmp.name, patient_id)
        result["filename"] = file.filename
        return result
    except PipelineUnavailable as e:
        raise HTTPException(503, f"Imaging pipeline not ready: {e}")
    except Exception as e:
        raise HTTPException(400, f"Could not process video: {e}")
    finally:
        os.unlink(tmp.name)


# ---- morning job: the "Daily Schedule Job" trigger --------------------------
# Runs the scheduling analysis once each morning so the doctor arrives to a
# prepared, parked-at-approval schedule instead of a spinner.

@app.on_event("startup")
def _warm_models():
    """Load Whisper, Silero VAD and the drafting model in the background, so the first
    dictation doesn't wait ~14 s for model loads. DPMS_WARMUP=0 skips it."""
    if os.getenv("DPMS_WARMUP", "1") == "0":
        return
    import threading

    def warm():
        try:
            from agents import voice
            voice.warm_up()
        except Exception as e:
            db.add_log("voice", "startup", f"Whisper warm-up failed: {e}")
        import llm
        llm.warm_up()
    threading.Thread(target=warm, daemon=True, name="model-warmup").start()


@app.on_event("startup")
def _resume_pipelines():
    """Durability: continue visit pipelines that were mid-step at shutdown."""
    try:
        from agents import visit_pipeline
        visit_pipeline.resume_incomplete()
    except Exception as e:
        db.add_log("pipeline", "startup", f"Could not resume pipelines: {e}")


@app.on_event("startup")
def _start_scheduler():
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except Exception:
        return

    def morning_run():
        # Scoring is fast and the Appointments screen re-scores on open, so the
        # job's value is the heads-up: post the day's outlook to Agent Activity.
        try:
            res = agents.run("scheduling", day=date.today().isoformat())
            exp, preds = res["expected"], res["predictions"]
            high = [p["patient"] for p in preds if p["band"] == "high" and not p["is_new"]]
            msg = (f"Morning schedule scored: {exp['n']} appointments, expect "
                   f"~{exp['attend_mean']} ({exp['low']}–{exp['high']}) to attend")
            if high:
                msg += f"; high no-show risk: {', '.join(high)}"
            db.add_log("scheduling", f"day:{res['date']}", msg + ".")
        except Exception as e:
            db.add_log("scheduling", "job:morning", f"Morning scoring failed: {e}")

    def weekly_retrain():
        try:
            result = agents.run("retraining", force=False, verbose=False)
            status = "promoted" if result.get("promoted") else "skipped"
            db.add_log("retraining", "model:noshow",
                       f"Weekly retrain {status}: {result.get('reason','')}")
        except Exception as e:
            db.add_log("retraining", "model:noshow", f"Weekly retrain error: {e}")

    sched = BackgroundScheduler(daemon=True)
    sched.add_job(morning_run, "cron", hour=7, minute=0, id="daily_schedule_job")
    sched.add_job(weekly_retrain, "cron", day_of_week="sun", hour=0, minute=0,
                  id="weekly_retrain_job")
    sched.start()
    app.state.scheduler = sched


# ---- ML / retraining API ---------------------------------------------------

@app.post("/api/ml/retrain")
def api_retrain(force: bool = False, user=Depends(require_admin)):
    """Trigger a retrain manually. force=true skips the MIN_NEW_OUTCOMES check."""
    try:
        return agents.run("retraining", force=force, verbose=False)
    except Exception as e:
        raise HTTPException(500, str(e))


# ---- Interoperability: FHIR, HL7 v2, DICOM, CSV (interop/) ------------------

@app.get("/api/patients/{patient_id}/fhir")
def api_patient_fhir(patient_id: int, user=Depends(require_auth)):
    """The patient's chart as a validated FHIR R4 Bundle."""
    from interop import fhir
    try:
        _own_patient(patient_id, user)
        return fhir.patient_bundle(patient_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/api/interop/fhir")
def api_import_fhir(resource: dict = Body(...), user=Depends(require_admin)):
    """Create or update a chart from a FHIR Patient (or a Bundle with allergies/conditions)."""
    from interop import fhir
    try:
        return fhir.import_patient(resource)
    except Exception as e:   # pydantic ValidationError, ValueError
        raise HTTPException(400, f"Not a valid FHIR Patient: {str(e)[:300]}")


@app.post("/api/interop/hl7")
async def api_import_hl7(request: Request, user=Depends(require_admin)):
    """HL7 v2 ADT^A04/A08 as the raw request body. Returns the chart id and the ACK."""
    from interop import hl7
    result = hl7.handle_adt((await request.body()).decode("utf-8", "replace"))
    if "error" in result:
        raise HTTPException(400, detail=result)
    return result


@app.post("/api/interop/csv/{kind}")
async def api_import_csv(kind: str, file: UploadFile = File(...), user=Depends(require_admin)):
    """Practice-management export: kind = patients | appointments."""
    from interop import csv_io
    text = (await file.read()).decode("utf-8-sig", "replace")
    if kind == "patients":
        return csv_io.import_patients(text)
    if kind == "appointments":
        return csv_io.import_appointments(text)
    raise HTTPException(404, "kind must be patients or appointments")


@app.post("/api/imaging/dicom")
async def api_imaging_dicom(file: UploadFile = File(...), patient_id: int = Form(None),
                            user=Depends(require_auth)):
    """A DICOM image through the same gate + detector as a photo upload."""
    if patient_id is not None:
        _own_patient(patient_id, user)
    import tempfile
    from interop import dicom
    from vision.pipeline import PipelineUnavailable
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
    tmp.write(await file.read())
    tmp.close()
    try:
        r = dicom.screen(tmp.name, patient_id)
        r["filename"] = file.filename
        return r
    except PipelineUnavailable as e:
        raise HTTPException(503, f"Imaging pipeline not ready: {e}")
    except Exception as e:
        raise HTTPException(400, f"Could not read DICOM: {e}")
    finally:
        os.unlink(tmp.name)


@app.get("/api/traces")
def api_traces(limit: int = 100, name: str = "", user=Depends(require_auth)):
    """Recent trace spans (agent runs, model calls, tool calls, approvals), newest first."""
    import tracing
    return tracing.recent(min(max(limit, 1), 500), name or None)


@app.get("/api/health")
def api_health(user=Depends(require_auth)):
    """Is everything the agents depend on up? Plus 24 h of per-step counts, errors, latency."""
    import llm
    import tracing
    from vision import pipeline as vp
    status = {"database": "ok"}
    try:
        tags = {m["model"] for m in llm._client().list()["models"]}
        status["ollama"] = "ok"
        wanted = {llm.resolve_model(t) for t in ("soap_note", "report_summary", "copilot")} | {"nomic-embed-text"}
        status["models"] = {m: any(t == m or t.startswith(m + ":") for t in tags) for m in sorted(wanted)}
    except Exception as e:
        status["ollama"] = f"unreachable: {str(e)[:120]}"
    status["vision_models"] = all(os.path.exists(p) for p in (vp.GATE_WEIGHTS, vp.GATE_META, vp.DETECTOR_WEIGHTS))
    status["steps_24h"] = tracing.stats(24)
    return status


@app.get("/api/ml/retrain-history")
def api_retrain_history(user=Depends(require_auth)):
    return db.retrain_history()


@app.get("/api/ml/model-registry")
def api_model_registry(user=Depends(require_auth)):
    db._ml_schema()
    conn = db.connect()
    rows = conn.execute(
        "SELECT id,version,path,auc,trained_on,status,created_at "
        "FROM model_registry ORDER BY id DESC LIMIT 20").fetchall()
    conn.close()
    return [dict(r) for r in rows]


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
