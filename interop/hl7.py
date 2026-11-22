"""
HL7 v2 ADT feed: patient registration (A04) and update (A08) messages from a
hospital system or practice-management system. Small parser of our own — v2 is
pipe-delimited text, and these messages only need a handful of fields:

  MSH-9   message type (ADT^A04 / ADT^A08)    MSH-10  control id (echoed in the ACK)
  PID-5   name (family^given)                 PID-7   date of birth (YYYYMMDD)
  PID-8   sex (M/F)                           PID-13  phone
  AL1-3   allergen (code^text)                DG1-3/4 diagnosis (code^text / text)

Returns the chart id plus an ACK (MSA|AA) — or AE with the reason — to send back.
"""
from __future__ import annotations

from datetime import date, datetime

from interop.fhir import _upsert


def _segments(message: str) -> list[list[str]]:
    lines = [ln for ln in message.replace("\r\n", "\r").replace("\n", "\r").split("\r") if ln.strip()]
    return [ln.split("|") for ln in lines]


def _field(seg: list[str], n: int, comp: int | None = None) -> str:
    """HL7 numbering: MSH-n is seg[n-1] (MSH-1 is the separator itself); others seg[n]."""
    idx = n - 1 if seg[0] == "MSH" else n
    val = seg[idx] if idx < len(seg) else ""
    if comp is not None:
        parts = val.split("^")
        val = parts[comp] if comp < len(parts) else ""
    return val.strip()


def _ack(msh: list[str] | None, code: str, text: str = "") -> str:
    ctrl = _field(msh, 10) if msh else ""
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    return (f"MSH|^~\\&|DENTAGENT|CLINIC|||{now}||ACK|ACK{ctrl}|P|2.5\r"
            f"MSA|{code}|{ctrl}" + (f"|{text}" if text else "") + "\r")


def handle_adt(message: str) -> dict:
    segs = _segments(message)
    msh = next((s for s in segs if s[0] == "MSH"), None)
    try:
        if msh is None:
            raise ValueError("no MSH segment")
        mtype = _field(msh, 9)
        if not mtype.startswith("ADT^A04") and not mtype.startswith("ADT^A08"):
            raise ValueError(f"unsupported message type {mtype or '(none)'}; expected ADT^A04 or ADT^A08")
        pid = next((s for s in segs if s[0] == "PID"), None)
        if pid is None:
            raise ValueError("no PID segment")
        family, given = _field(pid, 5, 0), _field(pid, 5, 1)
        name = f"{given} {family}".strip()
        if not name:
            raise ValueError("PID-5 (patient name) is empty")
        age = None
        dob = _field(pid, 7)[:8]
        if len(dob) == 8:
            bd = datetime.strptime(dob, "%Y%m%d").date()
            t = date.today()
            age = t.year - bd.year - ((t.month, t.day) < (bd.month, bd.day))
        sex = _field(pid, 8)[:1].upper() if _field(pid, 8)[:1].upper() in ("M", "F") else ""
        phone = _field(pid, 13, 0)
        allergies = ", ".join(filter(None, (_field(s, 3, 1) or _field(s, 3, 0) for s in segs if s[0] == "AL1")))
        conditions = ", ".join(filter(None, (_field(s, 3, 1) or _field(s, 4) for s in segs if s[0] == "DG1")))
        result = _upsert(name, phone, age, sex, allergies, conditions)
        return {**result, "message_type": mtype, "ack": _ack(msh, "AA")}
    except ValueError as e:
        return {"error": str(e), "ack": _ack(msh, "AE", str(e)[:80])}
