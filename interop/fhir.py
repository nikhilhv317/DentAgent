"""
FHIR R4 export/import. Uses the R4B models from `fhir.resources` to validate
every resource, so what leaves DentAgent is structurally valid FHIR.

Export (one patient -> Bundle of type "collection"):
  Patient, AllergyIntolerance (one per allergy), Condition (one per condition),
  Encounter (one per visit), Observation (one per imaging finding),
  DocumentReference (one per finalized report, text attached as base64).
"""
from __future__ import annotations

import base64
import re
import uuid
from datetime import date

import db

SYSTEM_MRN = "urn:dentagent:mrn"


def _ref(kind: str, rid) -> dict:
    return {"reference": f"{kind}/{rid}"}


def _gender(sex: str) -> str:
    return {"M": "male", "F": "female"}.get((sex or "").upper()[:1], "unknown")


def patient_bundle(patient_id: int) -> dict:
    from fhir.resources.R4B.bundle import Bundle
    p = db.get_patient(int(patient_id))
    if p is None:
        raise ValueError(f"patient {patient_id} not found")
    pid = f"dv-{p['id']}"
    family, _, given = (p["name"] or "").rpartition(" ")
    patient = {"resourceType": "Patient", "id": pid,
               "identifier": [{"system": SYSTEM_MRN, "value": p["mrn"]}],
               "name": [{"text": p["name"], "family": family or given, "given": [given] if family else []}],
               "gender": _gender(p["sex"]), "active": bool(p.get("active", 1))}
    if p.get("phone"):
        patient["telecom"] = [{"system": "phone", "value": p["phone"]}]
    if p.get("age"):   # the chart stores age, not date of birth — export the year as an estimate
        patient["extension"] = [{"url": "urn:dentagent:age-years", "valueInteger": int(p["age"])}]
    resources = [patient]
    for i, a in enumerate(t.strip() for t in re.split(r"[,;]", p.get("allergies") or "") if t.strip()):
        resources.append({"resourceType": "AllergyIntolerance", "id": f"{pid}-allergy-{i}",
                          "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]},
                          "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification", "code": "confirmed"}]},
                          "code": {"text": a}, "patient": _ref("Patient", pid)})
    for i, c in enumerate(t.strip() for t in re.split(r"[,;]", p.get("conditions") or "") if t.strip()):
        resources.append({"resourceType": "Condition", "id": f"{pid}-condition-{i}",
                          "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
                          "code": {"text": c}, "subject": _ref("Patient", pid)})
    for i, v in enumerate(p.get("visits", [])):
        reasons = [{"text": v["complaint"] or "visit"}]
        if v["diagnosis"]:
            reasons.append({"text": f"Diagnosis: {v['diagnosis']}"})
        resources.append({"resourceType": "Encounter", "id": f"{pid}-visit-{i}", "status": "finished",
                          "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "ambulatory"},
                          "subject": _ref("Patient", pid), "period": {"start": v["date"], "end": v["date"]},
                          "reasonCode": reasons})
    for i, n in enumerate(db.get_notes(int(patient_id), "imaging")):
        resources.append({"resourceType": "Observation", "id": f"{pid}-imaging-{i}", "status": "preliminary",
                          "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "imaging"}]}],
                          "code": {"text": "Dental image AI screening (clinician-approved)"},
                          "subject": _ref("Patient", pid), "effectiveDateTime": (n["created_at"] or "")[:10] or date.today().isoformat(),
                          "valueString": n["content"]})
    for r in db.list_reports("finalized"):
        if r["patient_id"] != p["id"]:
            continue
        full = db.get_report(r["id"])
        text = f"Summary: {full['ai_summary']}\nImaging: {full['imaging']}\nClinical: {full['clinical']}"
        resources.append({"resourceType": "DocumentReference", "id": f"{pid}-report-{r['id']}", "status": "current",
                          "type": {"text": "Dental visit report"}, "subject": _ref("Patient", pid),
                          "date": (full["finalized_at"] or "").replace(" ", "T") + "Z" if full["finalized_at"] else None,
                          "content": [{"attachment": {"contentType": "text/plain",
                                                      "data": base64.b64encode(text.encode()).decode()}}]})
        resources[-1] = {k: val for k, val in resources[-1].items() if val is not None}
    bundle = {"resourceType": "Bundle", "id": str(uuid.uuid4()), "type": "collection",
              "entry": [{"fullUrl": f"urn:dentagent:{r['resourceType']}/{r['id']}", "resource": r} for r in resources]}
    Bundle.model_validate(bundle)          # raises if anything isn't valid FHIR R4
    return bundle


def import_patient(resource: dict) -> dict:
    """Create (or reuse) a chart from a FHIR Patient, or a Bundle holding a Patient
    plus AllergyIntolerance / Condition resources. Returns {patient_id, created}."""
    from fhir.resources.R4B.patient import Patient
    entries = ([e["resource"] for e in resource.get("entry", [])] if resource.get("resourceType") == "Bundle"
               else [resource])
    pats = [r for r in entries if r.get("resourceType") == "Patient"]
    if not pats:
        raise ValueError("no Patient resource found")
    Patient.model_validate(pats[0])
    p = pats[0]
    nm = (p.get("name") or [{}])[0]
    name = nm.get("text") or " ".join(nm.get("given", []) + [nm.get("family", "")]).strip()
    if not name:
        raise ValueError("Patient has no name")
    phone = next((t["value"] for t in p.get("telecom", []) if t.get("system") == "phone"), "")
    sex = {"male": "M", "female": "F"}.get(p.get("gender", ""), "")
    age = None
    if p.get("birthDate"):
        bd = date.fromisoformat(p["birthDate"][:10])
        t = date.today()
        age = t.year - bd.year - ((t.month, t.day) < (bd.month, bd.day))
    allergies = ", ".join(r.get("code", {}).get("text", "") for r in entries
                          if r.get("resourceType") == "AllergyIntolerance" and r.get("code", {}).get("text"))
    conditions = ", ".join(r.get("code", {}).get("text", "") for r in entries
                           if r.get("resourceType") == "Condition" and r.get("code", {}).get("text"))
    return _upsert(name, phone, age, sex, allergies, conditions)


def _upsert(name, phone, age, sex, allergies, conditions) -> dict:
    """Shared by every importer: reuse a chart only on a name + phone match."""
    match = next((c for c in db.find_duplicate_candidates(name, phone)
                  if c["match_reason"] == "phone" and c["name"].strip().lower() == name.strip().lower()), None)
    if match:
        fields = {k: v for k, v in {"age": age, "sex": sex, "allergies": allergies,
                                    "conditions": conditions}.items() if v}
        if fields:
            db.update_patient(match["id"], fields)
        return {"patient_id": match["id"], "created": False}
    pid, mrn = db.create_patient(name, age, sex, phone, allergies, conditions)
    return {"patient_id": pid, "mrn": mrn, "created": True}
