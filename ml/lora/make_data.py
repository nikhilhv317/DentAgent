"""
Synthetic, grounded training data for the DentAgent LoRA adapter (roadmap item 9).

There are no real de-identified dental notes in this project, so the training
set is generated: random (fictional) patients, histories and visits, turned into
the EXACT prompts the app sends (clinical_assistant._prompt, report_agent._prompt
and their SYSTEM messages), paired with target notes that show the behaviour the
evals found missing in the base model:

  - cite a source tag on every sentence ([C] [F] [R] [I] [V2] ...)
  - never invent an examination: with no findings entered, the Objective says
    "No examination findings recorded today."
  - state every allergy and condition exactly as recorded
  - hedge diagnoses not established by the findings ("likely", "possible")

Every example is checked with the app's own grounding and safety checks before it
is kept, so the targets can't teach the model something the checks would flag.
Names, patients and wording from evals/cases are excluded (checked at the end).

    python ml/lora/make_data.py --n-soap 1200 --n-report 400

v2 (after the first adapter's evals): plans carry allergy/condition precautions
(latex-free materials, pregnancy, bleeding, penicillin alternatives, ...), and
extraction and halitosis scenarios were added — v1 was perfectly grounded but
gave every patient the same plan regardless of their record.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from agents import clinical_assistant, report_agent  # noqa: E402
from agents.safety import safety_check  # noqa: E402

FIRST = ["Aarav", "Meera", "Rohit", "Nisha", "Farhan", "Divya", "Imran", "Pooja", "Rahul", "Shreya", "Aditya",
         "Fatima", "Gaurav", "Hema", "Irfan", "Jaya", "Kiran", "Lata", "Mohan", "Neha", "Omkar", "Pallavi",
         "Rakesh", "Sunita", "Tarun", "Uma", "Varun", "Yamini", "Zoya", "Bhavna", "Chetan", "Esha", "Harish",
         "Isha", "Jatin", "Komal", "Naveen", "Ritu", "Sanjay", "Tanvi"]
LAST = ["Joshi", "Patil", "Kamath", "Bhat", "Hegde", "Shah", "Mehta", "Verma", "Gupta", "Naidu", "Sharma",
        "Chopra", "Banerjee", "Das", "Mishra", "Pandey", "Fernandes", "D'Souza", "Kapoor", "Saxena"]
ALLERGIES = ["Penicillin", "Latex", "Ibuprofen", "Aspirin", "Sulfa drugs", "Chlorhexidine", "Lidocaine"]
CONDITIONS = ["Type 2 diabetes", "Hypertension", "Asthma", "Pregnant (1st trimester)", "Pregnant (3rd trimester)",
              "Bleeding disorder", "Hypothyroidism", "Epilepsy", "On warfarin", "Rheumatoid arthritis"]
REGIONS = [("lower right", [46, 47, 45, 48]), ("lower left", [36, 37, 35, 38]),
           ("upper right", [16, 17, 15, 18]), ("upper left", [26, 27, 25, 28])]

# scenario -> complaint, findings (exam), imaging, assessment with/without findings, plan, past diagnosis
SCENARIOS = [
    dict(key="caries", complaint=["pain in the {region} molar", "toothache {region} for {d} days", "pain on chewing {region}"],
         findings=["Deep caries on #{t}; tender to percussion.", "Caries on #{t}. Tender to percussion on #{t}.",
                   "Distal caries on #{t}, no swelling."],
         imaging="Disease Detected: Advanced caries (x1, {p}%)",
         a_exam="Symptomatic carious #{t}, likely pulpitis", a_img="Possible advanced caries on imaging",
         a_none="possible caries", plan=["Periapical radiograph of #{t}", "Restoration or pulp therapy as indicated"],
         past="Caries #{t}"),
    dict(key="gingivitis", complaint=["bleeding gums when brushing", "gums bleed on brushing", "bleeding gums for {d} weeks"],
         findings=["Generalised bleeding on probing. Calculus on lower anteriors.", "Bleeding on probing; plaque on lower anteriors."],
         imaging="Disease Detected: Gingivitis (x1, {p}%)",
         a_exam="Bleeding on probing, likely gingivitis", a_img="Possible gingivitis on imaging",
         a_none="possible gingivitis", plan=["Scaling and polishing", "Oral hygiene instruction"],
         past="Generalised gingivitis"),
    dict(key="abscess", complaint=["swollen gum near the {region} tooth", "swelling {region} for {d} days"],
         findings=["Localised swelling and tenderness at #{t}. Pus on pressure.", "Swelling at #{t}; tender to percussion."],
         imaging="Disease Detected: Abnormal (x1, {p}%)",
         a_exam="Localised swelling at #{t}, likely periapical abscess", a_img="Abnormal finding on imaging, needs review",
         a_none="possible infection", plan=["Periapical radiograph", "Drainage and review if indicated"],
         past="Localised abscess #{t}"),
    dict(key="mobility", complaint=["loose tooth {region}", "wobbly tooth in the {region}"],
         findings=["Grade {g} mobility #{t}. {mm} mm pockets.", "Mobility grade {g} on #{t}; {mm} mm pocketing."],
         imaging=None, a_exam="Mobility of #{t}, likely periodontitis", a_img=None,
         a_none="possible periodontitis", plan=["Full periodontal charting", "Deep scaling"],
         past="Chronic periodontitis"),
    dict(key="sensitivity", complaint=["sensitivity to cold {region}", "sharp pain with cold drinks {region}"],
         findings=["Gingival recession on #{t}. Sensitivity to cold on #{t}.", "Recession on #{t}; no caries."],
         imaging=None, a_exam="Recession on #{t}, likely dentine hypersensitivity", a_img=None,
         a_none="possible dentine sensitivity", plan=["Desensitising toothpaste", "Review in 4 weeks"],
         past="Exposed dentine #{t}"),
    dict(key="fracture", complaint=["filling fell out on the {region} side", "chipped tooth {region}"],
         findings=["Fractured restoration #{t}.", "Fractured cusp on #{t}; no tenderness."],
         imaging=None, a_exam="Fractured restoration on #{t}", a_img=None,
         a_none="possible fractured restoration", plan=["Replace restoration on #{t}"],
         past="Fractured restoration #{t}"),
    dict(key="pericoronitis", complaint=["pain behind the last tooth {region}", "wisdom tooth ache {region}"],
         findings=["Partially erupted #{t} with inflamed operculum.", "Inflammation around partially erupted #{t}."],
         imaging=None, a_exam="Inflammation around #{t}, likely pericoronitis", a_img=None,
         a_none="possible pericoronitis", plan=["Irrigation and oral hygiene", "OPG to assess #{t}"],
         past="Impacted #{t}"),
    dict(key="routine", complaint=["routine check-up", "six-month review", "cleaning visit"],
         findings=["Mild plaque on lower anteriors.", "Mild calculus on lower anteriors."],
         imaging="Healthy — no findings",
         a_exam="Mild deposits, otherwise satisfactory", a_img="Healthy screen on imaging",
         a_none="routine review", plan=["Scaling and polishing", "Recall in 6 months"],
         past="Sound dentition"),
    dict(key="ulcer", complaint=["painful sore in the mouth", "mouth ulcer for {d} days"],
         findings=["Ulcer on the left buccal mucosa, {mm} mm.", "Ulcer on the lateral tongue, {mm} mm."],
         imaging="Disease Detected: Ulcer (x1, {p}%)",
         a_exam="Ulcer, likely aphthous", a_img="Possible ulcer on imaging",
         a_none="possible ulcer", plan=["Topical analgesic gel", "Review in 2 weeks; refer if persistent"],
         past="Aphthous ulcer"),
    # v2: extraction and halitosis (v1 had neither, so an extraction consult got a gum-disease plan)
    dict(key="extraction", complaint=["needs a tooth removed {region}", "extraction review {region}"],
         findings=["Non-restorable #{t}.", "Non-restorable #{t}; grade 1 mobility."],
         imaging=None, a_exam="Non-restorable #{t}, extraction indicated", a_img=None,
         a_none="possible extraction need", plan=["Extraction of #{t} with informed consent",
                                                  "Post-extraction instructions"],
         past="Non-restorable #{t}"),
    dict(key="halitosis", complaint=["bad smell from the mouth", "halitosis for {d} weeks"],
         findings=["Plaque and calculus generalised.", "Coated tongue; plaque on lower anteriors."],
         imaging=None, a_exam="Plaque deposits, likely cause of halitosis", a_img=None,
         a_none="possible plaque-related halitosis", plan=["Scaling and polishing",
                                                           "Tongue cleaning and oral hygiene instruction"],
         past="Halitosis"),
]

# v2: the plan must reflect the patient, not just the complaint. v1 plans ignored
# allergies and conditions, and the fine-tuned model learned to do the same (eval:
# no latex-free precaution, no pregnancy or bleeding precautions).
ANTIBIOTIC_SCENARIOS = {"abscess", "pericoronitis", "extraction"}
INVASIVE_SCENARIOS = {"extraction", "abscess", "mobility", "caries", "gingivitis", "routine", "halitosis"}


def _precautions(p: dict, scenario: str) -> list[str]:
    a, c, out = p["allergies"], p["conditions"], []
    if "Latex" in a:
        out.append("Use latex-free gloves and materials (latex allergy)")
    if "Penicillin" in a and scenario in ANTIBIOTIC_SCENARIOS:
        out.append("If antibiotics are needed, avoid penicillins because of the allergy; consider clindamycin")
    if "Lidocaine" in a and scenario in INVASIVE_SCENARIOS:
        out.append("Use a non-lidocaine local anaesthetic (lidocaine allergy)")
    if "Chlorhexidine" in a:
        out.append("Avoid chlorhexidine rinses (allergy)")
    if "Sulfa drugs" in a and scenario in ANTIBIOTIC_SCENARIOS:
        out.append("Avoid sulfonamide antibiotics (sulfa allergy)")
    if "Pregnant" in c:
        out.append("Pregnancy: defer elective radiographs and non-urgent treatment; pregnancy-safe analgesia only")
    if ("Bleeding disorder" in c or "On warfarin" in c) and scenario in INVASIVE_SCENARIOS:
        out.append("Bleeding precautions: haematology or INR check before invasive treatment; local haemostatic measures")
    if "Type 2 diabetes" in c:
        out.append("Monitor healing and infection risk given diabetes")
    if "Hypertension" in c and scenario in INVASIVE_SCENARIOS:
        out.append("Check blood pressure before treatment")
    if "Asthma" in c:
        out.append("Keep the patient's inhaler available during treatment")
    if "Epilepsy" in c and scenario in INVASIVE_SCENARIOS:
        out.append("Confirm seizure control and medication before treatment")
    return out


def _fill(s: str, ctx: dict) -> str:
    return s.format(**ctx)


def _patient(rng: random.Random, pid: int) -> dict:
    allergies = ", ".join(rng.sample(ALLERGIES, rng.choice([0, 0, 0, 1, 1, 2])))
    conditions = ", ".join(rng.sample(CONDITIONS, rng.choice([0, 0, 1, 1, 2])))
    visits = []
    for _ in range(rng.choice([0, 1, 1, 2, 3, 4])):
        sc = rng.choice(SCENARIOS)
        region, teeth = rng.choice(REGIONS)
        ctx = {"region": region, "t": rng.choice(teeth), "d": rng.randint(2, 10), "g": 2, "mm": 5, "p": 70}
        visits.append({"date": f"{rng.randint(2021, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                       "complaint": _fill(rng.choice(sc["complaint"]), ctx).capitalize(),
                       "diagnosis": _fill(sc["past"], ctx), "treatment": _fill(sc["plan"][0], ctx),
                       "notes": ""})
    visits.sort(key=lambda v: v["date"])
    return {"id": pid, "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "age": rng.randint(18, 80),
            "sex": rng.choice("MF"), "mrn": f"SYN-{pid:05d}", "allergies": allergies,
            "conditions": conditions, "visits": visits}


def _record_sentences(p: dict) -> list[str]:
    out = []
    if p["allergies"]:
        out.append(f"Allergies: {p['allergies']} [R].")
    if p["conditions"]:
        out.append(f"Medical conditions: {p['conditions']} [R].")
    return out


def _pain_plan(p: dict) -> str:
    if any(a in p["allergies"] for a in ("Ibuprofen", "Aspirin")):
        return "Paracetamol for pain; avoid NSAIDs given the recorded allergy"
    return "Paracetamol for pain if needed"


def soap_example(rng: random.Random, pid: int) -> dict:
    p = _patient(rng, pid)
    sc = rng.choice(SCENARIOS)
    region, teeth = rng.choice(REGIONS)
    ctx = {"region": region, "t": rng.choice(teeth), "d": rng.randint(2, 10), "g": rng.choice([1, 2]),
           "mm": rng.choice([4, 5, 6]), "p": rng.randint(55, 92)}
    complaint = _fill(rng.choice(sc["complaint"]), ctx)
    mode = rng.choices(["exam", "imaging", "none"], weights=[5, 2 if sc["imaging"] else 0, 3])[0]
    findings = _fill(rng.choice(sc["findings"]), ctx) if mode == "exam" else \
        _fill(sc["imaging"], ctx) if mode == "imaging" else ""
    from_exam = mode == "exam"

    subj = [f"Patient reports {complaint} [C]."]
    if p["visits"] and rng.random() < 0.6:
        n = rng.randrange(len(p["visits"]))
        v = p["visits"][n]
        subj.append(f"Previous {v['diagnosis'][0].lower() + v['diagnosis'][1:]} ({v['date']}) [V{n + 1}].")
    subj += _record_sentences(p)

    if mode == "exam":
        obj = " ".join(s.strip() + " [F]." for s in findings.rstrip(".").replace("; ", ". ").split(". "))
        assess = f"{_fill(sc['a_exam'], ctx)} [F]."
    elif mode == "imaging":
        obj = f"No examination findings recorded today. Imaging: {findings} [F]."
        assess = f"{_fill(sc['a_img'], ctx)}, clinical examination required [F]."
    else:
        obj = "No examination findings recorded today."
        assess = f"{complaint[0].upper() + complaint[1:]} reported, {_fill(sc['a_none'], ctx)} to be confirmed by examination [C]."
    # a tooth number in the plan only when the clinician's findings named one — the
    # model must not learn to invent tooth numbers anywhere in the note
    steps = sc["plan"] if mode == "exam" else [x.replace(" of #{t}", "").replace(" on #{t}", "")
                                               .replace(" to assess #{t}", "").replace("#{t}", "the affected tooth")
                                               for x in sc["plan"]]
    plan = [_fill(x, ctx) for x in steps] + ([_pain_plan(p)] if sc["key"] in ("caries", "abscess", "ulcer", "extraction") else [])
    plan += _precautions(p, sc["key"])
    if mode != "exam":
        plan.insert(0, "Clinical examination")
    soap = {"subjective": " ".join(subj), "objective": obj, "assessment": assess, "plan": plan}

    issues = clinical_assistant.ground_soap(p, complaint, findings, soap, from_exam, require_citations=True)
    text = "\n".join(soap[k] for k in ("subjective", "objective", "assessment")) + "\n" + "; ".join(plan)
    warnings = safety_check(p, text)
    prompt = clinical_assistant._prompt(p, complaint, findings, from_exam)
    return {"messages": [{"role": "system", "content": clinical_assistant.SYSTEM},
                         {"role": "user", "content": prompt},
                         {"role": "assistant", "content": json.dumps(soap, ensure_ascii=False)}],
            "ok": not issues and not warnings, "issues": [i["claim"] for i in issues] + warnings,
            "task": "soap", "complaint": complaint, "name": p["name"]}


CLINICAL = {"caries": "Restoration placed on #{t}.", "gingivitis": "Scaling and polishing done.",
            "abscess": "Drainage performed at #{t}.", "mobility": "Deep scaling done; mobility #{t} noted.",
            "sensitivity": "Desensitising agent applied to #{t}.", "fracture": "Restoration on #{t} replaced.",
            "pericoronitis": "Irrigation around #{t} done.", "routine": "Scaling and polishing done.",
            "ulcer": "Topical gel prescribed for the ulcer.",
            "extraction": "Extraction of #{t} under local anaesthesia; haemostasis achieved.",
            "halitosis": "Scaling done; tongue cleaning advised."}


def report_example(rng: random.Random, pid: int) -> dict:
    p = _patient(rng, pid)
    sc = rng.choice(SCENARIOS)
    region, teeth = rng.choice(REGIONS)
    ctx = {"region": region, "t": rng.choice(teeth), "d": 3, "g": 2, "mm": 5, "p": rng.randint(55, 92)}
    imaging = _fill(sc["imaging"], ctx) if sc["imaging"] and rng.random() < 0.6 else "No imaging findings on record."
    clinical = _fill(CLINICAL[sc["key"]], ctx) if rng.random() < 0.85 else ""
    lines = _record_sentences(p) or ["No allergies or medical conditions recorded [R]."]
    lines.append("No imaging findings [I]." if imaging.startswith("No imaging") else f"Imaging: {imaging} [I].")
    if clinical:
        lines.append(clinical.rstrip(".") + " [F].")
    else:
        lines.append("No clinician findings entered today.")
    summary = " ".join(lines)
    issues = report_agent.ground_report(p, imaging, clinical, summary, exam_given=bool(clinical),
                                        require_citations=True)
    warnings = safety_check(p, summary)
    return {"messages": [{"role": "system", "content": report_agent.SYSTEM},
                         {"role": "user", "content": report_agent._prompt(p, imaging, clinical)},
                         {"role": "assistant", "content": json.dumps({"summary": summary}, ensure_ascii=False)}],
            "ok": not issues and not warnings, "issues": [i["claim"] for i in issues] + warnings,
            "task": "report", "complaint": "", "name": p["name"]}


def _eval_overlap(examples: list[dict]) -> list[str]:
    """Names / complaint strings from evals/cases must not appear in training data."""
    import yaml
    cases = yaml.safe_load(open(os.path.join(ROOT, "evals", "cases", "drafting.yaml"), encoding="utf-8"))
    eval_complaints = {c.get("complaint", "").lower() for c in cases if c.get("complaint")}
    import db
    seed_names = {row[2].lower() for row in db.SEED_PATIENTS}
    bad = [e["name"] for e in examples if e["name"].lower() in seed_names]
    bad += [e["complaint"] for e in examples if e["complaint"].lower() in eval_complaints]
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-soap", type=int, default=1200)
    ap.add_argument("--n-report", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    kept, dropped = [], []
    for i in range(args.n_soap + args.n_report):
        ex = soap_example(rng, i) if i < args.n_soap else report_example(rng, i)
        (kept if ex["ok"] else dropped).append(ex)
    overlap = _eval_overlap(kept)
    if overlap:
        raise SystemExit(f"eval overlap found, refusing to write: {overlap[:5]}")
    rng.shuffle(kept)
    n_val = max(20, len(kept) // 20)
    os.makedirs(args.out, exist_ok=True)
    for name, rows in (("val.jsonl", kept[:n_val]), ("train.jsonl", kept[n_val:])):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps({"messages": e["messages"], "task": e["task"]}, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)} (train {len(kept) - n_val}, val {n_val}), dropped {len(dropped)} that failed the checks")
    from collections import Counter
    for reason, n in Counter(i for e in dropped for i in e["issues"]).most_common(8):
        print(f"  dropped because: {reason} ({n})")


if __name__ == "__main__":
    main()
