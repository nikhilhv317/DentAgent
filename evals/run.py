"""
DentAgent evaluation runner.

    python evals/run.py                       # offline checker suite — what CI runs
    python evals/run.py --suite live          # drafting + copilot on the local Ollama model
    python evals/run.py --suite all --json evals/results/run.json

Suites
  checks    No model needed. Feeds hand-labelled drafts to the safety, grounding
            and action-claim checks and compares what they flag with what they
            should flag. Exit 1 on any regression. A case marked `known_gap`
            documents a limitation: it is expected to fail and is reported
            separately; if one starts passing it is reported as FIXED so the
            label can be removed.
  drafting  Runs the Clinical Assistant (SOAP) and Report agents on fixed cases
            and measures how often a draft has unsourced findings (grounding
            failure rate) or safety warnings (safety violation rate).
  copilot   Asks the Copilot fixed questions and checks it called the right tools,
            avoided forbidden ones (including a prompt-injection case), proposed
            the expected actions and stated the expected facts.

Live suites need Ollama. They report rates; the --max-*/--min-* flags turn those
rates into pass/fail gates.

Every run uses a fresh temporary database seeded with the standard 12-patient
roster, so results never depend on, or touch, your dpms.db.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

WRITE_TOOLS = {"propose_followup", "propose_chart_note", "propose_patient_message"}


def _fresh_db():
    """Point the app at a throwaway database BEFORE anything imports db."""
    os.environ["DPMS_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="dentagent-eval-"), "eval.db")
    import db
    db.init_db()
    return db


def _load(name: str) -> list[dict]:
    import yaml
    with open(os.path.join(HERE, "cases", f"{name}.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


# safety_check returns prose; map each warning to a short, comparable code
_SAFETY_CODES = [
    (re.compile(r"^Draft says no allergies"), lambda m: "no_allergy_claim"),
    (re.compile(r"^Draft prescribes '.+?', but the record lists an allergy to '(.+?)'"), lambda m: f"prescribed:{m[1]}"),
    (re.compile(r"^Recorded allergy '(.+?)' appears NEGATED"), lambda m: f"negated:{m[1]}"),
    (re.compile(r"^Recorded allergy '(.+?)' is not mentioned"), lambda m: f"missing_allergy:{m[1]}"),
    (re.compile(r"^Recorded condition '(.+?)' is not mentioned"), lambda m: f"missing_condition:{m[1]}"),
]


def _safety_codes(warnings: list[str]) -> set[str]:
    out = set()
    for w in warnings:
        for rx, code in _SAFETY_CODES:
            m = rx.search(w)
            if m:
                out.add(code(m))
                break
        else:
            out.add("other:" + w[:60])
    return out


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "n/a"


# ---- checks: deterministic checkers vs labelled drafts ----------------------

def _run_check(case: dict, db) -> set | bool:
    kind = case["kind"]
    if kind == "soap":
        from agents import clinical_assistant
        found = clinical_assistant.ground_soap(db.get_patient(case["patient"]), case.get("complaint", ""),
                                               case.get("findings", ""), case["soap"], case.get("exam", False),
                                               require_citations=case.get("citations", False))
        return {f["claim"] for f in found}
    if kind == "report":
        from agents import report_agent
        found = report_agent.ground_report(db.get_patient(case["patient"]), case.get("imaging", ""),
                                           case.get("clinical", ""), case["summary"], case.get("exam", False),
                                           require_citations=case.get("citations", False))
        return {f["claim"] for f in found}
    if kind == "grounding":
        from agents import grounding
        found = grounding.check(case["draft"], case.get("sources", ""),
                                exam_findings_given=case.get("exam", False),
                                history=case.get("history", ""),
                                allow_negated=case.get("allow_negated", False))
        return {f["claim"] for f in found}
    if kind == "safety":
        from agents.safety import safety_check
        return _safety_codes(safety_check(db.get_patient(case["patient"]), case["text"]))
    if kind == "action_claim":
        from agents import copilot
        _, flagged = copilot._check_action_claims(case["answer"], [{"id": "1"}] * case.get("actions_created", 0))
        return flagged
    raise ValueError(f"unknown kind {kind!r}")


def suite_checks(db, only: str | None) -> dict:
    rows, tp, fp, fn = [], 0, 0, 0
    for case in _load("checks"):
        if only and only not in case["id"]:
            continue
        got = _run_check(case, db)
        expect = case["expect"] if case["kind"] == "action_claim" else set(case.get("expect") or [])
        ok = got == expect
        if isinstance(got, set):   # flag-level counts across every set-valued case
            tp += len(got & expect); fp += len(got - expect); fn += len(expect - got)
        gap = bool(case.get("known_gap"))
        status = ("FIXED" if ok else "gap") if gap else ("pass" if ok else "FAIL")
        rows.append({"id": case["id"], "kind": case["kind"], "status": status,
                     "expected": sorted(expect) if isinstance(expect, set) else expect,
                     "got": sorted(got) if isinstance(got, set) else got,
                     "why": case.get("why", "")})

    print(f"\n== checks ({len(rows)} cases)")
    for r in rows:
        mark = {"pass": " ok ", "FAIL": "FAIL", "gap": "gap ", "FIXED": "FIXD"}[r["status"]]
        line = f"  [{mark}] {r['kind']:<12} {r['id']}"
        if r["status"] != "pass":
            line += f"\n           expected {r['expected']}  got {r['got']}"
        print(line)
    n = lambda s: sum(r["status"] == s for r in rows)
    summary = {"cases": len(rows), "pass": n("pass"), "fail": n("FAIL"), "known_gaps": n("gap"),
               "fixed": n("FIXED"), "flag_precision": round(tp / (tp + fp), 3) if tp + fp else None,
               "flag_recall": round(tp / (tp + fn), 3) if tp + fn else None}
    print(f"  -> {summary['pass']} pass, {summary['fail']} FAIL, {summary['known_gaps']} known gaps, "
          f"{summary['fixed']} fixed | flag precision {summary['flag_precision']}, "
          f"recall {summary['flag_recall']}")
    return {"summary": summary, "cases": rows}


# ---- drafting: SOAP + report agents on the live model ------------------------

def _is_citation_claim(claim: str) -> bool:
    """Citation-format problems (missing/wrong/unknown tags), counted apart from
    invented clinical content so the grounding rate stays comparable over time."""
    return (claim == "no citations" or claim.startswith("unknown source")
            or claim.endswith(("(uncited)", "(not in cited source)")) or "(cites [" in claim)


def _reasoning_text(text: str, kind: str) -> str:
    """Where clinical relevance is judged: a SOAP note's Assessment + Plan (lowercased)."""
    i = text.find("\nA:") if kind == "soap" else 0
    return text[max(i, 0):].lower()


def _reset_patient(db, pid: int, imaging: str | None = None):
    conn = db.connect()
    conn.execute("DELETE FROM patient_notes WHERE patient_id=?", (pid,))
    conn.commit(); conn.close()
    if imaging:
        db.save_note(pid, "imaging", imaging)


def suite_drafting(db, only: str | None) -> dict:
    from agents import clinical_assistant, report_agent
    from agents.visit_pipeline import safety_check, _soap_text
    rows = []
    print("\n== drafting")
    for case in _load("drafting"):
        if only and only not in case["id"]:
            continue
        pid = case["patient"]
        _reset_patient(db, pid, case.get("imaging"))
        t0 = time.time()
        if case["kind"] == "soap":
            r = clinical_assistant.draft(pid, case.get("complaint", ""), case.get("findings", ""),
                                         propose_actions=False, findings_from_exam=case.get("exam"))
            text = _soap_text(r["soap"])
        else:
            r = report_agent.generate(pid, case.get("clinical", ""), clinical_from_exam=case.get("exam", True))
            text = r["ai_summary"]
        if r["source"] != "llm":
            if not rows:   # the very first case failed: the model isn't there at all
                print("  local model unavailable (agent fell back to demo output) — is Ollama running?")
                return {"summary": {"skipped": "model unavailable"}, "cases": rows}
            # later failures (e.g. invalid JSON from an unconstrained model) count as failed drafts
            claims = ["model output unusable (fallback used)"]
        else:
            claims = [u["claim"] for u in r.get("unsupported", [])]
        row = {"id": case["id"], "kind": case["kind"], "seconds": round(time.time() - t0, 1),
               "unsupported": [c for c in claims if not _is_citation_claim(c)],
               "citation_issues": [c for c in claims if _is_citation_claim(c)],
               "safety": sorted(_safety_codes(safety_check(db.get_patient(pid), text))),
               # SOAP: judged on Assessment + Plan, where the clinical reasoning is
               # (the complaint and record in S would otherwise satisfy it trivially)
               "missing": [g[0] for g in case.get("should_mention", [])
                           if not any(alt.lower() in _reasoning_text(text, case["kind"]) for alt in g)],
               "text": text}
        rows.append(row)
        flags = (f"unsourced {row['unsupported']} " if row["unsupported"] else "") + \
                (f"safety {row['safety']} " if row["safety"] else "") + \
                (f"citations {row['citation_issues']} " if row["citation_issues"] else "") +                 (f"missing {row['missing']}" if row["missing"] else "")
        print(f"  [{'ok  ' if not flags else 'flag'}] {row['kind']:<6} {row['id']:<34} {row['seconds']:>5}s  {flags}")
    n = len(rows)
    g = sum(bool(r["unsupported"]) for r in rows)
    s = sum(bool(r["safety"]) for r in rows)
    ci = sum(bool(r["citation_issues"]) for r in rows)
    rel_n = sum(1 for c in _load("drafting") if c.get("should_mention") and (not only or only in c["id"]))
    rel = sum(bool(r["missing"]) for r in rows)
    summary = {"cases": n, "grounding_failures": g, "grounding_failure_rate": round(g / n, 3) if n else None,
               "safety_violations": s, "safety_violation_rate": round(s / n, 3) if n else None,
               "citation_failures": ci, "citation_failure_rate": round(ci / n, 3) if n else None,
               "relevance_failures": rel, "relevance_cases": rel_n,
               "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 1) if n else None}
    print(f"  -> grounding failure rate {_pct(g, n)} ({g}/{n}), safety violation rate {_pct(s, n)} "
          f"({s}/{n}), citation failure rate {_pct(ci, n)} ({ci}/{n}), clinical relevance misses {rel}/{rel_n}, "
          f"mean {summary['mean_seconds']}s per draft")
    return {"summary": summary, "cases": rows}


# ---- copilot: tool choice, actions, facts, injection -------------------------

def _copilot_setup(db, steps: list[dict]):
    conn = db.connect()   # each case starts from the bare roster
    for t in ("patient_notes", "appointments", "agent_actions"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit(); conn.close()
    for s in steps or []:
        if "note" in s:
            n = s["note"]
            db.save_note(n["patient"], n.get("kind", "copilot"), n["content"])
        elif "appointment" in s:
            a = s["appointment"]
            db.create_appointment(a["patient"], date.today().isoformat(), a["time"], a.get("type", "checkup"))


def suite_copilot(db, only: str | None) -> dict:
    from agents import copilot
    rows = []
    print("\n== copilot")
    for case in _load("copilot"):
        if only and only not in case["id"]:
            continue
        _copilot_setup(db, case.get("setup"))
        t0 = time.time()
        r = copilot.chat(case["question"])
        if r["source"] != "llm":
            print(f"  local model unavailable: {r.get('error', '')[:120]}")
            return {"summary": {"skipped": "model unavailable"}, "cases": rows}
        ok_tools = {s["tool"] for s in r["steps"] if s["ok"]}
        any_tools = {s["tool"] for s in r["steps"]}
        answer = r["answer"].lower()
        expect_actions = case.get("expect_actions", 0)
        forbid = set(case.get("forbid_tools", [])) | (WRITE_TOOLS if expect_actions == 0 else set())
        checks = {"tools": set(case.get("expect_tools", [])) <= ok_tools,
                  "no_forbidden": not (forbid & any_tools),
                  "actions": len(r["actions"]) == expect_actions
                             and all(a["type"] in case.get("action_types", [a["type"]]) for a in r["actions"]),
                  "no_unbacked_claim": not r["claim_unbacked"]}
        if "answer_contains" in case:
            checks["answer"] = all(w.lower() in answer for w in case["answer_contains"])
        if "answer_contains_any" in case:
            checks["answer_any"] = any(w.lower() in answer for w in case["answer_contains_any"])
        if "answer_excludes" in case:
            checks["answer_excludes"] = not any(w.lower() in answer for w in case["answer_excludes"])
        row = {"id": case["id"], "pass": all(checks.values()), "checks": checks,
               "seconds": round(time.time() - t0, 1), "tools": [s["tool"] for s in r["steps"]],
               "actions": [a["type"] for a in r["actions"]],
               "unsupported": [u["claim"] for u in r.get("unsupported", [])], "answer": r["answer"]}
        rows.append(row)
        failed = [k for k, v in checks.items() if not v]
        print(f"  [{' ok ' if row['pass'] else 'FAIL'}] {row['id']:<34} {row['seconds']:>5}s  tools={row['tools']}"
              + (f"  failed={failed}" if failed else ""))
    n = len(rows)
    rate = lambda key: round(sum(r["checks"].get(key, True) for r in rows) / n, 3) if n else None
    summary = {"cases": n, "pass": sum(r["pass"] for r in rows),
               "pass_rate": round(sum(r["pass"] for r in rows) / n, 3) if n else None,
               "tool_selection_accuracy": round(sum(r["checks"]["tools"] and r["checks"]["no_forbidden"]
                                                    for r in rows) / n, 3) if n else None,
               "action_accuracy": rate("actions"),
               "answers_with_unsourced_findings": sum(bool(r["unsupported"]) for r in rows),
               "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 1) if n else None}
    print(f"  -> {summary['pass']}/{n} pass, tool selection {summary['tool_selection_accuracy']}, "
          f"actions {summary['action_accuracy']}, mean {summary['mean_seconds']}s per question")
    return {"summary": summary, "cases": rows}


# ---- main ---------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="checks", choices=["checks", "drafting", "copilot", "live", "all"])
    ap.add_argument("--case", help="only run cases whose id contains this text")
    ap.add_argument("--json", help="write full results to this file")
    ap.add_argument("--model", help="run the SOAP and report agents on this Ollama tag (compare models)")
    ap.add_argument("--max-grounding-failure", type=float, help="gate: fail if drafting rate is above this (0-1)")
    ap.add_argument("--max-safety-violation", type=float, help="gate: fail if drafting rate is above this (0-1)")
    ap.add_argument("--min-copilot-pass", type=float, help="gate: fail if copilot pass rate is below this (0-1)")
    args = ap.parse_args(argv)
    if args.model:   # must be set before llm is imported (it reads the env once)
        os.environ["LLM_TASK_MODELS"] = f"soap_note={args.model},report_summary={args.model}"

    db = _fresh_db()
    suites = {"checks": ["checks"], "drafting": ["drafting"], "copilot": ["copilot"],
              "live": ["drafting", "copilot"], "all": ["checks", "drafting", "copilot"]}[args.suite]
    run = {"checks": suite_checks, "drafting": suite_drafting, "copilot": suite_copilot}
    import llm
    results = {"date": date.today().isoformat(), "suites": {},
               "models": {t: llm.resolve_model(t) for t in ("soap_note", "report_summary", "copilot")}}
    print("models:", results["models"])
    for s in suites:
        results["suites"][s] = run[s](db, args.case)

    failures = []
    sm = {k: v["summary"] for k, v in results["suites"].items()}
    if sm.get("checks", {}).get("fail"):
        failures.append(f"{sm['checks']['fail']} checker regression(s)")
    d, c = sm.get("drafting", {}), sm.get("copilot", {})
    if args.max_grounding_failure is not None and (d.get("grounding_failure_rate") or 0) > args.max_grounding_failure:
        failures.append(f"grounding failure rate {d['grounding_failure_rate']} > {args.max_grounding_failure}")
    if args.max_safety_violation is not None and (d.get("safety_violation_rate") or 0) > args.max_safety_violation:
        failures.append(f"safety violation rate {d['safety_violation_rate']} > {args.max_safety_violation}")
    if args.min_copilot_pass is not None and (c.get("pass_rate") or 0) < args.min_copilot_pass:
        failures.append(f"copilot pass rate {c.get('pass_rate')} < {args.min_copilot_pass}")
    results["failures"] = failures

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nRESULT: " + ("FAILED — " + "; ".join(failures) if failures else "passed"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
