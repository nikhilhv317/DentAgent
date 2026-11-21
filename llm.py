"""
LLM layer  (local, via Ollama).

The ONLY place that talks to a model. Inference runs on a local Ollama server,
so patient data never leaves the machine — which is the point for a clinical
system. No API key, no cloud call.

`complete_json()` asks the model for JSON and parses it. When a `schema` is
given it is passed to Ollama's `format` parameter, which constrains generation
at the token level so the reply is guaranteed to match the shape — the thing
small local models otherwise get wrong. If Ollama isn't running or the model
isn't pulled, it raises LLMUnavailable so the caller can fall back to a
deterministic demo result, and the whole app still runs with nothing installed.

--- Model routing ---------------------------------------------------------------
Every call site names the TASK it's running (report_summary, soap_note, ...).
TASK_MODELS maps each task to the model tier that should serve it. This is real
routing infrastructure, not a single hardcoded model: moving one task to a
stronger model later (e.g. a SOAP note that needs more reasoning) is a one-line
change to TASK_MODELS, with zero changes to the calling agent.

v1 deployment (this AWS instance): every task is pinned to the SAME small
model — one model resident in Ollama, lower/predictable memory footprint,
one thing to pull and tune. The routing table exists so that changes when
there's a reason to, not by accident (see git history: this app once had
qwen3:4b, qwen3:8b, and qwen3:1.7b all referenced in different files at once —
routing through one table is what stops that from happening again).

Setup:
    # install Ollama from https://ollama.com, then:
    ollama pull qwen3:1.7b
    ollama serve            # usually already running as a service
"""
import os
import json

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")  # tiny + fast; qwen3 family so think=False works
HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Model tiers. Tasks are routed to a tier, and the tier to a tag, so swapping the
# model behind a tier (or moving a task between tiers after an eval) is one line.
TIERS: dict[str, str] = {
    "fast":   DEFAULT_MODEL,                                  # qwen3:1.7b, ~3 s/draft on the GPU
    "strong": os.getenv("STRONG_MODEL", "qwen2.5:3b"),        # bigger, already pulled locally
    # fine-tuned model (ml/lora/). "hf:<dir>" runs it with transformers on this
    # machine — Ollama >= 0.40 can't import safetensors on Windows (no MLX) and a
    # GGUF conversion needs llama.cpp. Point LORA_MODEL at an Ollama tag instead
    # once it's been converted.
    "lora":   os.getenv("LORA_MODEL", "hf:" + os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           "ml", "lora", "out", "merged")),
}

# The fine-tuned model wins every drafting metric (evals/README.md: 0% invented
# content / safety / citation failures, 0/8 relevance misses vs 19% / 6% / 75% /
# 1/8 for the base) — but it's a 3.4 GB local build, not in git. Route to it only
# when it's actually there; a fresh clone or the Docker image uses the base model.
_LORA_READY = (TIERS["lora"].startswith("hf:") and os.path.isdir(TIERS["lora"][3:])) or \
              (not TIERS["lora"].startswith("hf:") and bool(os.getenv("LORA_MODEL")))
_DRAFTING_TIER = "lora" if _LORA_READY else "fast"

# task name -> tier, set from measurements (evals/README.md "How the numbers moved").
TASK_TIERS: dict[str, str] = {
    "report_summary":       _DRAFTING_TIER,   # report_agent.py — 3-4 line summary
    "patient_brief":        "fast",   # patient_intelligence.py — clinical brief (not trained on)
    "soap_note":            _DRAFTING_TIER,   # clinical_assistant.py — SOAP note
    "scheduling_reminder":  "fast",   # scheduling_agent.py — SMS reminder text
}

# task name -> model tag. MODEL_<TASK> env vars override one task; LLM_TASK_MODELS
# ("soap_note=qwen2.5:3b,report_summary=qwen2.5:3b") overrides several — that's
# how `evals/run.py --model` compares models without code changes.
TASK_MODELS: dict[str, str] = {
    **{t: os.getenv(f"MODEL_{t.upper()}", TIERS[tier]) for t, tier in TASK_TIERS.items()},
    # copilot.py — multi-step tool calling. Same 1.7B model, but run with
    # thinking ON (see copilot._model): with thinking off it describes tool
    # calls in prose instead of making them. Measured on this CPU box, a
    # two-round question takes ~35s. The local "qwen3:4b" tag is the 2507
    # thinking-only build: it can't turn thinking off, Ollama leaks the
    # reasoning into the answer, and it ran ~4-5 min per question. If you want
    # a bigger model here, use a non-thinking build such as qwen3:4b-instruct-2507.
    "copilot":              os.getenv("COPILOT_MODEL", DEFAULT_MODEL),
}
for _pair in filter(None, os.getenv("LLM_TASK_MODELS", "").split(",")):
    _task, _, _tag = _pair.partition("=")
    if _task.strip() and _tag.strip():
        TASK_MODELS[_task.strip()] = _tag.strip()


def resolve_model(task: str | None) -> str:
    """The one place a task name becomes an actual model tag."""
    if task and task in TASK_MODELS:
        return TASK_MODELS[task]
    return DEFAULT_MODEL


class LLMUnavailable(Exception):
    pass


def _client():
    try:
        import ollama
    except ImportError as e:
        raise LLMUnavailable("ollama package not installed (pip install ollama)") from e
    return ollama.Client(host=HOST)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def complete_json(system: str, user: str, schema: dict | None = None,
                  task: str | None = None) -> dict:
    """
    Run one local completion and parse a JSON object out of it.

    schema: a JSON Schema dict (e.g. SomeModel.model_json_schema()). When given,
    Ollama constrains the output to match it. When None, plain JSON mode is used.
    task: which TASK_MODELS entry routes this call (see module docstring). Omit
    to use DEFAULT_MODEL directly.
    """
    from tracing import span
    model = resolve_model(task)
    with span("llm.complete_json", model=model, task=task or "") as s:
        result = _complete_json(model, system, user, schema, s)
    return result


_hf_models: dict = {}
_hf_lock = None


def _hf_complete(path: str, system: str, user: str, s) -> dict:
    """Local transformers backend for a fine-tuned model directory. The prompt is
    rendered exactly as in training (ml/lora/train.py): chat template with
    enable_thinking=False and " /no_think" on the user turn. Greedy decoding."""
    global _hf_lock
    import threading
    if _hf_lock is None:
        _hf_lock = threading.Lock()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise LLMUnavailable("install torch + transformers for hf: models") from e
    if not os.path.isdir(path):
        raise LLMUnavailable(f"fine-tuned model not found at {path} (see ml/lora/)")
    with _hf_lock:
        if path not in _hf_models:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(path)
            m = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map={"": device})
            _hf_models[path] = (tok, m.eval())
        tok, m = _hf_models[path]
        prompt = tok.apply_chat_template([{"role": "system", "content": system},
                                          {"role": "user", "content": user + " /no_think"}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(m.device)
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=900, do_sample=False, pad_token_id=tok.eos_token_id)
        new = out[0, ids["input_ids"].shape[1]:]
    s.set_attribute("prompt_eval_count", int(ids["input_ids"].shape[1]))
    s.set_attribute("eval_count", int(new.shape[0]))
    text = tok.decode(new, skip_special_tokens=True)
    try:
        return json.loads(_strip_fences(text))
    except json.JSONDecodeError as e:   # no constrained decoding here: a broken answer -> fallback
        raise LLMUnavailable(f"fine-tuned model returned invalid JSON: {e}") from e


def warm_up(tasks=("soap_note", "report_summary")):
    """Load the drafting model(s) before the first request: the fine-tuned model is a
    3.4 GB load (~10 s) that otherwise lands on the first dictation's SOAP draft;
    an Ollama model gets loaded into memory and kept there (keep_alive)."""
    for model in {resolve_model(t) for t in tasks}:
        try:
            if model.startswith("hf:"):
                class _NoSpan:
                    def set_attribute(self, *a): pass
                try:
                    _hf_complete(model[3:], "Reply with JSON.", 'Return {"ok": true}', _NoSpan())
                except LLMUnavailable:
                    pass          # loaded fine; a non-JSON reply to the dummy prompt doesn't matter
            else:
                _client().generate(model=model, prompt="", keep_alive="30m")
        except Exception:
            pass                  # warm-up is best effort; the real call reports errors


def _complete_json(model: str, system: str, user: str, schema: dict | None, s) -> dict:
    if model.startswith("hf:"):
        return _hf_complete(model[3:], system, user, s)
    client = _client()
    fmt = schema if schema else "json"
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        try:
            resp = client.chat(
                model=model, messages=msgs, format=fmt,
                think=False,            # skip Qwen3's slow "thinking" pass — big speedup
                keep_alive="30m",       # keep the model loaded between calls
                options={"temperature": 0},
            )
        except TypeError:
            # older ollama client without think= support
            resp = client.chat(model=model, messages=msgs, format=fmt,
                               keep_alive="30m", options={"temperature": 0})
        except Exception as e:
            if "think" not in str(e).lower():
                raise
            # a non-thinking model (e.g. the qwen2.5 "strong" tier) rejects think=
            resp = client.chat(model=model, messages=msgs, format=fmt,
                               keep_alive="30m", options={"temperature": 0})
        content = resp["message"]["content"]
        for k in ("prompt_eval_count", "eval_count"):   # tokens in / out, for cost & latency tracking
            if resp.get(k) is not None:
                s.set_attribute(k, int(resp[k]))
        return json.loads(_strip_fences(content))
    except LLMUnavailable:
        raise
    except Exception as e:  # connection refused, model missing, parse error -> demo
        raise LLMUnavailable(str(e)) from e
