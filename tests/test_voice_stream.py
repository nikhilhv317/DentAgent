"""Live dictation WebSocket: auth, role and clinic checks, and the stream protocol.
Whisper and Silero are stubbed here (no model downloads in CI; Silero rightly
ignores synthetic tones): the VAD stub calls loud frames speech. The real path
was verified end to end with recorded speech (see README, Voice-to-SOAP)."""
import json

import numpy as np
import pytest

import db


def _token(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password}).json()["token"]


def test_rejects_bad_token(client):
    with client.websocket_connect("/ws/dictation") as ws:
        ws.send_text(json.dumps({"type": "auth", "token": "nope", "patient_id": 1}))
        assert ws.receive_json()["type"] == "error"


def test_front_desk_cannot_dictate(client, staff):
    tok = _token(client, "desk", "desk-test-pw")
    with client.websocket_connect("/ws/dictation") as ws:
        ws.send_text(json.dumps({"type": "auth", "token": tok, "patient_id": 1}))
        assert "clinician" in ws.receive_json()["detail"]


def test_stream_protocol(client, clinician, monkeypatch):
    from agents import voice, voice_stream
    monkeypatch.setattr(voice_stream, "_vad_probs",
                        lambda a: np.array([1.0 if np.abs(a[-512:]).mean() > 0.05 else 0.0]))
    calls = []
    monkeypatch.setattr(voice, "transcribe_array", lambda audio, previous="": calls.append(len(audio)) or "Tender on percussion #46.")
    import agents
    monkeypatch.setattr(agents, "run", lambda key, **kw: {"soap": {"subjective": "", "objective": "", "assessment": "",
                                                                   "plan": []}, "kw": kw, "unsupported": []})
    tok = _token(client, "drsmile", "clin-test-pw")
    rate = 16000
    t = np.arange(int(1.5 * rate)) / rate
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)   # 'speech' for the energy stub
    pcm = (np.concatenate([np.zeros(rate // 2), tone, np.zeros(rate)]) * 32767).astype("<i2").tobytes()
    with client.websocket_connect("/ws/dictation") as ws:
        ws.send_text(json.dumps({"type": "auth", "token": tok, "patient_id": 1}))
        assert ws.receive_json()["type"] == "ready"
        for i in range(0, len(pcm), 3200):
            ws.send_bytes(pcm[i:i + 3200])
        ws.send_text(json.dumps({"type": "stop"}))
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] in ("note", "error"):
                break
    kinds = [m["type"] for m in msgs]
    assert "transcript" in kinds and kinds[-1] == "note", kinds
    final = next(m for m in msgs if m["type"] == "transcript")
    assert final["transcript"].startswith("Tender on percussion #46.") and calls
    assert msgs[-1]["note"]["kw"]["patient_id"] == 1
