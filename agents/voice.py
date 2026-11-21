"""
Voice-to-SOAP (roadmap item 4) — local speech-to-text, then the normal SOAP agent.

A clinician dictates ("Patient complains of pain lower right molar. On exam,
tender to percussion on 46, no swelling."). faster-whisper transcribes it on this
machine — audio never leaves the clinic — and the transcript goes to the
Clinical Assistant as the clinician's own dictated findings, so the note is
drafted, grounded, safety-checked and cited exactly like a typed one.

Model: Whisper "small" (WHISPER_MODEL to change) on the CPU in int8 — no GPU
libraries needed. Loaded once and kept warm (warm_up() at app start). For speed,
the live flow (agents/voice_stream.py) transcribes each utterance while the next
is being spoken, with greedy decoding; this file's transcribe() is the
upload-a-recording path.
"""
from __future__ import annotations

import os
import re

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
# Biases Whisper toward dental terms: without it "caries" came out as "carries".
DENTAL_PROMPT = ("Dental clinical dictation. Caries, gingivitis, periodontitis, pulpitis, pericoronitis, "
                 "calculus, plaque, abscess, percussion, mobility, pocketing, tooth #46, #38, mm.")
_model = None
_device = None
_lock = __import__("threading").Lock()


class VoiceUnavailable(Exception):
    pass


def _load():
    """CPU-only Whisper, int8, all cores. Thread-safe: loaded once and shared."""
    global _model, _device
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise VoiceUnavailable("install faster-whisper") from e
            _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8",
                                  cpu_threads=os.cpu_count() or 4)
            _device = "cpu"
    return _model


def warm_up():
    """Load Whisper + Silero and run one tiny transcription, so the first real
    dictation doesn't pay the ~2-5 s model load. Called in a thread at app start."""
    import numpy as np
    from faster_whisper.vad import get_vad_model
    get_vad_model()
    transcribe_array(np.zeros(16000, dtype=np.float32))


def transcribe_array(audio, previous: str = "") -> str:
    """One utterance (16 kHz float32, already cut by the VAD) -> text. Greedy decoding
    and no second VAD pass keep it fast on the CPU; the dental prompt plus the tail of
    what was already said keeps terms and context consistent across utterances."""
    model = _load()
    segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False,
                                   condition_on_previous_text=False, without_timestamps=True,
                                   initial_prompt=(DENTAL_PROMPT + " " + previous).strip())
    return " ".join(seg.text.strip() for seg in segments).strip()


def _decode(path: str, rate: int = 16000):
    """Any audio/video file (wav, webm/opus from the browser, m4a ...) -> mono float32
    at 16 kHz. Done here rather than by faster-whisper, whose decoder passes an
    argument newer PyAV (>=15) no longer accepts."""
    import av
    import numpy as np
    chunks = []
    with av.open(path) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    if not chunks:
        raise ValueError("the recording contains no audio")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def transcribe(audio_path: str) -> dict:
    """Transcribe one recording (upload path). Returns {text, language, seconds, device}."""
    audio = _decode(audio_path)
    segments, info = _load().transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                        without_timestamps=True, initial_prompt=DENTAL_PROMPT)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {"text": fix_dental_terms(text), "language": info.language,
            "seconds": round(info.duration, 1), "device": _device}


# Homophones Whisper gets wrong in dental dictation, corrected only in a dental
# context (a surface/severity word before, or a tooth number after) so the plain
# verb "carries" elsewhere is left alone.
_CARIES = re.compile(r"\b(?:(?<=deep )|(?<=early )|(?<=advanced )|(?<=distal )|(?<=mesial )|(?<=occlusal )|"
                     r"(?<=buccal )|(?<=lingual )|(?<=recurrent )|(?<=interproximal )|(?<=root )|(?<=active ))"
                     r"carries\b|\bcarries(?=\s+(?:on|at|in|of)?\s*(?:tooth\s*)?#?\d)", re.IGNORECASE)


def fix_dental_terms(text: str) -> str:
    return _CARIES.sub("caries", text or "")


# Dictation usually starts with what the patient says, then the exam. Split there
# so the complaint and the clinician's findings go to the right SOAP sources.
_EXAM_START = re.compile(r"\b(on (?:clinical )?exam(?:ination)?|exam(?:ination)? (?:shows|reveals|finds)|"
                         r"findings?:|objectively|clinically)\b", re.IGNORECASE)


def split_dictation(text: str) -> tuple[str, str]:
    """(complaint, findings). Without an exam marker everything is treated as findings."""
    m = _EXAM_START.search(text or "")
    if not m:
        return "", (text or "").strip()
    return text[:m.start()].strip(" .,;"), text[m.start():].strip()


def voice_soap(audio_path: str, patient_id: int) -> dict:
    """Transcribe a dictation and draft the SOAP note from it."""
    from agents import clinical_assistant
    t = transcribe(audio_path)
    if not t["text"]:
        raise ValueError("no speech was recognised in the recording")
    complaint, findings = split_dictation(t["text"])
    result = clinical_assistant.draft(patient_id, complaint, findings, findings_from_exam=True)
    result["transcript"] = t
    result["complaint_from_dictation"] = complaint
    result["findings_from_dictation"] = findings
    return result
