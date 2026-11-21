"""
Live dictation: Silero VAD + Whisper on the CPU, transcribing while you speak.

The upload flow (agents/voice.py) waits until the clinician stops, then runs
Whisper over the whole recording — on a CPU that's ~14 s for 17 s of speech, all
of it felt as waiting. Here the browser streams the microphone (16 kHz mono PCM)
over a WebSocket, Silero VAD finds the end of each utterance (a short pause), and
that utterance is transcribed immediately on a background thread while the next
one is being spoken. When the clinician presses stop, only the last utterance is
left to transcribe, then the SOAP note is drafted.

    session = DictationSession(on_event)      # on_event(dict) gets live updates
    session.feed(pcm_bytes)                   # 16 kHz, mono, int16 little-endian
    result = session.finish(patient_id)       # final transcript + SOAP draft

Events: {"type": "speech", "speaking": bool}
        {"type": "partial", "segment": n, "text": "...", "seconds": s, "transcript": "..."}
Everything runs on the CPU; Silero is the ONNX model bundled with faster-whisper.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

RATE = 16000
FRAME = 512                       # Silero's window at 16 kHz (32 ms)
START_PROB, END_PROB = 0.5, 0.35  # hysteresis: speech starts above / ends below
END_SILENCE_S = 0.6               # pause that ends an utterance
PRE_ROLL_S = 0.3                  # audio kept before speech onset (don't clip the first word)
MAX_UTTERANCE_S = 12.0            # force a cut in very long run-on speech
VAD_WINDOW_S = 1.0                # Silero is re-run on this much recent audio each step


def _vad_probs(audio: np.ndarray) -> np.ndarray:
    from faster_whisper.vad import get_vad_model
    n = len(audio) - len(audio) % FRAME
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(get_vad_model()(audio[-n:], num_samples=FRAME)).reshape(-1)


class DictationSession:
    def __init__(self, on_event=None):
        self.on_event = on_event or (lambda e: None)
        self.buf = np.zeros(0, dtype=np.float32)   # audio not yet assigned to a finished utterance
        self.speaking = False
        self.speech_start = 0                      # index into buf
        self.silence = 0                           # samples of trailing silence while speaking
        self.processed = 0                         # samples of buf already judged by VAD
        self.texts: list[str] = []
        self.jobs: queue.Queue = queue.Queue()
        self.done = threading.Event()
        self.total_audio_s = 0.0
        self.worker = threading.Thread(target=self._transcribe_loop, daemon=True)
        self.worker.start()

    # ---- audio in ---------------------------------------------------------------
    def feed(self, pcm: bytes):
        chunk = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.total_audio_s += len(chunk) / RATE
        self.buf = np.concatenate([self.buf, chunk])
        # judge new audio a frame at a time, using a 1 s window for context
        while len(self.buf) - self.processed >= FRAME:
            end = self.processed + FRAME
            window = self.buf[max(0, end - int(VAD_WINDOW_S * RATE)):end]
            p = float(_vad_probs(window)[-1]) if len(window) >= FRAME else 0.0
            self.processed = end
            self._step(p, end)

    def _step(self, prob: float, end: int):
        if not self.speaking:
            if prob >= START_PROB:
                self.speaking = True
                self.speech_start = max(0, end - FRAME - int(PRE_ROLL_S * RATE))
                self.silence = 0
                self.on_event({"type": "speech", "speaking": True})
            else:   # keep only the pre-roll while idle, so memory stays flat
                keep = int(PRE_ROLL_S * RATE) + FRAME
                if len(self.buf) > 4 * keep:
                    drop = len(self.buf) - keep
                    self.buf, self.processed = self.buf[drop:], self.processed - drop
            return
        self.silence = self.silence + FRAME if prob < END_PROB else 0
        too_long = end - self.speech_start > MAX_UTTERANCE_S * RATE
        if self.silence >= END_SILENCE_S * RATE or too_long:
            cut = end if too_long else end - self.silence + int(0.2 * RATE)
            self._emit_utterance(self.speech_start, cut)
            self.buf, self.processed = self.buf[cut:], self.processed - cut
            self.speaking, self.silence = False, 0
            self.on_event({"type": "speech", "speaking": False})

    def _emit_utterance(self, start: int, end: int):
        audio = self.buf[start:end]
        if len(audio) > 0.25 * RATE:            # ignore clicks / breaths
            self.jobs.put(audio.copy())

    # ---- transcription (background thread, CPU) -------------------------------------
    def _transcribe_loop(self):
        from agents import voice
        while True:
            audio = self.jobs.get()
            if audio is None:
                self.done.set()
                return
            t0 = time.time()
            text = voice.fix_dental_terms(voice.transcribe_array(audio, previous=" ".join(self.texts)[-200:]))
            if text:
                self.texts.append(text)
                self.on_event({"type": "partial", "segment": len(self.texts), "text": text,
                               "seconds": round(time.time() - t0, 2), "transcript": self.transcript()})

    def transcript(self) -> str:
        from agents import voice
        return voice.fix_dental_terms(" ".join(self.texts).strip())

    # ---- stop -----------------------------------------------------------------------
    def finish(self, patient_id: int | None = None, timeout: float = 120) -> dict:
        """Flush the last utterance, wait for transcription, then draft the SOAP note."""
        t0 = time.time()
        # only an utterance still in progress needs transcribing: audio the VAD has
        # judged silent is never sent to Whisper (it costs ~1.3 s on the CPU, and
        # Whisper can hallucinate words from silence)
        if self.speaking:
            self._emit_utterance(self.speech_start, len(self.buf))
        self.jobs.put(None)
        self.done.wait(timeout)
        text = self.transcript()
        out = {"transcript": text, "audio_seconds": round(self.total_audio_s, 1),
               "tail_transcribe_seconds": round(time.time() - t0, 2)}
        if patient_id is not None:
            if not text:
                raise ValueError("no speech was recognised in the recording")
            from agents import clinical_assistant, voice
            complaint, findings = voice.split_dictation(text)
            t1 = time.time()
            out["note"] = clinical_assistant.draft(patient_id, complaint, findings, findings_from_exam=True)
            out["draft_seconds"] = round(time.time() - t1, 2)
            out["complaint_from_dictation"], out["findings_from_dictation"] = complaint, findings
        return out
