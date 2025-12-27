"""Orchestrator combining all detectors."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from audio_processor import AudioStreamer
from silence_detector import SilenceDetector
from beep_detector import BeepDetector
from stt_processor import SpeechToTextProcessor, STTState
from results_aggregator import CallDecision


@dataclass
class VoicemailDecision:
    decision_time_sec: float
    confidence: float
    reason: str
    beep_detected: bool
    total_duration_sec: float
    transcript: str


class VoicemailAnalyzer:
    def __init__(
        self,
        use_stt: bool = False,
        use_llm: bool = False,
        chunk_size_ms: int = 250,
        silence_threshold_db: float = -40.0,
        silence_short_sec: float = 1.0,
        silence_long_sec: float = 1.5,
        timeout_sec: float = 25.0,
        llm_min_confidence: float = 0.75,
    ):
        self.streamer = AudioStreamer(chunk_size_ms=chunk_size_ms)
        self.silence = SilenceDetector(
            threshold_db=silence_threshold_db,
            min_silence_sec=silence_short_sec,
        )
        self.beep = BeepDetector()
        self.use_stt = use_stt
        self.silence_short_sec = silence_short_sec
        self.silence_long_sec = silence_long_sec
        self.timeout_sec = timeout_sec

        self.stt: Optional[SpeechToTextProcessor] = SpeechToTextProcessor() if use_stt else None

    def analyze_file(self, path: str) -> VoicemailDecision:
        self.streamer.load_file(path)
        elapsed = 0.0

        for chunk in self.streamer.stream():
            elapsed = chunk.timestamp + chunk.duration

            sa = self.silence.process_chunk(chunk)
            ba = self.beep.process_chunk(chunk)

            if self.stt is not None:
                self.stt.add_chunk(chunk)

            # PRIMARY: beep detection (STRICTER confidence threshold)
            if ba.beep_detected and ba.confidence >= 0.8:  # WAS 0.6, NOW 0.8
                return self._build_decision(elapsed, 1.0, "Beep detected", True)

            # TERTIARY: long silence (fallback)
            if sa.consecutive_silence_sec >= self.silence_long_sec:
                conf = min(0.6, sa.confidence + 0.2)
                return self._build_decision(elapsed, conf, "Extended silence", False)

            # Timeout
            if elapsed >= self.timeout_sec:
                return self._build_decision(elapsed, 0.45, "Timeout", False)

        return self._build_decision(self.streamer.duration, 0.3, "End of audio", False)


    def _build_decision(self, t: float, conf: float, reason: str, beep: bool) -> VoicemailDecision:
        transcript = ""
        if self.stt is not None:
            stt_state = self.stt.finalize()
            transcript = stt_state.transcript

        return VoicemailDecision(
            decision_time_sec=t,
            confidence=conf,
            reason=reason,
            beep_detected=beep,
            total_duration_sec=self.streamer.duration,
            transcript=transcript,
        )

    @staticmethod
    def to_call_decision(file_path: str, v: VoicemailDecision) -> CallDecision:
        return CallDecision(
            file_path=file_path,
            decision_time_sec=v.decision_time_sec,
            total_duration_sec=v.total_duration_sec,
            confidence=v.confidence,
            reason=v.reason,
            beep_detected=v.beep_detected,
        )
