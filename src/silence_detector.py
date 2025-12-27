"""Energy-based silence detector."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from audio_processor import AudioChunk


@dataclass
class SilenceAnalysis:
    is_silent: bool
    energy_db: float
    consecutive_silence_sec: float
    confidence: float


class SilenceDetector:
    def __init__(self, threshold_db: float = -40.0, min_silence_sec: float = 1.0, ref_rms: float = 1.0):
        self.threshold_db = float(threshold_db)
        self.min_silence_sec = float(min_silence_sec)
        self.ref_rms = float(ref_rms)
        self._consecutive_silence_sec: float = 0.0

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))

    def _to_db(self, rms: float) -> float:
        if rms <= 0:
            return -120.0
        return 20.0 * np.log10(rms / self.ref_rms)

    def reset(self) -> None:
        self._consecutive_silence_sec = 0.0

    def process_chunk(self, chunk: AudioChunk) -> SilenceAnalysis:
        rms = self._rms(chunk.samples)
        energy_db = self._to_db(rms)
        is_silent = energy_db < self.threshold_db

        if is_silent:
            self._consecutive_silence_sec += chunk.duration
        else:
            self._consecutive_silence_sec = 0.0

        if energy_db < self.threshold_db:
            diff = min(abs(energy_db - self.threshold_db), 40.0)
            confidence = diff / 40.0
        else:
            confidence = 0.0

        return SilenceAnalysis(
            is_silent=is_silent,
            energy_db=energy_db,
            consecutive_silence_sec=self._consecutive_silence_sec,
            confidence=confidence,
        )

    def is_definitely_silent(self) -> bool:
        return self._consecutive_silence_sec >= self.min_silence_sec
