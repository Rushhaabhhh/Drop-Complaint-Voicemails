"""Audio loading and chunked streaming utilities."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Generator
import numpy as np
import soundfile as sf


@dataclass
class AudioChunk:
    """Single chunk of mono audio."""
    samples: np.ndarray
    sample_rate: int
    timestamp: float
    duration: float
    index: int


class AudioStreamer:
    def __init__(self, chunk_size_ms: int = 250, target_sample_rate: int = 16000):
        self.chunk_size_ms = int(chunk_size_ms)
        self.target_sample_rate = int(target_sample_rate)
        self._audio = None
        self._sample_rate = None

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise RuntimeError("No audio loaded")
        return self._sample_rate

    @property
    def duration(self) -> float:
        if self._audio is None or self._sample_rate is None:
            return 0.0
        return len(self._audio) / float(self._sample_rate)

    def load_file(self, path: str | Path) -> None:
        path = Path(path)
        data, sr = sf.read(path, always_2d=True)
        mono = data.mean(axis=1).astype(np.float32)

        if sr != self.target_sample_rate:
            import librosa
            mono = librosa.resample(mono, orig_sr=sr, target_sr=self.target_sample_rate)
            self._sample_rate = self.target_sample_rate
        else:
            self._sample_rate = sr

        self._audio = mono

    def stream(self) -> Generator[AudioChunk, None, None]:
        if self._audio is None or self._sample_rate is None:
            raise RuntimeError("Audio not loaded")

        hop = int(self.sample_rate * self.chunk_size_ms / 1000.0)
        total = len(self._audio)
        idx = 0
        chunk_index = 0

        while idx < total:
            end = min(idx + hop, total)
            samples = self._audio[idx:end]
            timestamp = idx / float(self.sample_rate)
            duration = len(samples) / float(self.sample_rate)
            yield AudioChunk(
                samples=samples,
                sample_rate=self.sample_rate,
                timestamp=timestamp,
                duration=duration,
                index=chunk_index,
            )
            idx = end
            chunk_index += 1
