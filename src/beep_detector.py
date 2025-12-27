"""FFT-based beep detector."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from audio_processor import AudioChunk


@dataclass
class BeepAnalysis:
    beep_detected: bool
    peak_frequency: float
    peak_magnitude_db: float
    spectral_purity_db: float
    confidence: float


class BeepDetector:
    def __init__(self, target_freq_min: float = 400.0, target_freq_max: float = 1000.0, 
                 min_peak_db: float = -25.0, min_purity_db: float = 6.0):
        self.target_freq_min = float(target_freq_min)
        self.target_freq_max = float(target_freq_max)
        self.min_peak_db = float(min_peak_db)
        self.min_purity_db = float(min_purity_db)

    def _fft_magnitude(self, samples: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(samples) == 0:
            return np.array([]), np.array([])

        win = np.hanning(len(samples))
        windowed = samples * win

        spec = np.fft.rfft(windowed)
        mag = np.abs(spec) + 1e-9
        mag_db = 20.0 * np.log10(mag)
        freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)
        return freqs, mag_db

    def process_chunk(self, chunk: AudioChunk) -> BeepAnalysis:
        freqs, mag_db = self._fft_magnitude(chunk.samples, chunk.sample_rate)

        if freqs.size == 0:
            return BeepAnalysis(False, 0.0, -120.0, 0.0, 0.0)

        band_mask = (freqs >= self.target_freq_min) & (freqs <= self.target_freq_max)
        if not np.any(band_mask):
            return BeepAnalysis(False, 0.0, -120.0, 0.0, 0.0)

        band_freqs = freqs[band_mask]
        band_mag_db = mag_db[band_mask]

        peak_idx = int(np.argmax(band_mag_db))
        peak_freq = float(band_freqs[peak_idx])
        peak_mag_db = float(band_mag_db[peak_idx])

        median_band = float(np.median(band_mag_db))
        spectral_purity_db = peak_mag_db - median_band

        strong_enough = peak_mag_db >= self.min_peak_db
        pure_enough = spectral_purity_db >= self.min_purity_db
        beep_detected = strong_enough and pure_enough

        peak_margin = max(0.0, min(20.0, peak_mag_db - self.min_peak_db))
        purity_margin = max(0.0, min(20.0, spectral_purity_db - self.min_purity_db))
        confidence = (peak_margin / 20.0 + purity_margin / 20.0) / 2.0

        return BeepAnalysis(
            beep_detected=beep_detected,
            peak_frequency=peak_freq,
            peak_magnitude_db=peak_mag_db,
            spectral_purity_db=spectral_purity_db,
            confidence=confidence,
        )
