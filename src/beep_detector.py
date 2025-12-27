"""
Enhanced Beep Detector - Improved accuracy with multi-band analysis
"""
import numpy as np
from scipy import signal
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class BeepCandidate:
    """Information about a potential beep"""
    start_time: float
    dominant_freq: float
    amplitude: float
    duration: float
    confidence: float


class BeepDetector:
    def __init__(
        self,
        target_freq: float = 850,        # Common voicemail beep frequency
        freq_tolerance: float = 150,      # +/- tolerance
        min_duration: float = 0.15,       # Minimum beep duration
        amplitude_threshold: float = 0.05, # Lower threshold for quiet beeps
        harmonic_check: bool = True       # Check for harmonic clarity
    ):
        """
        Initialize enhanced beep detector with multi-band frequency analysis
        
        Args:
            target_freq: Target beep frequency (Hz)
            freq_tolerance: Frequency tolerance range (Hz)
            min_duration: Minimum duration for valid beep (seconds)
            amplitude_threshold: Minimum RMS amplitude threshold
            harmonic_check: Enable harmonic analysis for better discrimination
        """
        self.target_freq = target_freq
        self.freq_tolerance = freq_tolerance
        self.min_freq = target_freq - freq_tolerance
        self.max_freq = target_freq + freq_tolerance
        self.min_duration = min_duration
        self.amplitude_threshold = amplitude_threshold
        self.harmonic_check = harmonic_check
        
        # State tracking
        self.beep_start_time = None
        self.beep_detected = False
        self.beep_timestamp = None
        self.current_candidate = None
        self.all_candidates: List[BeepCandidate] = []
        
        # Rolling history for better detection
        self.freq_history = []
        self.amp_history = []
        self.history_window = 5  # Keep last 5 chunks
        
    def analyze_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        timestamp: float
    ) -> Optional[float]:
        """
        Analyze audio chunk for beep with enhanced multi-signal approach
        
        Args:
            audio_chunk: Audio data chunk
            sample_rate: Sample rate
            timestamp: Current timestamp
            
        Returns:
            Beep timestamp if detected, None otherwise
        """
        if len(audio_chunk) < 50:
            return None
        
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Early exit if too quiet
        if rms < self.amplitude_threshold:
            self._reset_candidate()
            return None
        
        # Perform FFT analysis
        # Use windowing for better frequency resolution
        window = signal.windows.hann(len(audio_chunk))
        windowed_chunk = audio_chunk * window
        
        fft = np.fft.rfft(windowed_chunk, n=len(audio_chunk) * 2)  # Zero-padding
        freqs = np.fft.rfftfreq(len(audio_chunk) * 2, 1/sample_rate)
        magnitudes = np.abs(fft)
        
        # Find dominant frequency
        if len(magnitudes) == 0:
            return None
            
        dominant_idx = np.argmax(magnitudes)
        dominant_freq = freqs[dominant_idx]
        peak_magnitude = magnitudes[dominant_idx]
        
        # Calculate average magnitude (excluding DC and very low frequencies)
        valid_idx = freqs > 50  # Exclude DC and low rumble
        avg_magnitude = np.mean(magnitudes[valid_idx]) if np.any(valid_idx) else 0
        
        # Calculate signal-to-noise ratio
        snr = peak_magnitude / avg_magnitude if avg_magnitude > 0 else 0
        
        # Update history
        self.freq_history.append(dominant_freq)
        self.amp_history.append(rms)
        if len(self.freq_history) > self.history_window:
            self.freq_history.pop(0)
            self.amp_history.pop(0)
        
        # Check if frequency is in beep range
        in_range = self.min_freq <= dominant_freq <= self.max_freq
        
        # Enhanced beep criteria
        has_clear_tone = snr > 2.0  # Clear dominant frequency
        is_sustained = self._check_frequency_stability()
        
        if in_range and has_clear_tone:
            # Additional harmonic check if enabled
            if self.harmonic_check:
                harmonic_score = self._analyze_harmonics(magnitudes, freqs, dominant_freq)
                is_beep_like = harmonic_score > 0.6  # Good harmonic structure
            else:
                is_beep_like = True
            
            if is_beep_like:
                # Start or continue tracking this beep
                if self.beep_start_time is None:
                    self.beep_start_time = timestamp
                    self.current_candidate = BeepCandidate(
                        start_time=timestamp,
                        dominant_freq=dominant_freq,
                        amplitude=rms,
                        duration=0.0,
                        confidence=0.0
                    )
                
                # Update duration
                duration = timestamp - self.beep_start_time
                
                # Calculate confidence based on multiple factors
                confidence = self._calculate_confidence(
                    duration=duration,
                    snr=snr,
                    freq_stability=is_sustained,
                    amplitude=rms
                )
                
                # Check if this qualifies as a beep
                if duration >= self.min_duration and confidence > 0.7:
                    if not self.beep_detected:
                        self.beep_detected = True
                        self.beep_timestamp = self.beep_start_time
                        
                        # Save this candidate
                        self.all_candidates.append(BeepCandidate(
                            start_time=self.beep_start_time,
                            dominant_freq=dominant_freq,
                            amplitude=rms,
                            duration=duration,
                            confidence=confidence
                        ))
                        
                        return self.beep_timestamp
        else:
            # Not a beep, reset
            self._reset_candidate()
        
        return None
    
    def _check_frequency_stability(self) -> bool:
        """Check if recent frequencies are stable (indicating sustained tone)"""
        if len(self.freq_history) < 3:
            return False
        
        # Calculate standard deviation of recent frequencies
        freq_std = np.std(self.freq_history[-3:])
        
        # Stable if frequencies vary by less than 50 Hz
        return freq_std < 50
    
    def _analyze_harmonics(
        self,
        magnitudes: np.ndarray,
        freqs: np.ndarray,
        fundamental: float
    ) -> float:
        """
        Analyze harmonic structure to distinguish beeps from speech
        
        Beeps typically have:
        - Strong fundamental frequency
        - Weaker or absent harmonics
        - Simple spectral structure
        
        Returns:
            Score between 0 and 1 (higher = more beep-like)
        """
        # Find magnitude at fundamental
        fundamental_idx = np.argmin(np.abs(freqs - fundamental))
        fundamental_mag = magnitudes[fundamental_idx]
        
        if fundamental_mag == 0:
            return 0.0
        
        # Check expected harmonic locations (2f, 3f)
        harmonic_score = 1.0
        
        for harmonic_num in [2, 3]:
            harmonic_freq = fundamental * harmonic_num
            if harmonic_freq < freqs[-1]:  # Within our frequency range
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_mag = magnitudes[harmonic_idx]
                
                # Beeps should have weak harmonics relative to fundamental
                ratio = harmonic_mag / fundamental_mag
                
                # Penalize strong harmonics (indicates speech/complex sound)
                if ratio > 0.3:
                    harmonic_score -= 0.2
        
        # Check spectral flatness (beeps should have concentrated energy)
        # Calculate energy in ±100Hz band around fundamental
        band_mask = (freqs >= fundamental - 100) & (freqs <= fundamental + 100)
        band_energy = np.sum(magnitudes[band_mask] ** 2)
        total_energy = np.sum(magnitudes ** 2)
        
        concentration = band_energy / total_energy if total_energy > 0 else 0
        
        # Beeps should have >50% energy concentrated in narrow band
        if concentration > 0.5:
            harmonic_score += 0.3
        
        return np.clip(harmonic_score, 0.0, 1.0)
    
    def _calculate_confidence(
        self,
        duration: float,
        snr: float,
        freq_stability: bool,
        amplitude: float
    ) -> float:
        """Calculate confidence score for beep detection"""
        confidence = 0.0
        
        # Duration score (longer = more confident)
        if duration >= self.min_duration:
            confidence += 0.3
        if duration >= 0.3:
            confidence += 0.1
        
        # SNR score (clearer tone = more confident)
        if snr > 2.5:
            confidence += 0.3
        elif snr > 2.0:
            confidence += 0.2
        
        # Frequency stability score
        if freq_stability:
            confidence += 0.2
        
        # Amplitude score
        if amplitude > self.amplitude_threshold * 2:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _reset_candidate(self):
        """Reset current candidate tracking"""
        self.beep_start_time = None
        self.current_candidate = None
    
    def get_best_candidate(self) -> Optional[BeepCandidate]:
        """Get the most confident beep candidate"""
        if not self.all_candidates:
            return None
        return max(self.all_candidates, key=lambda c: c.confidence)
    
    def has_detected_beep(self) -> bool:
        """Check if beep has been detected"""
        return self.beep_detected
    
    def get_beep_timestamp(self) -> Optional[float]:
        """Get the timestamp when beep was detected"""
        return self.beep_timestamp
    
    def get_all_candidates(self) -> List[BeepCandidate]:
        """Get all beep candidates detected"""
        return self.all_candidates
    
    def reset(self):
        """Reset detector state"""
        self.beep_start_time = None
        self.beep_detected = False
        self.beep_timestamp = None
        self.current_candidate = None
        self.all_candidates = []
        self.freq_history = []
        self.amp_history = []