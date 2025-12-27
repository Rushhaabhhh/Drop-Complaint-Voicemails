"""
Beep Detector - Detects voicemail beeps using signal processing
"""
import numpy as np
from scipy import signal
from typing import Optional, Tuple


class BeepDetector:
    def __init__(
        self,
        min_beep_freq: float = 700,
        max_beep_freq: float = 1200,
        min_duration: float = 0.3,
        amplitude_threshold: float = 0.15
    ):
        """
        Initialize beep detector
        
        Args:
            min_beep_freq: Minimum frequency for beep detection (Hz)
            max_beep_freq: Maximum frequency for beep detection (Hz)
            min_duration: Minimum duration for valid beep (seconds)
            amplitude_threshold: Minimum amplitude threshold
        """
        self.min_beep_freq = min_beep_freq
        self.max_beep_freq = max_beep_freq
        self.min_duration = min_duration
        self.amplitude_threshold = amplitude_threshold
        
        # Track beep detection state
        self.beep_start_time = None
        self.beep_detected = False
        self.beep_timestamp = None
        
    def analyze_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        timestamp: float
    ) -> Optional[float]:
        """
        Analyze audio chunk for beep
        
        Args:
            audio_chunk: Audio data chunk
            sample_rate: Sample rate
            timestamp: Current timestamp
            
        Returns:
            Beep timestamp if detected, None otherwise
        """
        if len(audio_chunk) < 100:  # Too short to analyze
            return None
        
        # Calculate amplitude
        amplitude = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Check if amplitude is sufficient
        if amplitude < self.amplitude_threshold:
            self.beep_start_time = None
            return None
        
        # Perform FFT to get frequency spectrum
        fft = np.fft.rfft(audio_chunk)
        freqs = np.fft.rfftfreq(len(audio_chunk), 1/sample_rate)
        magnitudes = np.abs(fft)
        
        # Find dominant frequency
        if len(magnitudes) > 0:
            dominant_freq_idx = np.argmax(magnitudes)
            dominant_freq = freqs[dominant_freq_idx]
            
            # Check if dominant frequency is in beep range
            if self.min_beep_freq <= dominant_freq <= self.max_beep_freq:
                # Check if this is a sustained tone
                peak_magnitude = magnitudes[dominant_freq_idx]
                avg_magnitude = np.mean(magnitudes)
                
                # Beep should have a clear dominant frequency
                if peak_magnitude > 3 * avg_magnitude:
                    if self.beep_start_time is None:
                        self.beep_start_time = timestamp
                    
                    # Check if beep duration is sufficient
                    beep_duration = timestamp - self.beep_start_time
                    if beep_duration >= self.min_duration and not self.beep_detected:
                        self.beep_detected = True
                        self.beep_timestamp = self.beep_start_time
                        return self.beep_timestamp
            else:
                self.beep_start_time = None
        
        return None
    
    def reset(self):
        """Reset detector state"""
        self.beep_start_time = None
        self.beep_detected = False
        self.beep_timestamp = None
    
    def has_detected_beep(self) -> bool:
        """Check if beep has been detected"""
        return self.beep_detected
    
    def get_beep_timestamp(self) -> Optional[float]:
        """Get the timestamp when beep was detected"""
        return self.beep_timestamp