"""
Silence Detector - Detects silence periods in audio
"""
import numpy as np
from typing import Optional


class SilenceDetector:
    def __init__(
        self,
        silence_threshold: float = 0.015,  # Slightly more sensitive
        min_silence_duration: float = 0.4   # Reduced from 0.5
    ):
        """
        Initialize silence detector
        
        Args:
            silence_threshold: RMS threshold below which audio is considered silent
            min_silence_duration: Minimum duration of silence to be significant (seconds)
        """
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        
        # Track silence state
        self.silence_start = None
        self.current_silence_duration = 0.0
        self.last_speech_time = 0.0
        
    def analyze_chunk(
        self,
        audio_chunk: np.ndarray,
        timestamp: float
    ) -> dict:
        """
        Analyze audio chunk for silence
        
        Args:
            audio_chunk: Audio data chunk
            timestamp: Current timestamp
            
        Returns:
            Dictionary with silence information
        """
        # Calculate RMS (Root Mean Square) amplitude
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        is_silent = rms < self.silence_threshold
        
        if is_silent:
            if self.silence_start is None:
                self.silence_start = timestamp
            
            self.current_silence_duration = timestamp - self.silence_start
        else:
            # Speech detected
            self.last_speech_time = timestamp
            self.silence_start = None
            self.current_silence_duration = 0.0
        
        return {
            'is_silent': is_silent,
            'rms': rms,
            'silence_duration': self.current_silence_duration,
            'time_since_last_speech': timestamp - self.last_speech_time if self.last_speech_time > 0 else 0
        }
    
    def has_significant_silence(self) -> bool:
        """Check if current silence duration exceeds minimum threshold"""
        return self.current_silence_duration >= self.min_silence_duration
    
    def get_silence_duration(self) -> float:
        """Get current silence duration"""
        return self.current_silence_duration
    
    def reset(self):
        """Reset detector state"""
        self.silence_start = None
        self.current_silence_duration = 0.0
        self.last_speech_time = 0.0