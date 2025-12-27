"""
Drop Strategy - Decides when to drop the voicemail message
"""
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class DropDecision:
    """Result of drop decision analysis"""
    should_drop: bool
    drop_time: Optional[float]
    confidence: str  # 'high', 'medium', 'low'
    reasoning: str


class VoicemailDropStrategy:
    def __init__(
        self,
        max_wait_time: float = 25.0,
        post_beep_delay: float = 0.2,
        post_greeting_delay: float = 1.5,
        required_silence: float = 1.0
    ):
        """
        Initialize drop strategy
        
        Args:
            max_wait_time: Maximum time to wait before fallback drop (seconds)
            post_beep_delay: Delay after beep detection (seconds)
            post_greeting_delay: Delay after greeting end detection (seconds)
            required_silence: Required silence duration to confirm greeting end (seconds)
        """
        self.max_wait_time = max_wait_time
        self.post_beep_delay = post_beep_delay
        self.post_greeting_delay = post_greeting_delay
        self.required_silence = required_silence
        
        # Track decision state
        self.drop_decision_made = False
        self.drop_time = None
        
    def analyze(
        self,
        timestamp: float,
        beep_detected: bool,
        beep_time: Optional[float],
        greeting_ended: bool,
        greeting_end_time: Optional[float],
        silence_duration: float,
        transcript: str
    ) -> DropDecision:
        """
        Analyze signals and decide if/when to drop message
        
        Args:
            timestamp: Current timestamp
            beep_detected: Whether beep has been detected
            beep_time: Timestamp of beep detection
            greeting_ended: Whether greeting end has been detected via STT
            greeting_end_time: Timestamp when greeting ended
            silence_duration: Current silence duration
            transcript: Current transcript
            
        Returns:
            DropDecision object
        """
        if self.drop_decision_made:
            return DropDecision(
                should_drop=True,
                drop_time=self.drop_time,
                confidence='high',
                reasoning='Decision already made'
            )
        
        # PRIORITY 1: Beep detected - most reliable signal
        if beep_detected and beep_time is not None:
            # Wait for beep to finish + small delay
            if timestamp >= beep_time + self.post_beep_delay:
                self.drop_decision_made = True
                self.drop_time = timestamp
                return DropDecision(
                    should_drop=True,
                    drop_time=timestamp,
                    confidence='high',
                    reasoning=f'Beep detected at {beep_time:.2f}s, dropping after {self.post_beep_delay}s delay'
                )
        
        # PRIORITY 2: Greeting ended + sufficient silence (no beep scenario)
        if greeting_ended and greeting_end_time is not None:
            time_since_greeting_end = timestamp - greeting_end_time
            
            # Wait for silence to confirm greeting truly ended
            if silence_duration >= self.required_silence:
                if time_since_greeting_end >= self.post_greeting_delay:
                    self.drop_decision_made = True
                    self.drop_time = timestamp
                    return DropDecision(
                        should_drop=True,
                        drop_time=timestamp,
                        confidence='high',
                        reasoning=f'Greeting ended at {greeting_end_time:.2f}s with {silence_duration:.2f}s silence'
                    )
        
        # PRIORITY 3: Extended silence without clear greeting end (cautious approach)
        if not greeting_ended and not beep_detected:
            # If we have significant silence and some speech was detected
            if silence_duration >= 2.0 and len(transcript) > 10:
                # Likely greeting ended but we didn't catch the phrase
                if timestamp >= 5.0:  # Minimum reasonable greeting time
                    self.drop_decision_made = True
                    self.drop_time = timestamp
                    return DropDecision(
                        should_drop=True,
                        drop_time=timestamp,
                        confidence='medium',
                        reasoning=f'Extended silence ({silence_duration:.2f}s) after speech detected'
                    )
        
        # PRIORITY 4: Timeout fallback
        if timestamp >= self.max_wait_time:
            self.drop_decision_made = True
            self.drop_time = timestamp
            return DropDecision(
                should_drop=True,
                drop_time=timestamp,
                confidence='low',
                reasoning=f'Timeout reached ({self.max_wait_time}s), fallback drop'
            )
        
        # Continue waiting
        return DropDecision(
            should_drop=False,
            drop_time=None,
            confidence='n/a',
            reasoning='Waiting for clear signal'
        )
    
    def reset(self):
        """Reset strategy state"""
        self.drop_decision_made = False
        self.drop_time = None
    
    def has_decided(self) -> bool:
        """Check if drop decision has been made"""
        return self.drop_decision_made
    
    def get_drop_time(self) -> Optional[float]:
        """Get decided drop time"""
        return self.drop_time