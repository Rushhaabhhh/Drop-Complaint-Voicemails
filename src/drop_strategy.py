"""
Enhanced Drop Strategy - Improved decision logic with confidence scoring
"""
from typing import Optional, Dict, List
from dataclasses import dataclass
import numpy as np


@dataclass
class DropDecision:
    """Result of drop decision analysis"""
    should_drop: bool
    drop_time: Optional[float]
    confidence: float  # 0.0 to 1.0
    reasoning: str
    method: str  # 'beep', 'speech', 'silence', 'timeout'
    signal_strengths: Dict[str, float]


class DropStrategy:
    def __init__(
        self,
        max_wait_time: float = 20.0,       # Reduced from 25s for faster fallback
        post_beep_delay: float = 0.15,     # Reduced from 0.2s
        post_greeting_delay: float = 1.0,  # Reduced from 1.5s
        required_silence: float = 0.8,     # Reduced from 1.0s
        min_greeting_duration: float = 2.0  # Minimum reasonable greeting time
    ):
        """
        Initialize enhanced drop strategy with multi-signal fusion
        
        Args:
            max_wait_time: Maximum time before fallback (seconds)
            post_beep_delay: Delay after beep detection (seconds)
            post_greeting_delay: Delay after greeting end (seconds)
            required_silence: Required silence for confirmation (seconds)
            min_greeting_duration: Minimum expected greeting duration (seconds)
        """
        self.max_wait_time = max_wait_time
        self.post_beep_delay = post_beep_delay
        self.post_greeting_delay = post_greeting_delay
        self.required_silence = required_silence
        self.min_greeting_duration = min_greeting_duration
        
        # Decision state
        self.drop_decision_made = False
        self.drop_time = None
        self.decision_history = []
        
        # Signal weights for fusion (can be tuned)
        self.weights = {
            'beep': 1.0,      # Highest priority
            'speech': 0.85,   # High confidence when clear phrase detected
            'silence': 0.6,   # Medium confidence
            'timeout': 0.3    # Low confidence fallback
        }
    
    def analyze(
        self,
        timestamp: float,
        beep_detected: bool,
        beep_time: Optional[float],
        beep_confidence: float,
        greeting_ended: bool,
        greeting_end_time: Optional[float],
        greeting_confidence: float,
        silence_duration: float,
        transcript: str,
        has_speech: bool
    ) -> DropDecision:
        """
        Multi-signal fusion for drop decision
        
        Args:
            timestamp: Current timestamp
            beep_detected: Whether beep detected
            beep_time: Beep timestamp
            beep_confidence: Beep detection confidence (0-1)
            greeting_ended: Whether greeting end detected via speech
            greeting_end_time: Greeting end timestamp
            greeting_confidence: Greeting end confidence (0-1)
            silence_duration: Current silence duration
            transcript: Full transcript
            has_speech: Whether any speech detected
            
        Returns:
            DropDecision with recommendation
        """
        if self.drop_decision_made:
            return self._create_decision(
                should_drop=True,
                drop_time=self.drop_time,
                confidence=1.0,
                reasoning="Decision already made",
                method="completed"
            )
        
        # Calculate signal strengths
        signals = self._evaluate_signals(
            timestamp=timestamp,
            beep_detected=beep_detected,
            beep_time=beep_time,
            beep_confidence=beep_confidence,
            greeting_ended=greeting_ended,
            greeting_end_time=greeting_end_time,
            greeting_confidence=greeting_confidence,
            silence_duration=silence_duration,
            transcript=transcript,
            has_speech=has_speech
        )
        
        # Decision logic with prioritization
        
        # PRIORITY 1: Beep detected (most reliable)
        if beep_detected and beep_time is not None and signals['beep'] > 0.7:
            elapsed = timestamp - beep_time
            
            if elapsed >= self.post_beep_delay:
                return self._finalize_decision(
                    drop_time=timestamp,
                    confidence=signals['beep'],
                    reasoning=f"Beep detected at {beep_time:.2f}s (conf: {beep_confidence:.2f})",
                    method="beep",
                    signal_strengths=signals
                )
        
        # PRIORITY 2: Strong speech + silence combination
        if greeting_ended and greeting_end_time is not None:
            elapsed = timestamp - greeting_end_time
            
            # Check if we have confirming silence
            if silence_duration >= self.required_silence:
                combined_confidence = (signals['speech'] + signals['silence']) / 2
                
                if elapsed >= self.post_greeting_delay and combined_confidence > 0.7:
                    return self._finalize_decision(
                        drop_time=timestamp,
                        confidence=combined_confidence,
                        reasoning=f"Greeting ended at {greeting_end_time:.2f}s, "
                                  f"{silence_duration:.1f}s silence (conf: {greeting_confidence:.2f})",
                        method="speech",
                        signal_strengths=signals
                    )
        
        # PRIORITY 3: Extended silence after speech (cautious)
        if has_speech and not beep_detected and not greeting_ended:
            if (timestamp >= self.min_greeting_duration and 
                silence_duration >= 1.5 and 
                signals['silence'] > 0.6):
                
                return self._finalize_decision(
                    drop_time=timestamp,
                    confidence=signals['silence'],
                    reasoning=f"Extended silence ({silence_duration:.1f}s) after speech detected",
                    method="silence",
                    signal_strengths=signals
                )
        
        # PRIORITY 4: Very long silence (likely dead air)
        if silence_duration >= 3.0 and timestamp >= 4.0:
            return self._finalize_decision(
                drop_time=timestamp,
                confidence=0.6,
                reasoning=f"Very long silence ({silence_duration:.1f}s) - likely safe to drop",
                method="silence",
                signal_strengths=signals
            )
        
        # PRIORITY 5: Timeout fallback
        if timestamp >= self.max_wait_time:
            return self._finalize_decision(
                drop_time=timestamp,
                confidence=signals['timeout'],
                reasoning=f"Timeout at {self.max_wait_time}s - fallback drop",
                method="timeout",
                signal_strengths=signals
            )
        
        # Continue waiting
        return self._create_decision(
            should_drop=False,
            drop_time=None,
            confidence=0.0,
            reasoning=f"Waiting for clear signal (silence: {silence_duration:.1f}s)",
            method="waiting",
            signal_strengths=signals
        )
    
    def _evaluate_signals(
        self,
        timestamp: float,
        beep_detected: bool,
        beep_time: Optional[float],
        beep_confidence: float,
        greeting_ended: bool,
        greeting_end_time: Optional[float],
        greeting_confidence: float,
        silence_duration: float,
        transcript: str,
        has_speech: bool
    ) -> Dict[str, float]:
        """
        Evaluate strength of each signal type
        
        Returns:
            Dictionary of signal strengths (0.0 to 1.0)
        """
        signals = {
            'beep': 0.0,
            'speech': 0.0,
            'silence': 0.0,
            'timeout': 0.0
        }
        
        # Beep signal strength
        if beep_detected and beep_time is not None:
            elapsed = timestamp - beep_time
            # Strong if beep was detected and enough time passed
            if elapsed >= self.post_beep_delay:
                signals['beep'] = beep_confidence * self.weights['beep']
        
        # Speech signal strength
        if greeting_ended and greeting_end_time is not None:
            elapsed = timestamp - greeting_end_time
            # Stronger as more time passes after detection
            time_factor = min(elapsed / self.post_greeting_delay, 1.0)
            signals['speech'] = greeting_confidence * time_factor * self.weights['speech']
        
        # Silence signal strength
        if has_speech and silence_duration > 0:
            # Stronger as silence duration increases
            silence_factor = min(silence_duration / self.required_silence, 1.0)
            # Bonus if we're past minimum greeting time
            time_bonus = 1.2 if timestamp >= self.min_greeting_duration else 0.8
            signals['silence'] = silence_factor * time_bonus * self.weights['silence']
        
        # Timeout signal (increases as we approach max_wait_time)
        if timestamp > 0:
            timeout_factor = min(timestamp / self.max_wait_time, 1.0)
            signals['timeout'] = timeout_factor * self.weights['timeout']
        
        return signals
    
    def _create_decision(
        self,
        should_drop: bool,
        drop_time: Optional[float],
        confidence: float,
        reasoning: str,
        method: str,
        signal_strengths: Optional[Dict[str, float]] = None
    ) -> DropDecision:
        """Create a DropDecision object"""
        return DropDecision(
            should_drop=should_drop,
            drop_time=drop_time,
            confidence=confidence,
            reasoning=reasoning,
            method=method,
            signal_strengths=signal_strengths or {}
        )
    
    def _finalize_decision(
        self,
        drop_time: float,
        confidence: float,
        reasoning: str,
        method: str,
        signal_strengths: Dict[str, float]
    ) -> DropDecision:
        """Finalize and record the drop decision"""
        self.drop_decision_made = True
        self.drop_time = drop_time
        
        decision = self._create_decision(
            should_drop=True,
            drop_time=drop_time,
            confidence=confidence,
            reasoning=reasoning,
            method=method,
            signal_strengths=signal_strengths
        )
        
        self.decision_history.append(decision)
        return decision
    
    def has_decided(self) -> bool:
        """Check if decision made"""
        return self.drop_decision_made
    
    def get_drop_time(self) -> Optional[float]:
        """Get decided drop time"""
        return self.drop_time
    
    def get_decision_history(self) -> List[DropDecision]:
        """Get history of all decisions"""
        return self.decision_history
    
    def reset(self):
        """Reset strategy state"""
        self.drop_decision_made = False
        self.drop_time = None
        self.decision_history = []