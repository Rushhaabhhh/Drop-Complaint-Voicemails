"""
Enhanced Voicemail Processor - Main orchestrator with improved accuracy
"""
from typing import Dict, Optional
from .audio_processor import AudioProcessor
from .beep_detector import BeepDetector
from .silence_detector import SilenceDetector
from .speech_analyzer import SpeechAnalyzer
from .drop_strategy import DropStrategy


class VoicemailProcessor:
    def __init__(
        self,
        deepgram_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize enhanced voicemail processor
        
        Args:
            deepgram_api_key: Deepgram API key for speech analysis
            openai_api_key: OpenAI API key (fallback)
        """
        # Initialize components
        self.audio_processor = AudioProcessor(chunk_duration_ms=100)
        self.beep_detector = BeepDetector(
            target_freq=850,
            freq_tolerance=200,
            min_duration=0.15,
            amplitude_threshold=0.04,
            harmonic_check=True
        )
        self.silence_detector = SilenceDetector(
            silence_threshold=0.012,
            min_silence_duration=0.3
        )
        self.speech_analyzer = SpeechAnalyzer(
            deepgram_api_key=deepgram_api_key,
            openai_api_key=openai_api_key
        )
        self.drop_strategy = DropStrategy(
            max_wait_time=20.0,
            post_beep_delay=0.15,
            post_greeting_delay=1.0,
            required_silence=0.8
        )
    
    def process_voicemail(self, audio_file: str, verbose: bool = True) -> Dict:
        """
        Process voicemail with enhanced multi-signal approach
        
        Args:
            audio_file: Path to voicemail audio file
            verbose: Print detailed progress
            
        Returns:
            Dictionary with comprehensive results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"📞 Processing: {audio_file}")
            print(f"{'='*70}")
        
        # Reset all components
        self.beep_detector.reset()
        self.silence_detector.reset()
        self.speech_analyzer.reset()
        self.drop_strategy.reset()
        
        # Process audio stream
        drop_decision = None
        last_progress_time = 0.0
        has_any_speech = False
        
        for timestamp, chunk, sr in self.audio_processor.stream_audio(audio_file):
            # Progress indicator
            if verbose and (timestamp - last_progress_time) >= 2.0:
                print(f"  ⏱️  Processing: {timestamp:.1f}s", end='\r')
                last_progress_time = timestamp
            
            # Detect beep
            beep_time = self.beep_detector.analyze_chunk(chunk, sr, timestamp)
            if beep_time and verbose:
                best = self.beep_detector.get_best_candidate()
                if best:
                    print(f"  🔔 Beep: {beep_time:.2f}s "
                          f"(freq: {best.dominant_freq:.0f}Hz, conf: {best.confidence:.2f})" 
                          + " " * 20)
            
            # Detect silence
            silence_info = self.silence_detector.analyze_chunk(chunk, timestamp)
            
            # Track if speech detected
            if not silence_info['is_silent']:
                has_any_speech = True
            
            # Show significant silence
            if verbose and silence_info['silence_duration'] > 1.0:
                if not hasattr(self, '_last_silence_print') or \
                   (timestamp - self._last_silence_print) > 2.0:
                    print(f"  🔇 Silence: {silence_info['silence_duration']:.1f}s" 
                          + " " * 30)
                    self._last_silence_print = timestamp
            
            # Add to speech buffer
            self.speech_analyzer.add_audio_chunk(chunk, sr, timestamp)
            
            # Transcribe periodically
            new_text = self.speech_analyzer.transcribe_buffer(timestamp)
            if new_text and verbose:
                display = new_text[:70] + "..." if len(new_text) > 70 else new_text
                print(f"  📝 \"{display}\"" + " " * 10)
            
            # Show greeting end detection
            if self.speech_analyzer.has_greeting_ended() and verbose:
                if not hasattr(self, '_greeting_shown'):
                    signal = self.speech_analyzer.get_best_signal()
                    if signal:
                        print(f"  ✅ Greeting end: {signal.timestamp:.2f}s "
                              f"(phrase: \"{signal.phrase}\", conf: {signal.confidence:.2f})")
                    self._greeting_shown = True
            
            # Get drop decision
            best_beep = self.beep_detector.get_best_candidate()
            best_greeting = self.speech_analyzer.get_best_signal()
            
            decision = self.drop_strategy.analyze(
                timestamp=timestamp,
                beep_detected=self.beep_detector.has_detected_beep(),
                beep_time=self.beep_detector.get_beep_timestamp(),
                beep_confidence=best_beep.confidence if best_beep else 0.0,
                greeting_ended=self.speech_analyzer.has_greeting_ended(),
                greeting_end_time=self.speech_analyzer.get_greeting_end_time(),
                greeting_confidence=best_greeting.confidence if best_greeting else 0.0,
                silence_duration=self.silence_detector.get_silence_duration(),
                transcript=self.speech_analyzer.get_transcript(),
                has_speech=has_any_speech
            )
            
            # Check if should drop
            if decision.should_drop and not drop_decision:
                drop_decision = decision
                if verbose:
                    print(f"\n  🎯 DROP at {decision.drop_time:.2f}s " + " " * 30)
                    print(f"     Method: {decision.method.upper()}")
                    print(f"     Confidence: {decision.confidence:.2f}")
                    print(f"     Reasoning: {decision.reasoning}")
                    
                    # Show signal strengths
                    if decision.signal_strengths:
                        strengths = ", ".join([
                            f"{k}: {v:.2f}" 
                            for k, v in decision.signal_strengths.items() 
                            if v > 0.1
                        ])
                        if strengths:
                            print(f"     Signals: {strengths}")
                break
        
        if verbose:
            print()  # Clear progress line
        
        # Final transcription pass if no decision yet
        if not drop_decision:
            self.speech_analyzer.transcribe_buffer(timestamp + 0.5, force=True)
            
            best_beep = self.beep_detector.get_best_candidate()
            best_greeting = self.speech_analyzer.get_best_signal()
            
            drop_decision = self.drop_strategy.analyze(
                timestamp=timestamp + 0.5,
                beep_detected=self.beep_detector.has_detected_beep(),
                beep_time=self.beep_detector.get_beep_timestamp(),
                beep_confidence=best_beep.confidence if best_beep else 0.0,
                greeting_ended=self.speech_analyzer.has_greeting_ended(),
                greeting_end_time=self.speech_analyzer.get_greeting_end_time(),
                greeting_confidence=best_greeting.confidence if best_greeting else 0.0,
                silence_duration=self.silence_detector.get_silence_duration(),
                transcript=self.speech_analyzer.get_transcript(),
                has_speech=has_any_speech
            )
        
        # Compile comprehensive results
        best_beep = self.beep_detector.get_best_candidate()
        best_greeting = self.speech_analyzer.get_best_signal()
        
        results = {
            'audio_file': audio_file,
            'drop_timestamp': drop_decision.drop_time if drop_decision else None,
            'confidence': drop_decision.confidence if drop_decision else 0.0,
            'method': drop_decision.method if drop_decision else 'none',
            'reasoning': drop_decision.reasoning if drop_decision else 'No decision',
            
            'detected_signals': {
                'beep': {
                    'detected': self.beep_detector.has_detected_beep(),
                    'timestamp': self.beep_detector.get_beep_timestamp(),
                    'confidence': best_beep.confidence if best_beep else 0.0,
                    'frequency': best_beep.dominant_freq if best_beep else None,
                    'all_candidates': len(self.beep_detector.get_all_candidates())
                },
                'speech': {
                    'greeting_ended': self.speech_analyzer.has_greeting_ended(),
                    'timestamp': self.speech_analyzer.get_greeting_end_time(),
                    'confidence': best_greeting.confidence if best_greeting else 0.0,
                    'phrase': best_greeting.phrase if best_greeting else None,
                    'transcript': self.speech_analyzer.get_transcript(),
                    'all_signals': len(self.speech_analyzer.get_all_signals())
                },
                'silence': {
                    'final_duration': self.silence_detector.get_silence_duration(),
                    'speech_detected': has_any_speech
                }
            }
        }
        
        if verbose:
            self._print_summary(results)
        
        # Cleanup temporary attributes
        if hasattr(self, '_greeting_shown'):
            delattr(self, '_greeting_shown')
        if hasattr(self, '_last_silence_print'):
            delattr(self, '_last_silence_print')
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print detailed summary of results"""
        print(f"\n{'='*70}")
        print(f"📊 RESULTS SUMMARY")
        print(f"{'='*70}")
        
        signals = results['detected_signals']
        
        # Transcript
        transcript = signals['speech']['transcript']
        if transcript:
            display = transcript[:100] + "..." if len(transcript) > 100 else transcript
            print(f"\n📝 Transcript: \"{display}\"")
        else:
            print(f"\n📝 Transcript: (none - no API key)")
        
        # Detection methods
        print(f"\n🔍 Detection Methods:")
        
        # Beep
        beep = signals['beep']
        if beep['detected']:
            print(f"   🔔 Beep: YES at {beep['timestamp']:.2f}s")
            print(f"      Frequency: {beep['frequency']:.0f}Hz")
            print(f"      Confidence: {beep['confidence']:.2f}")
            if beep['all_candidates'] > 1:
                print(f"      ({beep['all_candidates']} candidates found)")
        else:
            print(f"   🔔 Beep: NOT DETECTED")
        
        # Speech
        speech = signals['speech']
        if speech['greeting_ended']:
            print(f"   💬 Greeting End: YES at {speech['timestamp']:.2f}s")
            print(f"      Phrase: \"{speech['phrase']}\"")
            print(f"      Confidence: {speech['confidence']:.2f}")
            if speech['all_signals'] > 1:
                print(f"      ({speech['all_signals']} signals detected)")
        else:
            print(f"   💬 Greeting End: NOT DETECTED via speech")
        
        # Silence
        silence = signals['silence']
        print(f"   🔇 Final Silence: {silence['final_duration']:.1f}s")
        print(f"   🗣️  Speech Present: {'YES' if silence['speech_detected'] else 'NO'}")
        
        print(f"\n{'='*70}\n")