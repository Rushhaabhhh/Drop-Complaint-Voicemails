"""
Voicemail Processor - Main orchestrator that combines all components
"""
from typing import Dict, Optional
from .audio_processor import AudioProcessor
from .beep_detector import BeepDetector
from .silence_detector import SilenceDetector
from .speech_analyzer import SpeechAnalyzer
from .drop_strategy import VoicemailDropStrategy, DropDecision


class VoicemailProcessor:
    def __init__(self, deepgram_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        """
        Initialize voicemail processor with all components
        
        Args:
            deepgram_api_key: Deepgram API key for speech analysis
            openai_api_key: OpenAI API key (fallback)
        """
        self.audio_processor = AudioProcessor(chunk_duration_ms=100)
        self.beep_detector = BeepDetector()
        self.silence_detector = SilenceDetector()
        self.speech_analyzer = SpeechAnalyzer(
            deepgram_api_key=deepgram_api_key,
            openai_api_key=openai_api_key
        )
        self.drop_strategy = VoicemailDropStrategy()
        
    def process_voicemail(self, audio_file: str, verbose: bool = True) -> Dict:
        """
        Process a voicemail file and determine drop time
        
        Args:
            audio_file: Path to voicemail audio file
            verbose: Print progress information
            
        Returns:
            Dictionary with processing results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {audio_file}")
            print(f"{'='*60}")
        
        # Reset all detectors
        self.beep_detector.reset()
        self.silence_detector.reset()
        self.speech_analyzer.reset()
        self.drop_strategy.reset()
        
        # Stream audio and analyze
        drop_decision = None
        last_progress_time = 0.0
        
        for timestamp, chunk, sr in self.audio_processor.stream_audio(audio_file):
            # Show progress every 2 seconds
            if verbose and (timestamp - last_progress_time) >= 2.0:
                print(f"  ⏱️  Processing: {timestamp:.1f}s", end='\r')
                last_progress_time = timestamp
            
            # Analyze beep
            beep_time = self.beep_detector.analyze_chunk(chunk, sr, timestamp)
            if beep_time and verbose:
                print(f"  🔔 Beep detected at {beep_time:.2f}s" + " " * 20)
            
            # Analyze silence
            silence_info = self.silence_detector.analyze_chunk(chunk, timestamp)
            
            # Show silence info periodically
            if verbose and silence_info['is_silent'] and silence_info['silence_duration'] > 1.0:
                if not hasattr(self, '_last_silence_print') or (timestamp - self._last_silence_print) > 2.0:
                    print(f"  🔇 Silence: {silence_info['silence_duration']:.1f}s" + " " * 20)
                    self._last_silence_print = timestamp
            
            # Add to speech analyzer buffer
            self.speech_analyzer.add_audio_chunk(chunk, sr, timestamp)
            
            # Transcribe periodically
            new_transcript = self.speech_analyzer.transcribe_buffer(timestamp)
            if new_transcript and verbose:
                print(f"  📝 Transcript: {new_transcript[:60]}..." if len(new_transcript) > 60 else f"  📝 Transcript: {new_transcript}")
            
            if self.speech_analyzer.has_greeting_ended() and verbose and not hasattr(self, '_greeting_end_printed'):
                print(f"  ✅ Greeting end detected at {self.speech_analyzer.get_greeting_end_time():.2f}s")
                self._greeting_end_printed = True
            
            # Get drop decision
            decision = self.drop_strategy.analyze(
                timestamp=timestamp,
                beep_detected=self.beep_detector.has_detected_beep(),
                beep_time=self.beep_detector.get_beep_timestamp(),
                greeting_ended=self.speech_analyzer.has_greeting_ended(),
                greeting_end_time=self.speech_analyzer.get_greeting_end_time(),
                silence_duration=self.silence_detector.get_silence_duration(),
                transcript=self.speech_analyzer.get_transcript()
            )
            
            # Check if we should drop
            if decision.should_drop and not drop_decision:
                drop_decision = decision
                if verbose:
                    print(f"\n  🎯 DROP DECISION at {decision.drop_time:.2f}s" + " " * 20)
                    print(f"     Confidence: {decision.confidence}")
                    print(f"     Reasoning: {decision.reasoning}")
                break
        
        if verbose:
            print()  # Clear the progress line
        
        # Final transcription if needed
        if not drop_decision:
            self.speech_analyzer.transcribe_buffer(timestamp + 1, force=True)
            
            # Make final decision
            drop_decision = self.drop_strategy.analyze(
                timestamp=timestamp + 1,
                beep_detected=self.beep_detector.has_detected_beep(),
                beep_time=self.beep_detector.get_beep_timestamp(),
                greeting_ended=self.speech_analyzer.has_greeting_ended(),
                greeting_end_time=self.speech_analyzer.get_greeting_end_time(),
                silence_duration=self.silence_detector.get_silence_duration(),
                transcript=self.speech_analyzer.get_transcript()
            )
        
        # Compile results
        results = {
            'audio_file': audio_file,
            'drop_timestamp': drop_decision.drop_time if drop_decision else None,
            'confidence': drop_decision.confidence if drop_decision else 'unknown',
            'reasoning': drop_decision.reasoning if drop_decision else 'No decision made',
            'detected_signals': {
                'beep_detected': self.beep_detector.has_detected_beep(),
                'beep_timestamp': self.beep_detector.get_beep_timestamp(),
                'greeting_ended': self.speech_analyzer.has_greeting_ended(),
                'greeting_end_timestamp': self.speech_analyzer.get_greeting_end_time(),
                'transcript': self.speech_analyzer.get_transcript(),
                'final_silence_duration': self.silence_detector.get_silence_duration()
            }
        }
        
        if verbose:
            print(f"\n  📊 Full Transcript: {results['detected_signals']['transcript']}")
            print(f"{'='*60}\n")
        
        # Clean up the temporary attribute
        if hasattr(self, '_greeting_end_printed'):
            delattr(self, '_greeting_end_printed')
        
        return results