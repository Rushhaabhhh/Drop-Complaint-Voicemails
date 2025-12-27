"""
Enhanced Speech Analyzer - Better transcription and greeting detection
"""
import os
import numpy as np
import tempfile
import soundfile as sf
from typing import Optional, List, Dict
import re
from dataclasses import dataclass


@dataclass
class GreetingSignal:
    """Information about detected greeting end signal"""
    timestamp: float
    phrase: str
    confidence: float
    context: str


class SpeechAnalyzer:
    def __init__(self, deepgram_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        """
        Initialize enhanced speech analyzer with better phrase detection
        
        Args:
            deepgram_api_key: Deepgram API key (preferred - free tier available)
            openai_api_key: OpenAI API key (fallback)
        """
        self.deepgram_key = deepgram_api_key or os.getenv('DEEPGRAM_API_KEY')
        self.openai_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        self.client = None
        self.client_type = None
        
        # Initialize transcription client
        self._init_client()
        
        # Enhanced end-of-greeting phrases with confidence weights
        self.end_phrases = {
            # High confidence (definitive endings)
            "after the beep": 1.0,
            "after the tone": 1.0,
            "at the tone": 1.0,
            "at the beep": 1.0,
            "leave a message": 0.95,
            "leave your message": 0.95,
            "leave me a message": 0.95,
            "leave a detailed message": 0.95,
            
            # Medium-high confidence
            "please leave": 0.85,
            "kindly leave": 0.85,
            "feel free to leave": 0.85,
            "go ahead and leave": 0.85,
            
            # Medium confidence (common endings but less definitive)
            "thank you": 0.7,
            "thanks": 0.7,
            "have a great day": 0.75,
            "have a good day": 0.75,
            "talk to you soon": 0.75,
            "speak to you soon": 0.75,
            
            # Lower confidence (might be mid-greeting)
            "goodbye": 0.6,
            "bye": 0.6,
            "can't take your call": 0.65,
            "unable to answer": 0.65,
            "not available": 0.65,
        }
        
        # Phrases that indicate greeting is still ongoing
        self.ongoing_phrases = [
            "you've reached",
            "this is",
            "my name is",
            "i'm calling",
            "for",  # "for sales press 1"
            "press",
            "dial",
            "if you",
        ]
        
        # State management
        self.audio_buffer = []
        self.sample_rate = None
        self.full_transcript = ""
        self.last_transcription_time = 0.0
        self.greeting_end_signals: List[GreetingSignal] = []
        self.greeting_end_detected = False
        self.greeting_end_time = None
        self.transcription_chunks = []
        
    def _init_client(self):
        """Initialize Deepgram or OpenAI client"""
        # Try Deepgram first (free tier, good for phone audio)
        if self.deepgram_key and self.deepgram_key != 'your_deepgram_api_key_here':
            try:
                from deepgram import DeepgramClient
                self.client = DeepgramClient(api_key=self.deepgram_key)
                self.client_type = 'deepgram'
                print("  ✅ Using Deepgram for transcription (recommended)")
            except Exception as e:
                print(f"  ⚠️  Deepgram init failed: {e}")
        
        # Fallback to OpenAI
        if not self.client and self.openai_key and self.openai_key != 'your_openai_api_key_here':
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_key)
                self.client_type = 'openai'
                print("  ✅ Using OpenAI Whisper for transcription")
            except Exception as e:
                print(f"  ⚠️  OpenAI init failed: {e}")
        
        if not self.client:
            print("  ℹ️  No transcription - using beep + silence detection only")
    
    def add_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int, timestamp: float):
        """Add audio chunk to buffer"""
        self.audio_buffer.extend(audio_chunk)
        self.sample_rate = sample_rate
    
    def transcribe_buffer(self, timestamp: float, force: bool = False) -> Optional[str]:
        """
        Transcribe accumulated audio buffer
        
        Args:
            timestamp: Current timestamp
            force: Force transcription
            
        Returns:
            New transcription text or None
        """
        if not self.client:
            return None
        
        # Transcribe every 1.5 seconds for better real-time detection
        if not force and (timestamp - self.last_transcription_time) < 1.5:
            return None
        
        if not self.audio_buffer or len(self.audio_buffer) < 100:
            return None
        
        try:
            # Save buffer to temp file
            audio_array = np.array(self.audio_buffer, dtype=np.float32)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, audio_array, self.sample_rate)
                temp_path = f.name
            
            # Transcribe
            if self.client_type == 'deepgram':
                text = self._transcribe_deepgram(temp_path)
            else:
                text = self._transcribe_openai(temp_path)
            
            os.unlink(temp_path)
            
            if text:
                # Store this chunk with timestamp
                self.transcription_chunks.append({
                    'timestamp': timestamp,
                    'text': text
                })
                
                # Update full transcript if new content
                text_lower = text.lower().strip()
                if text_lower and text_lower not in self.full_transcript.lower():
                    self.full_transcript += " " + text
                    self.last_transcription_time = timestamp
                    
                    # Analyze for greeting end
                    if not self.greeting_end_detected:
                        self._analyze_greeting_end(text, timestamp)
                    
                    return text
        
        except Exception as e:
            if not hasattr(self, '_error_shown'):
                print(f"  ⚠️  Transcription unavailable: {str(e)[:50]}")
                print(f"      Using beep + silence detection only")
                self._error_shown = True
        
        return None
    
    def _transcribe_deepgram(self, audio_path: str) -> Optional[str]:
        """Transcribe using Deepgram with optimized settings for voicemail"""
        try:
            from deepgram import PrerecordedOptions, FileSource
            
            with open(audio_path, 'rb') as audio_file:
                buffer_data = audio_file.read()
            
            payload: FileSource = {'buffer': buffer_data}
            
            # Optimized options for voicemail/phone audio
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                language="en-US",
                punctuate=True,
                utterances=False,
                diarize=False
            )
            
            response = self.client.listen.rest.v('1').transcribe_file(payload, options)
            
            if response and hasattr(response, 'results'):
                transcript = response.results.channels[0].alternatives[0].transcript
                return transcript.strip()
        
        except Exception as e:
            raise Exception(f"Deepgram error: {e}")
        
        return None
    
    def _transcribe_openai(self, audio_path: str) -> Optional[str]:
        """Transcribe using OpenAI Whisper"""
        try:
            with open(audio_path, 'rb') as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            return transcription.text.strip()
        
        except Exception as e:
            raise Exception(f"OpenAI error: {e}")
    
    def _analyze_greeting_end(self, text: str, timestamp: float):
        """
        Analyze transcript for greeting end signals with confidence scoring
        
        Args:
            text: Transcription text to analyze
            timestamp: Current timestamp
        """
        text_lower = text.lower().strip()
        
        # Check if greeting is still ongoing
        is_ongoing = any(phrase in text_lower for phrase in self.ongoing_phrases)
        
        # Check for end phrases
        best_match = None
        best_confidence = 0.0
        
        for phrase, confidence in self.end_phrases.items():
            if phrase in text_lower:
                # Apply penalty if ongoing indicators present
                if is_ongoing and confidence < 0.8:
                    confidence *= 0.7
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = phrase
        
        if best_match and best_confidence >= 0.7:
            # Strong signal for greeting end
            signal = GreetingSignal(
                timestamp=timestamp,
                phrase=best_match,
                confidence=best_confidence,
                context=text_lower[:100]
            )
            
            self.greeting_end_signals.append(signal)
            
            # Mark as detected if high confidence
            if not self.greeting_end_detected:
                self.greeting_end_detected = True
                self.greeting_end_time = timestamp
        
        # Additional heuristic: Long transcript likely means greeting ended
        elif not self.greeting_end_detected and len(self.full_transcript) > 80:
            # Check if transcript contains voicemail indicators
            voicemail_indicators = [
                'voicemail', 'voice mail', 'mailbox',
                'not available', 'cannot take', "can't take",
                'leave', 'message'
            ]
            
            indicator_count = sum(1 for ind in voicemail_indicators if ind in text_lower)
            
            if indicator_count >= 2:
                # Likely a complete greeting
                signal = GreetingSignal(
                    timestamp=timestamp,
                    phrase="multiple indicators",
                    confidence=0.75,
                    context="Long transcript with voicemail indicators"
                )
                self.greeting_end_signals.append(signal)
                self.greeting_end_detected = True
                self.greeting_end_time = timestamp
    
    def get_best_signal(self) -> Optional[GreetingSignal]:
        """Get the highest confidence greeting end signal"""
        if not self.greeting_end_signals:
            return None
        return max(self.greeting_end_signals, key=lambda s: s.confidence)
    
    def has_greeting_ended(self) -> bool:
        """Check if greeting has ended"""
        return self.greeting_end_detected
    
    def get_greeting_end_time(self) -> Optional[float]:
        """Get timestamp when greeting ended"""
        return self.greeting_end_time
    
    def get_transcript(self) -> str:
        """Get full transcript"""
        return self.full_transcript.strip()
    
    def get_all_signals(self) -> List[GreetingSignal]:
        """Get all detected greeting end signals"""
        return self.greeting_end_signals
    
    def reset(self):
        """Reset analyzer state"""
        self.audio_buffer = []
        self.sample_rate = None
        self.full_transcript = ""
        self.last_transcription_time = 0.0
        self.greeting_end_signals = []
        self.greeting_end_detected = False
        self.greeting_end_time = None
        self.transcription_chunks = []
        if hasattr(self, '_error_shown'):
            delattr(self, '_error_shown')