"""
Speech Analyzer - Transcribes audio and analyzes greeting completion
Supports both Deepgram (free) and OpenAI Whisper
"""
import os
import numpy as np
import tempfile
import soundfile as sf
from typing import Optional, List
import re


class SpeechAnalyzer:
    def __init__(self, deepgram_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        """
        Initialize speech analyzer with Deepgram (preferred) or OpenAI
        
        Args:
            deepgram_api_key: Deepgram API key (free tier available)
            openai_api_key: OpenAI API key (fallback)
        """
        self.deepgram_key = deepgram_api_key or os.getenv('DEEPGRAM_API_KEY')
        self.openai_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        self.client = None
        self.client_type = None
        
        # Try Deepgram first (free and good for voicemail)
        if self.deepgram_key and self.deepgram_key != 'your_deepgram_api_key_here':
            try:
                from deepgram import DeepgramClient, PrerecordedOptions, FileSource
                self.client = DeepgramClient(api_key=self.deepgram_key)
                self.client_type = 'deepgram'
                print("  ✅ Using Deepgram for transcription (free)")
            except Exception as e:
                print(f"  ⚠️  Deepgram initialization failed: {e}")
                print(f"      Trying without transcription...")
        
        # Fallback to OpenAI if Deepgram not available
        if not self.client and self.openai_key and self.openai_key != 'your_openai_api_key_here':
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_key)
                self.client_type = 'openai'
                print("  ✅ Using OpenAI Whisper for transcription")
            except Exception as e:
                print(f"  ⚠️  OpenAI initialization failed: {e}")
        
        if not self.client:
            print("  ℹ️  No API key configured - using beep + silence detection only")
        
        # Common end-of-greeting phrases
        self.end_phrases = [
            "leave a message",
            "leave your message",
            "after the beep",
            "after the tone",
            "at the tone",
            "please leave",
            "thank you",
            "thanks",
            "goodbye",
            "bye",
            "speak after",
            "message after",
            "leave me a message",
            "can't take your call"
        ]
        
        # Accumulated audio buffer for transcription
        self.audio_buffer = []
        self.sample_rate = None
        self.full_transcript = ""
        self.last_transcription_time = 0.0
        self.greeting_end_detected = False
        self.greeting_end_time = None
        
    def add_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int, timestamp: float):
        """
        Add audio chunk to buffer for transcription
        
        Args:
            audio_chunk: Audio data chunk
            sample_rate: Sample rate
            timestamp: Current timestamp
        """
        self.audio_buffer.extend(audio_chunk)
        self.sample_rate = sample_rate
    
    def transcribe_buffer(self, timestamp: float, force: bool = False) -> Optional[str]:
        """
        Transcribe accumulated audio buffer using Deepgram or OpenAI
        
        Args:
            timestamp: Current timestamp
            force: Force transcription even if buffer is small
            
        Returns:
            Transcription text or None
        """
        # Skip if no client available
        if not self.client:
            return None
        
        # Transcribe every 2 seconds or when forced
        if not force and (timestamp - self.last_transcription_time) < 2.0:
            return None
        
        if not self.audio_buffer:
            return None
        
        try:
            # Save buffer to temporary file
            audio_array = np.array(self.audio_buffer)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_array, self.sample_rate)
                temp_path = temp_file.name
            
            # Transcribe based on client type
            if self.client_type == 'deepgram':
                transcription_text = self._transcribe_deepgram(temp_path)
            elif self.client_type == 'openai':
                transcription_text = self._transcribe_openai(temp_path)
            else:
                transcription_text = None
            
            # Clean up temp file
            os.unlink(temp_path)
            
            if transcription_text:
                # Update transcript
                if transcription_text not in self.full_transcript:
                    self.full_transcript += " " + transcription_text
                    self.last_transcription_time = timestamp
                    
                    # Check for greeting end
                    if not self.greeting_end_detected:
                        self._check_greeting_end(timestamp)
                    
                    return transcription_text
        
        except Exception as e:
            # Only print error once per file
            if not hasattr(self, '_error_printed'):
                print(f"  ⚠️  Transcription unavailable: {e}")
                print(f"      Will use beep + silence detection only")
                self._error_printed = True
        
        return None
    
    def _transcribe_deepgram(self, audio_path: str) -> Optional[str]:
        """Transcribe using Deepgram"""
        try:
            from deepgram import PrerecordedOptions, FileSource
            
            with open(audio_path, 'rb') as audio_file:
                buffer_data = audio_file.read()
            
            payload: FileSource = {
                'buffer': buffer_data
            }
            
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                language="en"
            )
            
            response = self.client.listen.prerecorded.v('1').transcribe_file(
                payload,
                options
            )
            
            # Extract transcript from response
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
    
    def _check_greeting_end(self, timestamp: float):
        """
        Check if transcript indicates greeting has ended
        
        Args:
            timestamp: Current timestamp
        """
        transcript_lower = self.full_transcript.lower()
        
        # Check for end phrases
        for phrase in self.end_phrases:
            if phrase in transcript_lower:
                self.greeting_end_detected = True
                self.greeting_end_time = timestamp
                return
        
        # Additional heuristic: if transcript is long enough and mentions voicemail
        if len(self.full_transcript) > 30:
            voicemail_keywords = ['voicemail', 'voice mail', 'message']
            if any(keyword in transcript_lower for keyword in voicemail_keywords):
                # Likely a voicemail greeting that has completed
                self.greeting_end_detected = True
                self.greeting_end_time = timestamp
    
    def has_greeting_ended(self) -> bool:
        """Check if greeting has been detected as ended"""
        return self.greeting_end_detected
    
    def get_greeting_end_time(self) -> Optional[float]:
        """Get timestamp when greeting ended"""
        return self.greeting_end_time
    
    def get_transcript(self) -> str:
        """Get full transcript"""
        return self.full_transcript.strip()
    
    def reset(self):
        """Reset analyzer state"""
        self.audio_buffer = []
        self.sample_rate = None
        self.full_transcript = ""
        self.last_transcription_time = 0.0
        self.greeting_end_detected = False
        self.greeting_end_time = None
        # Reset error flag for next file
        if hasattr(self, '_error_printed'):
            delattr(self, '_error_printed')