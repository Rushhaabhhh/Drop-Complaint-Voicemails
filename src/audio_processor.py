"""
Audio Processor - Handles streaming and chunking of audio files
"""
import numpy as np
import librosa
import soundfile as sf
from typing import Generator, Tuple


class AudioProcessor:
    def __init__(self, chunk_duration_ms: int = 100):
        """
        Initialize audio processor
        
        Args:
            chunk_duration_ms: Duration of each audio chunk in milliseconds
        """
        self.chunk_duration_ms = chunk_duration_ms
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            raise
    
    def stream_audio(self, file_path: str) -> Generator[Tuple[float, np.ndarray, int], None, None]:
        """
        Simulate real-time audio streaming by yielding chunks
        
        Args:
            file_path: Path to audio file
            
        Yields:
            Tuple of (timestamp, audio_chunk, sample_rate)
        """
        audio, sr = self.load_audio(file_path)
        
        # Calculate samples per chunk
        samples_per_chunk = int((self.chunk_duration_ms / 1000.0) * sr)
        
        # Stream audio in chunks
        total_samples = len(audio)
        current_sample = 0
        
        while current_sample < total_samples:
            # Extract chunk
            end_sample = min(current_sample + samples_per_chunk, total_samples)
            chunk = audio[current_sample:end_sample]
            
            # Calculate timestamp
            timestamp = current_sample / sr
            
            yield timestamp, chunk, sr
            
            current_sample = end_sample
    
    def get_audio_duration(self, file_path: str) -> float:
        """
        Get total duration of audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        audio, sr = self.load_audio(file_path)
        return len(audio) / sr
    
    def save_audio(self, audio: np.ndarray, sr: int, output_path: str):
        """
        Save audio to file
        
        Args:
            audio: Audio data
            sr: Sample rate
            output_path: Output file path
        """
        sf.write(output_path, audio, sr)