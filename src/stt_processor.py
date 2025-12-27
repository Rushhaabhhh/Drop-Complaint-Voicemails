"""Speech-to-text placeholder."""

from __future__ import annotations
from dataclasses import dataclass
from audio_processor import AudioChunk


@dataclass
class STTState:
    transcript: str = ""
    last_update_time: float = 0.0


class SpeechToTextProcessor:
    """Placeholder STT - Whisper optional later."""
    
    def __init__(self, model_name: str = "base"):
        self.sample_rate = 16000
        self.buffer = []
        self.state = STTState()

    def add_chunk(self, chunk: AudioChunk) -> None:
        if chunk.sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz, got {chunk.sample_rate}")
        self.buffer.append(chunk.samples.copy())

    def finalize(self) -> STTState:
        if not self.buffer:
            self.state.transcript = ""
            return self.state
        self.state.transcript = ""  # No STT for now
        return self.state
