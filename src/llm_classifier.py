"""
LLM-based greeting end classifier (Claude).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import os

from .silence_detector import SilenceAnalysis
from .beep_detector import BeepAnalysis
from .stt_processor import STTState

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


@dataclass
class LLMDecision:
    confidence: float  # 0–1
    reason: str


class GreetingEndClassifier:
    """
    Calls an LLM (Claude) to estimate whether the voicemail greeting has ended.
    """

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing in environment")

        if anthropic is None:
            raise ImportError("anthropic package not installed. pip install anthropic")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def classify(
        self,
        transcript_state: STTState,
        silence_analysis: SilenceAnalysis,
        beep_analysis: BeepAnalysis,
        elapsed_seconds: float,
    ) -> LLMDecision:
        prompt = f"""
You are helping decide when a voicemail greeting has ended in an automated system.

Information you have:
- Transcript so far: {transcript_state.transcript!r}
- Consecutive silence seconds: {silence_analysis.consecutive_silence_sec:.2f}
- Current chunk energy dB: {silence_analysis.energy_db:.1f}
- Beep detected: {beep_analysis.beep_detected}
- Beep confidence: {beep_analysis.confidence:.2f}
- Elapsed audio time: {elapsed_seconds:.2f} seconds

Question: Has the greeting finished and the mailbox is ready for recording?

Respond ONLY in this JSON format:
{{"confidence": <0-100 integer>, "reason": "<short explanation>"}}
""".strip()

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        # Expect a single JSON block as text
        content = "".join(block.text for block in msg.content if block.type == "text")  # type: ignore
        import json

        try:
            data = json.loads(content)
            conf = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", "")).strip()
        except Exception:
            conf = 0.0
            reason = "LLM returned invalid JSON."

        conf = max(0.0, min(100.0, conf))
        return LLMDecision(confidence=conf / 100.0, reason=reason)
