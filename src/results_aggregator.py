"""Results aggregation and export."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import json
import csv


@dataclass
class CallDecision:
    file_path: str
    decision_time_sec: float
    total_duration_sec: float
    confidence: float
    reason: str
    beep_detected: bool


class ResultsAggregator:
    def __init__(self):
        self.decisions: List[CallDecision] = []

    def add_decision(self, decision: CallDecision) -> None:
        self.decisions.append(decision)

    def to_json(self) -> Dict[str, Any]:
        return {"results": [asdict(d) for d in self.decisions]}

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.decisions:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.decisions[0]).keys()))
            writer.writeheader()
            for d in self.decisions:
                writer.writerow(asdict(d))
