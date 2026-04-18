"""Collects and persists training tuples for offline perceptron calibration."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingTuple:
    session_id:    str
    turn_id:       str
    timestamp:     float
    text:          str
    genome_id:     str

    # Intermediate supervision targets — None until labelled
    event_label:   int   | None = None   # P1 target
    action_target: int   | None = None   # P6 target

    # Feedback signal: +1 success, -1 failure, 0 neutral
    feedback:      float        = 0.0
    feedback_type: str          = ''

    # P1–P6 outputs from this turn (for replaying)
    event_probs:   list[float]  = field(default_factory=list)
    action_probs:  list[float]  = field(default_factory=list)
    thought_vec:   list[float]  = field(default_factory=list)
    conflict_vec:  list[float]  = field(default_factory=list)
    regulator_snapshot: dict    = field(default_factory=dict)
    action_id:     int          = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            'session_id':   self.session_id,
            'turn_id':      self.turn_id,
            'timestamp':    self.timestamp,
            'text':         self.text,
            'genome_id':    self.genome_id,
            'event_label':  self.event_label,
            'action_target':self.action_target,
            'feedback':     self.feedback,
            'feedback_type':self.feedback_type,
            'event_probs':  self.event_probs,
            'action_probs': self.action_probs,
            'thought_vec':  self.thought_vec,
            'conflict_vec': self.conflict_vec,
            'regulator_snapshot': self.regulator_snapshot,
            'action_id':    self.action_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'TrainingTuple':
        t = cls(
            session_id=d['session_id'], turn_id=d['turn_id'],
            timestamp=float(d['timestamp']), text=d['text'],
            genome_id=d['genome_id'],
        )
        t.event_label    = d.get('event_label')
        t.action_target  = d.get('action_target')
        t.feedback       = float(d.get('feedback', 0.0))
        t.feedback_type  = d.get('feedback_type', '')
        t.event_probs    = d.get('event_probs', [])
        t.action_probs   = d.get('action_probs', [])
        t.thought_vec    = d.get('thought_vec', [])
        t.conflict_vec   = d.get('conflict_vec', [])
        t.regulator_snapshot = d.get('regulator_snapshot', {})
        t.action_id      = int(d.get('action_id', -1))
        return t


class TrainingStore:
    """Append-only JSONL store for training tuples."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, tup: TrainingTuple) -> None:
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(tup.to_dict(), ensure_ascii=False) + '\n')

    def load_all(self) -> list[TrainingTuple]:
        if not self.path.exists():
            return []
        tuples = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                try:
                    tuples.append(TrainingTuple.from_dict(json.loads(line)))
                except Exception:
                    pass
        return tuples

    def load_recent(self, n: int = 500) -> list[TrainingTuple]:
        return self.load_all()[-n:]

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip())

    def apply_feedback(self, turn_id: str, feedback: float, feedback_type: str) -> bool:
        """Rewrite last matching turn with updated feedback. Returns True if found."""
        if not self.path.exists():
            return False
        lines = self.path.read_text(encoding='utf-8').splitlines()
        found = False
        new_lines = []
        for line in reversed(lines):
            if not found and line.strip():
                try:
                    d = json.loads(line)
                    if d.get('turn_id') == turn_id:
                        d['feedback'] = feedback
                        d['feedback_type'] = feedback_type
                        line = json.dumps(d, ensure_ascii=False)
                        found = True
                except Exception:
                    pass
            new_lines.append(line)
        if found:
            self.path.write_text('\n'.join(reversed(new_lines)) + '\n', encoding='utf-8')
        return found
