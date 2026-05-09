"""
Dataset layer: loads DataSets/* → TrainingTuple[] with event_label and action_target filled.

Sources:
  1. empatheticdialogues  (76 k utterances, 31 emotion categories)
  2. archive / DailyDialog (11 k dialogs, emotion 0-6, act 1-4)
  3. DialogStudio/Prosocial (safety signals)
  4. idea_attractors       (quality labels: wise/misleading/false_or_harmful/mobilizing)

All sources produce TrainingTuple objects compatible with perceptron_trainer.train_p1/p6.

Also provides build_coordinate_dataset() → CoordinateTrainingRecord[] for P-module training.
"""
from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .cognitive_pipeline import (
    ACTION_FAMILIES, ACTION_IDX, EVENT_TYPES, EVENT_IDX,
    CognitiveRuntime, RegulatorState,
)
from .genome import PersonalityGenome
from .training_store import TrainingTuple

# ─── Paths ────────────────────────────────────────────────────────────────────

_DATASETS = Path(__file__).parent.parent / 'DataSets'

# ─── EmpatheticDialogues mappings ─────────────────────────────────────────────

# 31 emotion labels → event_type (P1 target)
EMOTION_TO_EVENT: dict[str, str] = {
    'afraid':        'danger',
    'angry':         'criticism',
    'annoyed':       'criticism',
    'anticipating':  'opportunity',
    'anxious':       'uncertainty',
    'apprehensive':  'uncertainty',
    'ashamed':       'shame_trigger',
    'caring':        'intimacy',
    'confident':     'opportunity',
    'content':       'neutral',
    'disappointed':  'failure',
    'disgusted':     'criticism',
    'embarrassed':   'shame_trigger',
    'excited':       'opportunity',
    'faithful':      'intimacy',
    'furious':       'criticism',
    'grateful':      'praise',
    'guilty':        'shame_trigger',
    'hopeful':       'opportunity',
    'impressed':     'praise',
    'jealous':       'rejection',
    'joyful':        'praise',
    'lonely':        'rejection',
    'nostalgic':     'neutral',
    'prepared':      'opportunity',
    'proud':         'praise',
    'sad':           'failure',
    'sentimental':   'neutral',
    'surprised':     'novelty',
    'terrified':     'danger',
    'trusting':      'intimacy',
}

# 31 emotion labels → action_family (P6 target — what the system should do)
EMOTION_TO_ACTION: dict[str, str] = {
    'afraid':        'freeze',
    'angry':         'attack',
    'annoyed':       'seek_control',
    'anticipating':  'approach',
    'anxious':       'freeze',
    'apprehensive':  'freeze',
    'ashamed':       'withdraw',
    'caring':        'connect',
    'confident':     'approach',
    'content':       'approach',
    'disappointed':  'placate',
    'disgusted':     'avoid',
    'embarrassed':   'withdraw',
    'excited':       'approach',
    'faithful':      'connect',
    'furious':       'attack',
    'grateful':      'connect',
    'guilty':        'self_protect',
    'hopeful':       'approach',
    'impressed':     'approach',
    'jealous':       'avoid',
    'joyful':        'connect',
    'lonely':        'ask_for_help',
    'nostalgic':     'connect',
    'prepared':      'plan_small_step',
    'proud':         'approach',
    'sad':           'ask_for_help',
    'sentimental':   'connect',
    'surprised':     'freeze',
    'terrified':     'freeze',
    'trusting':      'connect',
}

# ─── Archive / DailyDialog mappings ───────────────────────────────────────────

# emotion: 0=neutral 1=disgust 2=surprise 3=fear 4=happiness 5=sadness 6=anger
ARCHIVE_EMOTION_TO_EVENT: dict[int, str] = {
    0: 'neutral',
    1: 'criticism',    # disgust
    2: 'novelty',      # surprise
    3: 'danger',       # fear
    4: 'praise',       # happiness
    5: 'failure',      # sadness
    6: 'criticism',    # anger
}

# act: 1=inform 2=question 3=directive 4=commissive
# combined with emotion → action
_ARCHIVE_EMOTION_ACT_TO_ACTION: dict[tuple[int, int], str] = {
    (0, 1): 'analyze',
    (0, 2): 'ask_for_help',
    (0, 3): 'seek_control',
    (0, 4): 'plan_small_step',
    (1, 1): 'avoid',
    (1, 2): 'seek_control',
    (1, 3): 'attack',
    (1, 4): 'avoid',
    (2, 1): 'approach',
    (2, 2): 'approach',
    (2, 3): 'reframe',
    (2, 4): 'approach',
    (3, 1): 'freeze',
    (3, 2): 'freeze',
    (3, 3): 'seek_control',
    (3, 4): 'self_protect',
    (4, 1): 'connect',
    (4, 2): 'connect',
    (4, 3): 'approach',
    (4, 4): 'approach',
    (5, 1): 'ask_for_help',
    (5, 2): 'ask_for_help',
    (5, 3): 'placate',
    (5, 4): 'placate',
    (6, 1): 'attack',
    (6, 2): 'seek_control',
    (6, 3): 'attack',
    (6, 4): 'seek_control',
}

# ─── Prosocial safety mappings ─────────────────────────────────────────────────

SAFETY_TO_EVENT: dict[str, str] = {
    '__needs_intervention__': 'danger',
    '__needs_caution__':       'criticism',
    '__casual__':              'neutral',
}

SAFETY_TO_ACTION: dict[str, str] = {
    '__needs_intervention__': 'self_protect',
    '__needs_caution__':       'analyze',
    '__casual__':              'approach',
}

# ─── Idea attractor mappings ───────────────────────────────────────────────────

ATTRACTOR_QUALITY_TO_EVENT: dict[str, str] = {
    'wise':             'neutral',
    'mixed':            'uncertainty',
    'mobilizing':       'loss_of_control',
    'misleading':       'uncertainty',
    'false_or_harmful': 'danger',
}

ATTRACTOR_QUALITY_TO_ACTION: dict[str, str] = {
    'wise':             'analyze',
    'mixed':            'reframe',
    'mobilizing':       'seek_control',
    'misleading':       'avoid',
    'false_or_harmful': 'self_protect',
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_tuple(
    text: str,
    event_label: int | None,
    action_target: int | None,
    feedback: float = 0.0,
    genome_id: str = 'default',
    session_id: str = 'dataset',
) -> TrainingTuple:
    import uuid
    t = TrainingTuple(
        session_id=session_id,
        turn_id=str(uuid.uuid4())[:8],
        timestamp=time.time(),
        text=text,
        genome_id=genome_id,
    )
    t.event_label   = event_label
    t.action_target = action_target
    t.feedback      = feedback
    return t


def _split_indices(n: int, train_frac: float = 0.8, seed: int = 42) -> tuple[list[int], list[int]]:
    import random
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    cut = int(n * train_frac)
    return indices[:cut], indices[cut:]


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_empathetic_dialogues(
    split: str = 'train',
    max_examples: int = 5000,
) -> list[TrainingTuple]:
    """
    EmpatheticDialogues CSV → TrainingTuple[].
    Uses the conversation-level emotion context for both event_label and action_target.
    Only processes speaker_idx=0 utterances (the ones sharing experience — the "user").
    """
    fname = {'train': 'train.csv', 'test': 'test.csv', 'valid': 'valid.csv'}.get(split, 'train.csv')
    path = _DATASETS / 'empatheticdialogues' / fname
    if not path.exists():
        return []

    tuples: list[TrainingTuple] = []
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(tuples) >= max_examples:
                break
            utterance = str(row.get('utterance') or '').replace('_comma_', ',').strip()
            emotion   = str(row.get('context') or '').strip().lower()
            if not utterance or not emotion:
                continue
            event_name  = EMOTION_TO_EVENT.get(emotion)
            action_name = EMOTION_TO_ACTION.get(emotion)
            if event_name is None or action_name is None:
                continue
            tuples.append(_make_tuple(
                text         = utterance,
                event_label  = EVENT_IDX.get(event_name),
                action_target= ACTION_IDX.get(action_name),
                session_id   = f'empathetic_{split}',
            ))
    return tuples


def load_archive_dialogues(
    split: str = 'train',
    max_examples: int = 3000,
) -> list[TrainingTuple]:
    """
    Archive/DailyDialog CSV → TrainingTuple[].
    Extracts the last utterance from each dialog with its emotion/act labels.
    """
    fname = {'train': 'train.csv', 'test': 'test.csv', 'valid': 'validation.csv'}.get(split, 'train.csv')
    path = _DATASETS / 'archive' / fname
    if not path.exists():
        return []

    tuples: list[TrainingTuple] = []
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(tuples) >= max_examples:
                break
            try:
                dialog_raw = str(row.get('dialog') or '')
                acts_raw   = str(row.get('act') or '')
                emos_raw   = str(row.get('emotion') or '')

                # Parse lists of int
                acts = list(map(int, re.findall(r'\d+', acts_raw)))
                emos = list(map(int, re.findall(r'\d+', emos_raw)))
                if not acts or not emos:
                    continue

                # Extract individual utterances from the dialog string
                utterances = re.findall(r"'([^']+)'", dialog_raw)
                if not utterances:
                    utterances = [dialog_raw[:200]]

                # Use last few utterances
                for i, utt in enumerate(utterances[-4:]):
                    utt = utt.strip()
                    if len(utt) < 10:
                        continue
                    idx = min(i, len(emos) - 1)
                    emotion = emos[idx]
                    act     = acts[idx] if idx < len(acts) else acts[-1]
                    event_name  = ARCHIVE_EMOTION_TO_EVENT.get(emotion, 'neutral')
                    action_name = _ARCHIVE_EMOTION_ACT_TO_ACTION.get(
                        (emotion, act),
                        'analyze',
                    )
                    tuples.append(_make_tuple(
                        text         = utt,
                        event_label  = EVENT_IDX.get(event_name),
                        action_target= ACTION_IDX.get(action_name),
                        session_id   = f'archive_{split}',
                    ))
                    if len(tuples) >= max_examples:
                        break
            except Exception:
                continue
    return tuples


def load_prosocial(max_examples: int = 500) -> list[TrainingTuple]:
    """Prosocial DialogStudio → safety-critical training tuples."""
    path = _DATASETS / 'DialogStudio' / 'Prosocial_converted_examples.json'
    if not path.exists():
        return []

    tuples: list[TrainingTuple] = []
    with path.open(encoding='utf-8') as f:
        data = json.load(f)

    for ex in data.values():
        if len(tuples) >= max_examples:
            break
        for turn in ex.get('log', []):
            if len(tuples) >= max_examples:
                break
            text = str(turn.get('user utterance') or '').strip()
            sl   = str(turn.get('original user side information', {}).get('safety_label') or '').strip()
            if not text or not sl:
                continue
            event_name  = SAFETY_TO_EVENT.get(sl)
            action_name = SAFETY_TO_ACTION.get(sl)
            if event_name is None:
                continue
            # Safety-critical examples get positive feedback weight
            feedback = 1.0 if sl == '__needs_intervention__' else 0.5
            tuples.append(_make_tuple(
                text         = text,
                event_label  = EVENT_IDX.get(event_name),
                action_target= ACTION_IDX.get(action_name),
                feedback     = feedback,
                session_id   = 'prosocial',
            ))
    return tuples


def load_idea_attractors() -> list[TrainingTuple]:
    """Idea attractors seed → training tuples based on quality label."""
    path = _DATASETS / 'idea_attractors' / 'idea_attractors_seed.jsonl'
    if not path.exists():
        return []

    tuples: list[TrainingTuple] = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line.strip())
            text    = str(row.get('text') or '').strip()
            quality = str(row.get('quality_label') or '').strip()
            if not text or not quality:
                continue
            event_name  = ATTRACTOR_QUALITY_TO_EVENT.get(quality)
            action_name = ATTRACTOR_QUALITY_TO_ACTION.get(quality)
            if event_name is None:
                continue
            feedback = 1.0 if quality == 'false_or_harmful' else 0.5
            tuples.append(_make_tuple(
                text         = text,
                event_label  = EVENT_IDX.get(event_name),
                action_target= ACTION_IDX.get(action_name),
                feedback     = feedback,
                session_id   = 'idea_attractors',
            ))
    return tuples


# ─── Full dataset builder ─────────────────────────────────────────────────────

def build_dataset(
    split: str = 'train',
    max_empathetic: int = 5000,
    max_archive: int = 2000,
    include_safety: bool = True,
    include_attractors: bool = True,
) -> list[TrainingTuple]:
    """
    Build merged training/test dataset from all sources.
    Returns shuffled TrainingTuple[] with event_label and action_target filled.
    """
    import random
    tuples: list[TrainingTuple] = []
    tuples += load_empathetic_dialogues(split=split,  max_examples=max_empathetic)
    tuples += load_archive_dialogues(split=split,     max_examples=max_archive)
    if include_safety:
        tuples += load_prosocial()
    if include_attractors:
        tuples += load_idea_attractors()

    # Deterministic shuffle so train/test are stable
    rng = random.Random(42 if split == 'train' else 99)
    rng.shuffle(tuples)
    return tuples


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_p1(
    rt: CognitiveRuntime,
    tuples: list[TrainingTuple],
) -> dict:
    """
    Evaluate P1 (EventEncoder) accuracy on labelled examples.
    Returns accuracy + per-event breakdown.
    """
    import numpy as np
    labelled = [t for t in tuples if t.event_label is not None and t.text]
    if not labelled:
        return {'accuracy': 0.0, 'n': 0}

    correct = 0
    per_event: dict[str, dict] = {}
    for t in labelled:
        probs, _ = rt.p1.encode(t.text)
        predicted = int(np.argmax(probs))
        hit = predicted == t.event_label
        if hit:
            correct += 1
        name = EVENT_TYPES[t.event_label]
        if name not in per_event:
            per_event[name] = {'correct': 0, 'total': 0}
        per_event[name]['total'] += 1
        if hit:
            per_event[name]['correct'] += 1

    accuracy = correct / len(labelled)
    per_event_acc = {
        k: round(v['correct'] / v['total'], 3)
        for k, v in per_event.items() if v['total'] > 0
    }
    return {
        'accuracy': round(accuracy, 4),
        'n': len(labelled),
        'correct': correct,
        'per_event': per_event_acc,
    }


def evaluate_p6(
    rt: CognitiveRuntime,
    tuples: list[TrainingTuple],
    genome: PersonalityGenome | None = None,
) -> dict:
    """
    Evaluate full pipeline (P1→P6) accuracy on action targets.
    Runs forward pass for each example and checks if action_name matches target.
    """
    import numpy as np
    labelled = [t for t in tuples if t.action_target is not None and t.text]
    if not labelled:
        return {'accuracy': 0.0, 'n': 0}

    if genome is None:
        genome = PersonalityGenome('default')

    reg = RegulatorState.from_genome(genome)
    correct = 0
    per_action: dict[str, dict] = {}

    for t in labelled:
        try:
            cog_out, _ = rt.forward(t.text, genome, reg)
            predicted = cog_out.action_id
            hit = predicted == t.action_target
            if hit:
                correct += 1
            name = ACTION_FAMILIES[t.action_target]
            if name not in per_action:
                per_action[name] = {'correct': 0, 'total': 0}
            per_action[name]['total'] += 1
            if hit:
                per_action[name]['correct'] += 1
        except Exception:
            pass

    accuracy = correct / max(1, len(labelled))
    per_action_acc = {
        k: round(v['correct'] / v['total'], 3)
        for k, v in per_action.items() if v['total'] > 0
    }
    return {
        'accuracy': round(accuracy, 4),
        'n': len(labelled),
        'correct': correct,
        'per_action': per_action_acc,
    }


def evaluate_event_guided_actions(
    rt: CognitiveRuntime,
    tuples: list[TrainingTuple],
) -> dict:
    """
    Event-guided action accuracy: P1 detects event → lookup known best action.

    This evaluates whether P1 has learned to detect events well enough that
    applying the EMOTION_TO_ACTION mapping to P1 predictions produces the
    correct action target. Bypasses P2-P6 intentionally.

    This is the valid metric when GenomeStore has only default persona (all 0.5
    params) — P2-P5 produce near-constant vectors with uniform genome, so P6
    cannot be measured meaningfully.
    """
    import numpy as np
    from .cognitive_pipeline import EVENT_TYPES as ET, ACTION_IDX as AI

    # Build event → action lookup from EMOTION_TO_EVENT / EMOTION_TO_ACTION tables
    EVENT_TO_ACTION: dict[str, str] = {}
    for emo, evt in EMOTION_TO_EVENT.items():
        act = EMOTION_TO_ACTION.get(emo)
        if act:
            EVENT_TO_ACTION.setdefault(evt, act)  # first mapping wins

    labelled = [t for t in tuples if t.event_label is not None and t.action_target is not None and t.text]
    if not labelled:
        return {'accuracy': 0.0, 'n': 0}

    correct = 0
    for t in labelled:
        probs, _ = rt.p1.encode(t.text)
        pred_event = ET[int(np.argmax(probs))]
        pred_action = EVENT_TO_ACTION.get(pred_event)
        if pred_action and AI.get(pred_action) == t.action_target:
            correct += 1

    accuracy = correct / len(labelled)
    return {
        'accuracy': round(accuracy, 4),
        'n': len(labelled),
        'correct': correct,
        'note': 'P1-predicted event → action lookup; bypasses P2-P6',
    }


def evaluate_p6_diversity(
    rt: CognitiveRuntime,
    tuples: list[TrainingTuple],
    genome: 'PersonalityGenome',
    n_samples: int = 100,
) -> dict:
    """
    Check that P6 predicts diverse actions (not collapsed to a single one).
    With uniform genome, P6 may collapse — this detects that condition.
    """
    import numpy as np
    from .cognitive_pipeline import RegulatorState

    sampled = [t for t in tuples if t.text][:n_samples]
    if not sampled:
        return {'unique_actions': 0, 'collapsed': True}

    predicted = set()
    for t in sampled:
        try:
            cog_out, _ = rt.forward(t.text, genome, RegulatorState.from_genome(genome))
            predicted.add(cog_out.action_name)
        except Exception:
            pass

    return {
        'unique_actions': len(predicted),
        'actions_seen':   sorted(predicted),
        'collapsed':      len(predicted) <= 1,
    }


def label_distribution(tuples: list[TrainingTuple]) -> dict:
    """Return event_label and action_target distributions for inspection."""
    from collections import Counter
    events  = Counter(EVENT_TYPES[t.event_label]  for t in tuples if t.event_label  is not None)
    actions = Counter(ACTION_FAMILIES[t.action_target] for t in tuples if t.action_target is not None)
    return {
        'total': len(tuples),
        'event_counts':  dict(events.most_common()),
        'action_counts': dict(actions.most_common()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE DATASET  (P-vector training records)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoordinateTrainingRecord:
    source: str
    text: str
    role: str
    context_matrix: list[dict[str, Any]]
    vector: dict[str, dict[str, Any]]
    confidence: float
    session_id: str
    persona_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'source': self.source,
            'text': self.text,
            'role': self.role,
            'context_matrix': self.context_matrix,
            'vector': self.vector,
            'confidence': self.confidence,
            'session_id': self.session_id,
            'persona_name': self.persona_name,
        }


# ─── Emotion → partial P-vector mappings ──────────────────────────────────────

# EmpatheticDialogues 31 emotions → {P_id: label} partial overrides
EMOTION_TO_PVECTOR: dict[str, dict[str, str]] = {
    'angry':        {'F13': 'attack', 'F22': 'dominance', 'F35': 'escalation', 'F15': 'tense'},
    'furious':      {'F13': 'attack', 'F22': 'dominance', 'F35': 'escalation', 'F15': 'overloaded'},
    'afraid':       {'F12': 'vulnerability', 'F11': 'defense', 'F15': 'tense'},
    'terrified':    {'F12': 'vulnerability', 'F37': 'rupture', 'F15': 'overloaded'},
    'caring':       {'F16': 'care', 'F20': 'friendliness', 'F32': 'approach'},
    'disgusted':    {'F18': 'devaluation', 'F33': 'distancing'},
    'annoyed':      {'F13': 'attack', 'F15': 'tense', 'F33': 'distancing'},
    'sad':          {'F12': 'vulnerability', 'F33': 'distancing'},
    'lonely':       {'F12': 'vulnerability', 'F33': 'distancing', 'F37': 'rupture'},
    'grateful':     {'F16': 'care', 'F20': 'friendliness', 'F32': 'approach', 'F17': 'respect'},
    'joyful':       {'F20': 'friendliness', 'F32': 'approach', 'F9': 'confidence'},
    'excited':      {'F20': 'friendliness', 'F32': 'approach', 'F9': 'confidence'},
    'impressed':    {'F17': 'respect', 'F32': 'approach'},
    'ashamed':      {'F12': 'vulnerability', 'F42': 'admission', 'F11': 'defense'},
    'guilty':       {'F12': 'vulnerability', 'F42': 'admission'},
    'embarrassed':  {'F12': 'vulnerability', 'F11': 'defense'},
    'trusting':     {'F17': 'respect', 'F40': 'sincerity', 'F32': 'approach'},
    'faithful':     {'F40': 'sincerity', 'F39': 'contact_maintenance', 'F32': 'approach'},
    'jealous':      {'F21': 'hidden_hostility', 'F33': 'distancing', 'F36': 'sharpening'},
    'anxious':      {'F8': 'doubt', 'F12': 'vulnerability', 'F15': 'tense'},
    'apprehensive': {'F8': 'doubt', 'F15': 'tense'},
    'disappointed': {'F18': 'devaluation', 'F33': 'distancing', 'F12': 'vulnerability'},
    'hopeful':      {'F9': 'confidence', 'F32': 'approach'},
    'confident':    {'F9': 'confidence'},
    'proud':        {'F9': 'confidence', 'F22': 'dominance'},
    'anticipating': {'F9': 'confidence', 'F32': 'approach'},
    'prepared':     {'F9': 'confidence'},
    'surprised':    {'F15': 'tense', 'F8': 'doubt'},
    'content':      {'F15': 'calm', 'F20': 'friendliness'},
    'nostalgic':    {'F44': 'reframing', 'F40': 'sincerity'},
    'sentimental':  {'F40': 'sincerity', 'F12': 'vulnerability'},
}

# Archive DailyDialog: act (1-4) → P1 label
ARCHIVE_ACT_TO_P1: dict[int, str] = {
    1: 'statement',   # inform
    2: 'question',    # question
    3: 'directive',   # directive
    4: 'statement',   # commissive
}

# Archive DailyDialog: emotion (0-6) → partial P-vector
ARCHIVE_EMOTION_TO_PVECTOR: dict[int, dict[str, str]] = {
    0: {},
    1: {'F18': 'devaluation', 'F33': 'distancing'},
    2: {'F15': 'tense'},
    3: {'F12': 'vulnerability', 'F15': 'tense'},
    4: {'F20': 'friendliness', 'F32': 'approach'},
    5: {'F12': 'vulnerability', 'F33': 'distancing'},
    6: {'F13': 'attack', 'F22': 'dominance', 'F15': 'tense'},
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _partial_pvector(partial: dict[str, str]) -> dict[str, dict[str, Any]]:
    from .message_vector_runtime import empty_coordinate_vector
    vector = empty_coordinate_vector()
    for pid, label in partial.items():
        if pid in vector:
            vector[pid] = {'main': label, 'extra': []}
    return vector


_SESSION_TURN_RE = re.compile(r'^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?\]$')


def _parse_session_file(path: Path) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    current_role: str | None = None
    current_lines: list[str] = []

    for raw_line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = raw_line.rstrip()
        if _SESSION_TURN_RE.match(line.strip()):
            if current_role and current_lines:
                turns.append((current_role, '\n'.join(current_lines).strip()))
            current_role = None
            current_lines = []
        elif current_role is None:
            if line.startswith('user: '):
                current_role = 'user'
                current_lines = [line[6:].strip()]
            elif line.startswith('assistant: '):
                current_role = 'assistant'
                current_lines = [line[11:].strip()]
        else:
            current_lines.append(line)

    if current_role and current_lines:
        turns.append((current_role, '\n'.join(current_lines).strip()))

    return [(r, t) for r, t in turns if len(t.strip()) >= 5]


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_session_coordinate_vectors(
    max_records: int = 300,
) -> list[CoordinateTrainingRecord]:
    """Parse session .txt files → bootstrap P-vectors via MessageVectorRuntime."""
    try:
        from .message_vector_runtime import MessageVectorRuntime
    except Exception:
        return []

    try:
        mvr = MessageVectorRuntime()
    except Exception:
        return []

    records: list[CoordinateTrainingRecord] = []
    runtime_root = Path(__file__).parent.parent / 'runtime'
    session_files = sorted(runtime_root.rglob('sessions/*.txt'))

    for sf in session_files:
        if len(records) >= max_records:
            break
        session_id = sf.stem
        try:
            turns = _parse_session_file(sf)
        except Exception:
            continue

        context_matrix: list[dict[str, Any]] = []
        for role, text in turns:
            if len(records) >= max_records:
                break
            try:
                vector = mvr.predict_vector(
                    text=text,
                    role=role,
                    context_matrix=context_matrix,
                )
                records.append(CoordinateTrainingRecord(
                    source='session',
                    text=text[:500],
                    role=role,
                    context_matrix=[
                        {'role': c['role'], 'text': str(c.get('text') or '')[:200], 'vector': c.get('vector', {})}
                        for c in context_matrix[-4:]
                    ],
                    vector=vector,
                    confidence=0.55,
                    session_id=session_id,
                    persona_name='',
                ))
                context_matrix.append({'role': role, 'text': text[:300], 'vector': vector})
                if len(context_matrix) > 8:
                    context_matrix = context_matrix[-8:]
            except Exception:
                continue

    return records


def load_empathetic_coordinate_vectors(
    split: str = 'train',
    max_records: int = 3000,
) -> list[CoordinateTrainingRecord]:
    """EmpatheticDialogues emotion labels → partial P-vectors."""
    fname = {'train': 'train.csv', 'test': 'test.csv', 'valid': 'valid.csv'}.get(split, 'train.csv')
    path = _DATASETS / 'empatheticdialogues' / fname
    if not path.exists():
        return []

    records: list[CoordinateTrainingRecord] = []
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(records) >= max_records:
                break
            utterance = str(row.get('utterance') or '').replace('_comma_', ',').strip()
            emotion   = str(row.get('context') or '').strip().lower()
            if not utterance or not emotion:
                continue
            partial = EMOTION_TO_PVECTOR.get(emotion)
            if partial is None:
                continue
            vector = _partial_pvector(partial)
            records.append(CoordinateTrainingRecord(
                source='empathetic',
                text=utterance[:500],
                role='user',
                context_matrix=[],
                vector=vector,
                confidence=0.70,
                session_id=f'empathetic_{split}',
                persona_name='',
            ))
    return records


def load_archive_coordinate_vectors(
    split: str = 'train',
    max_records: int = 2000,
) -> list[CoordinateTrainingRecord]:
    """DailyDialog act+emotion → partial P-vectors."""
    fname = {'train': 'train.csv', 'test': 'test.csv', 'valid': 'validation.csv'}.get(split, 'train.csv')
    path = _DATASETS / 'archive' / fname
    if not path.exists():
        return []

    records: list[CoordinateTrainingRecord] = []
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(records) >= max_records:
                break
            try:
                dialog_raw = str(row.get('dialog') or '')
                acts_raw   = str(row.get('act') or '')
                emos_raw   = str(row.get('emotion') or '')
                acts = list(map(int, re.findall(r'\d+', acts_raw)))
                emos = list(map(int, re.findall(r'\d+', emos_raw)))
                if not acts or not emos:
                    continue
                utterances = re.findall(r"'([^']+)'", dialog_raw)
                if not utterances:
                    utterances = [dialog_raw[:200]]
                for i, utt in enumerate(utterances[-4:]):
                    utt = utt.strip()
                    if len(utt) < 10:
                        continue
                    if len(records) >= max_records:
                        break
                    idx = min(i, len(emos) - 1)
                    emotion = emos[idx]
                    act     = acts[idx] if idx < len(acts) else acts[-1]
                    p1_label = ARCHIVE_ACT_TO_P1.get(act, 'statement')
                    partial: dict[str, str] = {'F1': p1_label}
                    partial.update(ARCHIVE_EMOTION_TO_PVECTOR.get(emotion, {}))
                    vector = _partial_pvector(partial)
                    records.append(CoordinateTrainingRecord(
                        source='archive',
                        text=utt[:500],
                        role='user',
                        context_matrix=[],
                        vector=vector,
                        confidence=0.65,
                        session_id=f'archive_{split}',
                        persona_name='',
                    ))
            except Exception:
                continue
    return records


def load_correction_coordinate_vectors() -> list[CoordinateTrainingRecord]:
    """Gold correction records → high-confidence P-vector labels for assistant output quality."""
    corrections_path = (
        Path(__file__).parent.parent
        / 'runtime' / 'development' / 'memory' / 'training_examples' / 'global.jsonl'
    )
    if not corrections_path.exists():
        return []

    records: list[CoordinateTrainingRecord] = []
    with corrections_path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            input_text     = str(row.get('input_text') or '').strip()
            correct_output = str(row.get('correct_output') or '').strip()
            session_id     = str(row.get('session_id') or 'corrections')
            persona_name   = str(row.get('persona_name') or '')

            if input_text:
                records.append(CoordinateTrainingRecord(
                    source='correction',
                    text=input_text[:500],
                    role='user',
                    context_matrix=[],
                    vector=_partial_pvector({'F1': 'question', 'F5': 'independent'}),
                    confidence=1.0,
                    session_id=session_id,
                    persona_name=persona_name,
                ))

            if correct_output:
                # Correct assistant response should be direct, sincere, not masking
                vector = _partial_pvector({
                    'F1': 'statement',
                    'F2': 'direct',
                    'F3': 'literal',
                    'F40': 'sincerity',
                    'F41': 'unclear',
                })
                records.append(CoordinateTrainingRecord(
                    source='correction',
                    text=correct_output[:500],
                    role='assistant',
                    context_matrix=[],
                    vector=vector,
                    confidence=1.0,
                    session_id=session_id,
                    persona_name=persona_name,
                ))
    return records


# ─── Full coordinate dataset builder ──────────────────────────────────────────

def build_coordinate_dataset(
    split: str = 'train',
    include_sessions: bool = True,
    include_empathetic: bool = True,
    include_archive: bool = True,
    include_corrections: bool = True,
    max_sessions: int = 300,
    max_empathetic: int = 3000,
    max_archive: int = 2000,
) -> list[CoordinateTrainingRecord]:
    """
    Build a labeled coordinate-vector dataset from all available sources.
    Returns CoordinateTrainingRecord[] ready for P-module training.

    Each record has:
      - text / role / context_matrix
      - vector: full {P_id: {main, extra}} dict (partial coords set from label mapping,
        rest at defaults from empty_coordinate_vector)
      - confidence: 0.55 (session bootstrap) / 0.65 (archive) / 0.70 (empathetic) / 1.0 (gold)
    """
    import random
    records: list[CoordinateTrainingRecord] = []

    if include_corrections:
        records += load_correction_coordinate_vectors()

    if include_sessions:
        records += load_session_coordinate_vectors(max_records=max_sessions)

    if include_empathetic:
        records += load_empathetic_coordinate_vectors(split=split, max_records=max_empathetic)

    if include_archive:
        records += load_archive_coordinate_vectors(split=split, max_records=max_archive)

    rng = random.Random(42 if split == 'train' else 99)
    rng.shuffle(records)
    return records


def coordinate_dataset_stats(records: list[CoordinateTrainingRecord]) -> dict[str, Any]:
    """Summarise a CoordinateTrainingRecord list: source breakdown + P-coord distributions."""
    from collections import Counter, defaultdict
    sources: Counter = Counter(r.source for r in records)
    roles:   Counter = Counter(r.role   for r in records)
    by_coord: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        for pid, choice in r.vector.items():
            main = str((choice or {}).get('main') or '')
            if main:
                by_coord[pid][main] += 1
    top_coords = {
        pid: dict(counter.most_common(5))
        for pid, counter in sorted(by_coord.items())
    }
    return {
        'total': len(records),
        'by_source': dict(sources),
        'by_role':   dict(roles),
        'top_labels_per_coord': top_coords,
    }
