from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from .message_vector_registry import P_REGISTRY, P_REGISTRY_BY_ID, CoordinateInterpreterSpec
from .runtime_config import get_runtime_config

_MODEL_LOCK = Lock()
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁёԱ-Ֆա-ֆ一-龥_'-]+", re.UNICODE)
_HASH_DIM = 224
_VECTOR_DIM = 256
_STRUCT_OFFSET = _HASH_DIM
_CONTEXT_WINDOW = 4
_BOOTSTRAP_VERSION = 'message_vector_bootstrap_v1'

_QUESTION_WORDS = {'why', 'what', 'who', 'how', 'where', 'when', 'зачем', 'почему', 'что', 'кто', 'как', 'когда', 'где'}
_DIRECTIVE_CUES = {'must', 'should', 'need', 'stop', 'listen', 'должен', 'надо', 'нужно', 'перестань', 'слушай'}
_DOUBT_CUES = {'maybe', 'perhaps', 'not sure', 'сомневаюсь', 'может', 'кажется', 'не знаю'}
_CONFIDENCE_CUES = {'definitely', 'sure', 'exactly', 'точно', 'уверен', 'конечно'}
_CARE_CUES = {'care', 'support', 'here for you', 'забочусь', 'поддержу', 'береги', 'понимаю'}
_ATTACK_CUES = {'stupid', 'idiot', 'worthless', 'дурак', 'идиот', 'тупой', 'жалкий'}
_DEVALUE_CUES = {'nothing', 'nobody', 'useless', 'ерунда', 'ничего не стоишь', 'пустяк'}
_RESPECT_CUES = {'respect', 'value', 'уважаю', 'ценю'}
_FRIENDLY_CUES = {'hi', 'hello', 'thanks', 'привет', 'спасибо', 'рад', 'приятно'}
_SARCASM_CUES = {'yeah right', 'sure', 'ну да', 'ага конечно', 'молодец', 'браво'}
_MANIPULATION_CUES = {'you owe', 'after all i', 'если бы ты', 'после всего что я', 'должен мне'}
_PRESSURE_CUES = {'now', 'answer me', 'immediately', 'сейчас', 'ответь', 'немедленно'}
_RECONCILE_CUES = {'sorry', 'let us fix', 'давай нормально', 'извини', 'помиримся'}
_DISTANCE_CUES = {'leave me alone', 'go away', 'отстань', 'уйди', 'не пиши'}
_SOFTEN_CUES = {'please', 'maybe', 'пожалуйста', 'если хочешь'}
_HONESTY_CUES = {'honestly', 'sincerely', 'честно', 'искренне'}
_MASK_CUES = {'just kidding', 'без обид', 'просто шучу'}
_PRAISE_CUES = {'great', 'good job', 'well done', 'молодец', 'умница', 'браво'}
_ADMISSION_CUES = {'i was wrong', 'my mistake', 'я был неправ', 'признаю'}
_DENIAL_CUES = {'no', 'not true', 'неправда', 'нет', 'не было'}
_REFRAME_CUES = {'actually', 'rather', 'иначе', 'скорее', 'на самом деле'}
_REPROACH_CUES = {'after all', 'again', 'опять', 'как всегда', 'после всего'}
_CONTAINMENT_CUES = {'calm down', 'let us stay', 'спокойно', 'давай без'}
_VULNERABLE_CUES = {'hurt', 'afraid', 'scared', 'больно', 'страшно', 'тяжело'}
_CONCESSION_CUES = {'okay', 'fine', 'ладно', 'хорошо, пусть'}
_THOUGHT_CUES = {'i think', 'мне кажется', 'думаю', 'thought'}
_QUOTE_CUES = {'"', "'", '«', '»'}

# P50/P51: vector values that don't carry topic identity (structural/neutral labels)
_CONTEXT_TOPIC_NEUTRAL = frozenset({
    'absent', 'neutral', 'unclear', 'continuation', 'defined', 'direct', 'literal',
    'statement', 'non_answer', 'independent', 'opening_new_direction', 'calm',
    'no_change', 'full_window', 'last_turn_only',
})


@dataclass(slots=True)
class CoordinatePrediction:
    main: str
    extra: list[str]
    source: str
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            'main': self.main,
            'extra': list(self.extra),
            'source': self.source,
            'scores': dict(self.scores),
        }


def _models_path() -> Path:
    path = get_runtime_config().paths.memory_root / 'message_vector_models'
    path.mkdir(parents=True, exist_ok=True)
    return path / 'p_registry_v1.json'


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _stable_slot(token: str) -> int:
    digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).hexdigest()
    return int(digest, 16) % _HASH_DIM


def _normalized_text(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip().lower()


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalized_text(text))


def _contains_any(text: str, markers: set[str]) -> bool:
    lowered = _normalized_text(text)
    return any(marker in lowered for marker in markers)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _default_main(interpreter_id: str) -> str:
    defaults = {
        'P1': 'statement',
        'P2': 'direct',
        'P3': 'literal',
        'P4': 'non_answer',
        'P5': 'independent',
        'P6': 'opening_new_direction',
        'P7': 'defined',
        'P8': 'absent',
        'P9': 'absent',
        'P10': 'absent',
        'P11': 'absent',
        'P12': 'absent',
        'P13': 'absent',
        'P14': 'absent',
        'P15': 'calm',
        'P16': 'absent',
        'P17': 'absent',
        'P18': 'absent',
        'P19': 'absent',
        'P20': 'absent',
        'P21': 'absent',
        'P22': 'neutral',
        'P23': 'neutral',
        'P24': 'absent',
        'P25': 'absent',
        'P26': 'absent',
        'P27': 'absent',
        'P28': 'absent',
        'P29': 'absent',
        'P30': 'absent',
        'P31': 'absent',
        'P32': 'neutral',
        'P33': 'neutral',
        'P34': 'neutral',
        'P35': 'neutral',
        'P36': 'neutral',
        'P37': 'neutral',
        'P38': 'neutral',
        'P39': 'neutral',
        'P40': 'unclear',
        'P41': 'unclear',
        'P42': 'absent',
        'P43': 'absent',
        'P44': 'absent',
        'P45': 'absent',
        'P46': 'absent',
        'P47': 'absent',
        'P48': 'continuation',
        'P49': 'neutral',
    }
    return defaults.get(interpreter_id, P_REGISTRY_BY_ID[interpreter_id].options[0])


def empty_coordinate_vector() -> dict[str, dict[str, Any]]:
    return {
        spec.interpreter_id: {
            'main': _default_main(spec.interpreter_id),
            'extra': [],
        }
        for spec in P_REGISTRY
    }


def normalize_coordinate_choice(interpreter_id: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    spec = P_REGISTRY_BY_ID[interpreter_id]
    data = dict(payload or {})
    main = str(data.get('main') or _default_main(interpreter_id)).strip()
    if main not in spec.options:
        main = _default_main(interpreter_id)
    extra = []
    for item in list(data.get('extra') or []):
        label = str(item or '').strip()
        if label and label != main and label in spec.options and label not in extra:
            extra.append(label)
    return {'main': main, 'extra': extra}


def normalize_coordinate_vector(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    vector = empty_coordinate_vector()
    if not isinstance(payload, dict):
        return vector
    for interpreter_id in vector:
        if interpreter_id in payload:
            vector[interpreter_id] = normalize_coordinate_choice(interpreter_id, payload.get(interpreter_id))
    return vector


class MessageVectorRuntime:
    def __init__(self) -> None:
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        path = _models_path()
        if not path.exists():
            return {'version': 1, 'bootstrap_version': _BOOTSTRAP_VERSION, 'interpreters': {}}
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            payload = {}
        interpreters = payload.get('interpreters') if isinstance(payload, dict) else {}
        return {
            'version': 1,
            'bootstrap_version': _BOOTSTRAP_VERSION,
            'interpreters': dict(interpreters) if isinstance(interpreters, dict) else {},
        }

    def _save_state(self) -> None:
        path = _models_path()
        path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding='utf-8')

    def build_feature_vector(
        self,
        *,
        text: str,
        role: str,
        context_matrix: list[dict[str, Any]],
        persona_name: str = '',
    ) -> np.ndarray:
        features = np.zeros((_VECTOR_DIM,), dtype=np.float32)
        cleaned = _normalized_text(text)
        tokens = _tokens(cleaned)
        for index, token in enumerate(tokens):
            features[_stable_slot(f'tok:{token}')] += 1.0
            if index + 1 < len(tokens):
                features[_stable_slot(f'big:{token}|{tokens[index + 1]}')] += 0.35
        if persona_name:
            features[_stable_slot(f'persona:{_normalized_text(persona_name)}')] += 0.3
        for offset, item in enumerate(context_matrix[-_CONTEXT_WINDOW:]):
            item_role = str(item.get('role') or '').strip().lower() or 'unknown'
            features[_stable_slot(f'ctx-role:{offset}:{item_role}')] += 0.2
            for interpreter_id, choice in dict(item.get('vector') or {}).items():
                main = str((choice or {}).get('main') or '').strip()
                if main:
                    features[_stable_slot(f'ctx:{offset}:{interpreter_id}:{main}')] += 0.4
                for extra in list((choice or {}).get('extra') or [])[:2]:
                    label = str(extra or '').strip()
                    if label:
                        features[_stable_slot(f'ctxx:{offset}:{interpreter_id}:{label}')] += 0.18
        features[_STRUCT_OFFSET + 0] = min(len(tokens) / 18.0, 1.0)
        features[_STRUCT_OFFSET + 1] = 1.0 if '?' in text else 0.0
        features[_STRUCT_OFFSET + 2] = min(text.count('!') / 3.0, 1.0)
        features[_STRUCT_OFFSET + 3] = 1.0 if any(mark in text for mark in _QUOTE_CUES) else 0.0
        features[_STRUCT_OFFSET + 4] = 1.0 if role == 'user' else 0.0
        features[_STRUCT_OFFSET + 5] = 1.0 if role == 'assistant' else 0.0
        features[_STRUCT_OFFSET + 6] = min(len(context_matrix) / float(_CONTEXT_WINDOW), 1.0)
        features[_STRUCT_OFFSET + 7] = min(sum(1 for token in tokens if token in _QUESTION_WORDS) / 2.0, 1.0)
        features[_STRUCT_OFFSET + 8] = min(sum(1 for token in tokens if token in {'i', 'я'}) / 2.0, 1.0)
        features[_STRUCT_OFFSET + 9] = min(sum(1 for token in tokens if token in {'you', 'ты', 'тебе', 'твой'}) / 2.0, 1.0)
        features[_STRUCT_OFFSET + 10] = min(sum(1 for ch in text if ch.isupper()) / max(len(text), 1), 1.0)
        norm = float(np.linalg.norm(features))
        if norm > 1e-8:
            features /= norm
        return features

    def _bootstrap_prediction(
        self,
        spec: CoordinateInterpreterSpec,
        *,
        text: str,
        role: str,
        context_matrix: list[dict[str, Any]],
        partial_vector: dict[str, dict[str, Any]],
    ) -> CoordinatePrediction:
        lowered = _normalized_text(text)
        last_context = context_matrix[-1] if context_matrix else {}
        last_vector = dict(last_context.get('vector') or {})
        hostile_context = any(
            str((last_vector.get(pid) or {}).get('main') or '') in {'attack', 'devaluation', 'humiliation', 'escalation', 'sharpening', 'toward_escalation'}
            for pid in ('P13', 'P18', 'P19', 'P35', 'P36', 'P49')
        )
        positive_mask = _contains_any(lowered, _PRAISE_CUES | _CARE_CUES | _FRIENDLY_CUES)
        movement_labels = {
            str((partial_vector.get(pid) or {}).get('main') or '')
            for pid in ('P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'P38', 'P39')
        }

        def base(main: str, *extra: str) -> CoordinatePrediction:
            allowed_extra = [label for label in extra if label and label != main and label in spec.options]
            return CoordinatePrediction(main=main, extra=allowed_extra, source='bootstrap', scores={main: 1.0})

        pid = spec.interpreter_id
        if pid == 'P1':
            if any(mark in text for mark in ('"', '«', '»')):
                return base('quote')
            if '?' in text or any(token in _QUESTION_WORDS for token in _tokens(lowered)):
                return base('question', 'statement' if role == 'assistant' else '')
            if _contains_any(lowered, _DIRECTIVE_CUES):
                return base('directive')
            if _contains_any(lowered, _THOUGHT_CUES):
                return base('thought')
            return base('statement')
        if pid == 'P2':
            if hostile_context and positive_mask:
                return base('masked', 'direct')
            return base('direct')
        if pid == 'P3':
            if lowered.startswith('разве') or lowered.startswith('ну ') or lowered.startswith('who even'):
                return base('rhetorical_move')
            return base('literal')
        if pid == 'P4':
            if _contains_any(lowered, {'whatever', 'неважно', 'потом'}):
                return base('avoidance')
            if context_matrix and role == 'assistant' and '?' not in text:
                return base('answer')
            return base('non_answer')
        if pid == 'P5':
            return base('reaction' if context_matrix else 'independent')
        if pid == 'P6':
            if _contains_any(lowered, {'bye', 'пока', 'ладно все', 'закрыли'}):
                return base('closing')
            return base('opening_new_direction')
        if pid == 'P7':
            return base('diffuse' if _contains_any(lowered, _DOUBT_CUES | _SOFTEN_CUES) else 'defined')
        if pid == 'P8':
            return base('doubt' if _contains_any(lowered, _DOUBT_CUES) else 'absent')
        if pid == 'P9':
            return base('confidence' if _contains_any(lowered, _CONFIDENCE_CUES) else 'absent')
        if pid == 'P10':
            return base('inner_conflict' if _contains_any(lowered, _DOUBT_CUES) and _contains_any(lowered, _CONFIDENCE_CUES | _CARE_CUES | _ATTACK_CUES) else 'absent')
        if pid == 'P11':
            return base('defense' if _contains_any(lowered, {'i just', 'я просто', 'but i', 'но я'}) or hostile_context else 'absent')
        if pid == 'P12':
            return base('vulnerability' if _contains_any(lowered, _VULNERABLE_CUES) else 'absent')
        if pid == 'P13':
            return base('attack' if _contains_any(lowered, _ATTACK_CUES | {'shut up', 'заткнись'}) else 'absent')
        if pid == 'P14':
            return base('containment' if _contains_any(lowered, _CONTAINMENT_CUES) else 'absent')
        if pid == 'P15':
            if text.count('!') >= 2 or float(sum(1 for ch in text if ch.isupper()) / max(len(text), 1)) > 0.18:
                return base('overloaded')
            if 'attack' in {str((partial_vector.get('P13') or {}).get('main') or ''), str((partial_vector.get('P30') or {}).get('main') or '')}:
                return base('tense')
            return base('calm')
        if pid == 'P16':
            return base('care' if _contains_any(lowered, _CARE_CUES) else 'absent')
        if pid == 'P17':
            return base('respect' if _contains_any(lowered, _RESPECT_CUES) else 'absent')
        if pid == 'P18':
            return base('devaluation' if _contains_any(lowered, _DEVALUE_CUES) else 'absent')
        if pid == 'P19':
            return base('humiliation' if _contains_any(lowered, {'pathetic', 'жалкий', 'ничтожество'}) else 'absent')
        if pid == 'P20':
            return base('friendliness' if _contains_any(lowered, _FRIENDLY_CUES) else 'absent')
        if pid == 'P21':
            if positive_mask and hostile_context:
                return base('hidden_hostility')
            return base('absent')
        if pid == 'P22':
            return base('dominance' if _contains_any(lowered, _DIRECTIVE_CUES | _PRESSURE_CUES) else 'neutral')
        if pid == 'P23':
            return base('concession' if _contains_any(lowered, _CONCESSION_CUES) else 'neutral')
        if pid == 'P24':
            if positive_mask and hostile_context:
                return base('false_praise', 'sarcasm')
            if _contains_any(lowered, _SARCASM_CUES):
                return base('sarcasm')
            return base('absent')
        if pid == 'P25':
            return base('irony' if _contains_any(lowered, {'ironic', 'ирония', 'конечно же'}) else 'absent')
        if pid == 'P26':
            return base('mockery' if _contains_any(lowered, {'mock', 'насмешка', 'чудик'}) else 'absent')
        if pid == 'P27':
            return base('mask_of_praise' if positive_mask and hostile_context else 'absent')
        if pid == 'P28':
            return base('mask_of_care' if _contains_any(lowered, _CARE_CUES) and hostile_context else 'absent')
        if pid == 'P29':
            return base('manipulation' if _contains_any(lowered, _MANIPULATION_CUES) else 'absent')
        if pid == 'P30':
            return base('pressure' if _contains_any(lowered, _PRESSURE_CUES) or text.count('?') >= 2 else 'absent')
        if pid == 'P31':
            return base('false_softening' if _contains_any(lowered, _MASK_CUES | {'just', 'просто'}) and any(label in movement_labels for label in {'escalation', 'sharpening'}) else 'absent')
        if pid == 'P32':
            return base('approach' if any(str((partial_vector.get(pid2) or {}).get('main') or '') in {'care', 'respect', 'friendliness'} for pid2 in ('P16', 'P17', 'P20')) else 'neutral')
        if pid == 'P33':
            return base('distancing' if _contains_any(lowered, _DISTANCE_CUES) or str((partial_vector.get('P13') or {}).get('main') or '') == 'attack' else 'neutral')
        if pid == 'P34':
            return base('reconciliation' if _contains_any(lowered, _RECONCILE_CUES) else 'neutral')
        if pid == 'P35':
            return base('escalation' if any(str((partial_vector.get(pid2) or {}).get('main') or '') in {'attack', 'pressure', 'humiliation'} for pid2 in ('P13', 'P30', 'P19')) else 'neutral')
        if pid == 'P36':
            return base('sharpening' if any(str((partial_vector.get(pid2) or {}).get('main') or '') in {'sarcasm', 'mockery', 'hidden_hostility'} for pid2 in ('P24', 'P26', 'P21')) else 'neutral')
        if pid == 'P37':
            return base('rupture' if _contains_any(lowered, _DISTANCE_CUES | {'block', 'разговор окончен'}) else 'neutral')
        if pid == 'P38':
            return base('softening' if _contains_any(lowered, _SOFTEN_CUES | _RECONCILE_CUES) else 'neutral')
        if pid == 'P39':
            return base('contact_maintenance' if context_matrix and not _contains_any(lowered, _DISTANCE_CUES) else 'neutral')
        if pid == 'P40':
            return base('sincerity' if _contains_any(lowered, _HONESTY_CUES) and str((partial_vector.get('P41') or {}).get('main') or '') != 'masking' else 'unclear')
        if pid == 'P41':
            return base('masking' if any(str((partial_vector.get(pid2) or {}).get('main') or '') in {'false_praise', 'mask_of_praise', 'mask_of_care', 'false_softening'} for pid2 in ('P24', 'P27', 'P28', 'P31')) else 'unclear')
        if pid == 'P42':
            return base('admission' if _contains_any(lowered, _ADMISSION_CUES) else 'absent')
        if pid == 'P43':
            return base('denial' if _contains_any(lowered, _DENIAL_CUES) else 'absent')
        if pid == 'P44':
            return base('reframing' if _contains_any(lowered, _REFRAME_CUES) else 'absent')
        if pid == 'P45':
            if _contains_any(lowered, _PRAISE_CUES) and any(str((partial_vector.get(pid2) or {}).get('main') or '') in {'pressure', 'containment'} for pid2 in ('P30', 'P14')):
                return base('corrective_praise')
            return base('absent')
        if pid == 'P46':
            return base('false_praise' if positive_mask and hostile_context else 'absent')
        if pid == 'P47':
            return base('hidden_reproach' if _contains_any(lowered, _REPROACH_CUES) else 'absent')
        if pid == 'P48':
            if str((partial_vector.get('P41') or {}).get('main') or '') == 'masking':
                return base('masking_shift')
            if str((partial_vector.get('P34') or {}).get('main') or '') == 'reconciliation':
                return base('repair_attempt')
            if any(str((partial_vector.get(pid2) or {}).get('main') or '') in {'escalation', 'sharpening', 'attack'} for pid2 in ('P35', 'P36', 'P13')):
                return base('conflict_reply')
            if str((partial_vector.get('P44') or {}).get('main') or '') == 'reframing':
                return base('reinterpretation')
            return base('continuation')
        if pid == 'P49':
            if str((partial_vector.get('P34') or {}).get('main') or '') == 'reconciliation' or str((partial_vector.get('P38') or {}).get('main') or '') == 'softening':
                return base('toward_repair')
            if str((partial_vector.get('P33') or {}).get('main') or '') == 'distancing' or str((partial_vector.get('P37') or {}).get('main') or '') == 'rupture':
                return base('toward_distance')
            if str((partial_vector.get('P35') or {}).get('main') or '') == 'escalation' or str((partial_vector.get('P36') or {}).get('main') or '') == 'sharpening':
                return base('toward_escalation')
            if str((partial_vector.get('P41') or {}).get('main') or '') == 'masking':
                return base('toward_masking')
            if str((partial_vector.get('P39') or {}).get('main') or '') == 'contact_maintenance' or str((partial_vector.get('P32') or {}).get('main') or '') == 'approach':
                return base('toward_contact')
            return base('neutral')
        if pid == 'P50':
            # Detect topic shift: abrupt change in semantic direction cues
            prev_main_topics = set()
            for entry in context_matrix[-2:]:
                for coord in dict(entry.get('vector') or {}).values():
                    m = str((coord or {}).get('main') or '').strip()
                    if m and m not in _CONTEXT_TOPIC_NEUTRAL:
                        prev_main_topics.add(m)
            cur_cues = set()
            for coord in partial_vector.values():
                m = str((coord or {}).get('main') or '').strip()
                if m and m not in _CONTEXT_TOPIC_NEUTRAL:
                    cur_cues.add(m)
            overlap = prev_main_topics & cur_cues
            if not context_matrix:
                return base('no_change')
            if not prev_main_topics or len(overlap) / max(len(prev_main_topics | cur_cues), 1) < 0.15:
                return base('hard_shift')
            if len(overlap) / max(len(prev_main_topics | cur_cues), 1) < 0.4:
                return base('soft_shift')
            if str((partial_vector.get('P6') or {}).get('main') or '') == 'closing':
                return base('return_to_topic')
            return base('no_change')
        if pid == 'P51':
            # Determine where relevant context starts based on P50
            p50_main = str((partial_vector.get('P50') or {}).get('main') or '')
            if p50_main == 'hard_shift':
                return base('since_topic_shift')
            if p50_main == 'soft_shift':
                return base('since_topic_shift', 'full_window')
            if p50_main == 'return_to_topic':
                return base('anchor_only')
            if not context_matrix:
                return base('last_turn_only')
            return base('full_window')
        return base(_default_main(pid))

    def _trained_prediction(
        self,
        spec: CoordinateInterpreterSpec,
        *,
        features: np.ndarray,
        bootstrap: CoordinatePrediction,
    ) -> CoordinatePrediction:
        interpreter_state = dict(self._state.get('interpreters', {}).get(spec.interpreter_id) or {})
        centroids = dict(interpreter_state.get('centroids') or {})
        scores: dict[str, float] = {}
        for label, payload in centroids.items():
            values = np.array(list(payload.get('values') or []), dtype=np.float32)
            if values.shape != features.shape:
                continue
            scores[label] = _cosine(features, values)
        if not scores:
            return bootstrap
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ranked[0]
        if best_score < 0.16:
            return bootstrap
        extra = [
            label
            for label, score in ranked[1:4]
            if label != best_label and score >= best_score - 0.08
        ][:3]
        return CoordinatePrediction(main=best_label, extra=extra, source='trained', scores=scores)

    def predict_vector(
        self,
        *,
        text: str,
        role: str,
        context_matrix: list[dict[str, Any]],
        persona_name: str = '',
    ) -> dict[str, dict[str, Any]]:
        features = self.build_feature_vector(text=text, role=role, context_matrix=context_matrix, persona_name=persona_name)
        vector: dict[str, dict[str, Any]] = {}
        for spec in P_REGISTRY:
            bootstrap = self._bootstrap_prediction(spec, text=text, role=role, context_matrix=context_matrix, partial_vector=vector)
            trained = self._trained_prediction(spec, features=features, bootstrap=bootstrap)
            vector[spec.interpreter_id] = {'main': trained.main, 'extra': list(trained.extra)}
        return normalize_coordinate_vector(vector)

    def record_annotation(
        self,
        *,
        text: str,
        role: str,
        context_matrix: list[dict[str, Any]],
        coordinates: dict[str, Any],
        persona_name: str = '',
    ) -> None:
        vector = normalize_coordinate_vector(coordinates)
        features = self.build_feature_vector(text=text, role=role, context_matrix=context_matrix, persona_name=persona_name)
        with _MODEL_LOCK:
            self._state = self._load_state()
            interpreters = self._state.setdefault('interpreters', {})
            for spec in P_REGISTRY:
                choice = vector.get(spec.interpreter_id) or {'main': _default_main(spec.interpreter_id), 'extra': []}
                interpreter_state = dict(interpreters.get(spec.interpreter_id) or {})
                centroids = dict(interpreter_state.get('centroids') or {})
                self._update_centroid(centroids, choice.get('main'), features, weight=1.0)
                for extra in list(choice.get('extra') or []):
                    self._update_centroid(centroids, extra, features, weight=0.45)
                interpreter_state['centroids'] = centroids
                interpreter_state['updated_at'] = _utc_now()
                interpreters[spec.interpreter_id] = interpreter_state
            self._state['interpreters'] = interpreters
            self._save_state()

    def _update_centroid(self, centroids: dict[str, Any], label: str, features: np.ndarray, *, weight: float) -> None:
        clean_label = str(label or '').strip()
        if not clean_label:
            return
        payload = dict(centroids.get(clean_label) or {})
        count = float(payload.get('count') or 0.0)
        values = np.array(list(payload.get('values') or []), dtype=np.float32)
        if values.shape != features.shape:
            values = np.zeros_like(features)
        next_count = count + float(weight)
        if next_count <= 0:
            return
        values = ((values * count) + (features * float(weight))) / next_count
        centroids[clean_label] = {
            'count': round(next_count, 6),
            'values': [round(float(item), 8) for item in values.tolist()],
        }

    def predict_transition_interpretation(
        self,
        *,
        context_matrix: list[dict[str, Any]],
        current_vector: dict[str, Any],
    ) -> dict[str, Any]:
        from_labels: list[str] = []
        for item in context_matrix[-_CONTEXT_WINDOW:]:
            vector = dict(item.get('vector') or {})
            for pid in ('P35', 'P36', 'P37', 'P38', 'P39', 'P49'):
                main = str((vector.get(pid) or {}).get('main') or '').strip()
                if main and main not in {'neutral', 'continuation'} and main not in from_labels:
                    from_labels.append(main)
        to_labels: list[str] = []
        for pid in ('P24', 'P27', 'P28', 'P35', 'P36', 'P38', 'P45', 'P46', 'P49'):
            main = str((current_vector.get(pid) or {}).get('main') or '').strip()
            if main and main not in {'neutral', 'absent', 'unclear', 'continuation'} and main not in to_labels:
                to_labels.append(main)
        transition_type = 'continuation'
        target = str((current_vector.get('P49') or {}).get('main') or '').strip()
        if target == 'toward_escalation' or any(label in {'escalation', 'sharpening'} for label in to_labels):
            transition_type = 'escalation'
        elif target == 'toward_repair' or any(label in {'reconciliation', 'softening'} for label in to_labels):
            transition_type = 'repair'
        elif target == 'toward_masking' or any(label in {'mask_of_praise', 'mask_of_care', 'false_praise'} for label in to_labels):
            transition_type = 'masking'
        elif target == 'toward_distance' or any(label in {'distancing', 'rupture'} for label in to_labels):
            transition_type = 'distancing'
        elif target == 'toward_contact':
            transition_type = 'contact'
        return {
            'from': from_labels[:4],
            'to': to_labels[:4],
            'type': transition_type,
        }

    def build_history_flow(
        self,
        *,
        context_matrix: list[dict[str, Any]],
        current_vector: dict[str, Any],
    ) -> list[str]:
        flow: list[str] = []
        for item in context_matrix[-_CONTEXT_WINDOW:]:
            vector = dict(item.get('vector') or {})
            for pid in ('P35', 'P36', 'P37', 'P38', 'P39', 'P49'):
                main = str((vector.get(pid) or {}).get('main') or '').strip()
                if main and main not in {'neutral', 'continuation'} and main not in flow:
                    flow.append(main)
        for pid in ('P35', 'P36', 'P38', 'P39', 'P49'):
            main = str((current_vector.get(pid) or {}).get('main') or '').strip()
            if main and main not in {'neutral', 'continuation'} and main not in flow:
                flow.append(main)
        return flow[:6]


_RUNTIME: MessageVectorRuntime | None = None


def get_message_vector_runtime() -> MessageVectorRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = MessageVectorRuntime()
    return _RUNTIME
