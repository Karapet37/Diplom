"""
annotate_datasets.py — добавляет P-family метки к существующим датасетам.

Для каждой реплики из каждого датасета:
1. Применяет существующие метки (emotion, act, context) → карта на P-семьи
2. Прогоняет правила P-реестра → дополнительные слабые сигналы
3. Выгружает training/p_examples/P{n}.jsonl — один файл на P-семью

Формат выходной строки:
    {"p_id": "P47", "variant": "hidden_reproach", "text": "...",
     "score": 0.9, "source": "empatheticdialogues", "confidence": "label_mapped"}
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# путь к проекту
PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from agent_system.p_subsystem_registry import PSubsystemRegistry

OUT_DIR = PROJECT / 'training' / 'p_examples'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_DIR = PROJECT / 'DataSets'

# ─── Карты существующих меток → P-семьи ──────────────────────────────────────

# archive/train.csv: emotion codes 0-6
ARCHIVE_EMOTION_MAP: dict[int, list[tuple[str, str, float]]] = {
    0: [],  # neutral
    1: [('P21', 'hidden_hostility', 0.65), ('P18', 'devaluation', 0.55)],   # disgust
    2: [('P8', 'doubt', 0.50), ('P7', 'diffuse', 0.40)],                    # surprise
    3: [('P12', 'vulnerability', 0.80), ('P15', 'tense', 0.70),             # fear
        ('P14', 'containment', 0.55)],
    4: [('P9', 'confidence', 0.70), ('P20', 'genuine_friendliness', 0.65),  # happy
        ('P32', 'approach', 0.55)],
    5: [('P12', 'vulnerability', 0.70), ('P33', 'protective_distancing', 0.55),  # sad
        ('P14', 'containment', 0.50)],
    6: [('P13', 'attack', 0.75), ('P15', 'overloaded', 0.65),               # angry
        ('P35', 'sudden_escalation', 0.50)],
}

# archive/train.csv: act codes 1-4
ARCHIVE_ACT_MAP: dict[int, list[tuple[str, str, float]]] = {
    1: [('P1', 'statement', 0.85)],          # inform
    2: [('P1', 'question', 0.90)],           # question
    3: [('P1', 'directive', 0.85)],          # directive
    4: [('P1', 'performative', 0.80)],       # commissive
}

# empatheticdialogues: context strings
EMPATHETIC_CONTEXT_MAP: dict[str, list[tuple[str, str, float]]] = {
    'angry':        [('P13', 'attack', 0.75), ('P15', 'tense', 0.70), ('P35', 'slow_escalation', 0.50)],
    'sentimental':  [('P32', 'approach', 0.70), ('P40', 'sincerity', 0.70), ('P12', 'controlled_vulnerability', 0.55)],
    'joyful':       [('P9', 'confidence', 0.70), ('P20', 'genuine_friendliness', 0.70), ('P32', 'approach', 0.55)],
    'terrified':    [('P12', 'vulnerability', 0.80), ('P15', 'tense', 0.80), ('P14', 'containment', 0.60)],
    'disgusted':    [('P18', 'devaluation', 0.65), ('P21', 'hidden_hostility', 0.65), ('P19', 'soft_humiliation', 0.45)],
    'anticipating': [('P9', 'confidence', 0.55), ('P6', 'opening', 0.55)],
    'excited':      [('P9', 'confidence', 0.75), ('P20', 'genuine_friendliness', 0.60), ('P15', 'tense', 0.40)],
    'sad':          [('P12', 'vulnerability', 0.75), ('P33', 'protective_distancing', 0.55), ('P14', 'suppression', 0.50)],
    'afraid':       [('P12', 'vulnerability', 0.80), ('P15', 'tense', 0.75)],
    'surprised':    [('P8', 'doubt', 0.55), ('P7', 'diffuse', 0.45)],
    'faithful':     [('P40', 'sincerity', 0.70), ('P17', 'genuine_respect', 0.60)],
    'grateful':     [('P42', 'admission', 0.65), ('P40', 'sincerity', 0.65)],
    'impressed':    [('P17', 'genuine_respect', 0.65), ('P9', 'confidence', 0.50)],
    'embarrassed':  [('P12', 'vulnerability', 0.70), ('P41', 'emotional_masking', 0.55)],
    'ashamed':      [('P12', 'vulnerability', 0.75), ('P43', 'emotional_denial', 0.55)],
    'devastated':   [('P12', 'vulnerability', 0.85), ('P37', 'hard_rupture', 0.60)],
    'disappointed': [('P47', 'hidden_reproach', 0.70), ('P33', 'protective_distancing', 0.55)],
    'guilty':       [('P42', 'admission', 0.70), ('P47', 'hidden_guilt', 0.65)],
    'jealous':      [('P47', 'hidden_longing', 0.65), ('P43', 'emotional_denial', 0.60)],
    'lonely':       [('P12', 'vulnerability', 0.75), ('P47', 'hidden_plea', 0.65)],
    'nostalgic':    [('P47', 'hidden_longing', 0.75), ('P50', 'return_to_topic', 0.50)],
    'prepared':     [('P9', 'confidence', 0.65), ('P7', 'defined', 0.60)],
    'proud':        [('P9', 'confidence', 0.75), ('P22', 'soft_dominance', 0.55)],
    'trusting':     [('P40', 'sincerity', 0.75), ('P32', 'approach', 0.65)],
    'apprehensive': [('P8', 'doubt', 0.65), ('P12', 'controlled_vulnerability', 0.55)],
    'caring':       [('P16', 'genuine_care', 0.80), ('P32', 'approach', 0.65)],
    'confident':    [('P9', 'confidence', 0.80), ('P7', 'defined', 0.60)],
    'content':      [('P9', 'confidence', 0.60), ('P40', 'sincerity', 0.55)],
    'furious':      [('P13', 'attack', 0.85), ('P15', 'overloaded', 0.80), ('P35', 'sudden_escalation', 0.70)],
    'hopeful':      [('P32', 'cautious_approach', 0.65), ('P8', 'doubt', 0.40)],
    'surprised':    [('P8', 'doubt', 0.55)],
}

# P29/P30/P47 из prosocial
PROSOCIAL_SAFETY_MAP = {
    'threat': [('P29', 'fear_manipulation', 0.75), ('P47', 'hidden_threat', 0.80)],
    'harm':   [('P13', 'attack', 0.75), ('P19', 'humiliation', 0.65)],
}

# ─── Накопитель результатов ───────────────────────────────────────────────────

# {p_id: [{"p_id": ..., "variant": ..., "text": ..., "score": ..., "source": ..., "confidence": ...}]}
_BUCKETS: dict[str, list[dict]] = defaultdict(list)


def _add(p_id: str, variant: str, text: str, score: float,
         source: str, confidence: str = 'rule') -> None:
    text = text.strip()
    if not text or len(text) < 5:
        return
    _BUCKETS[p_id].append({
        'p_id': p_id,
        'variant': variant,
        'text': text,
        'score': round(score, 3),
        'source': source,
        'confidence': confidence,  # label_mapped | rule | combined
    })


def _apply_label_map(
    text: str,
    label_list: list[tuple[str, str, float]],
    source: str,
    confidence: str = 'label_mapped',
) -> None:
    for p_id, variant, score in label_list:
        _add(p_id, variant, text, score, source, confidence)


# ─── Английские keyword-правила ──────────────────────────────────────────────
# Дополняют P-реестр для English-датасетов.
# Язык-независимые: правила P-реестра работают на русских словах,
# здесь добавляем английские эквиваленты.

_EN_RULES: dict[str, list[tuple[str, str, float]]] = {
    # P1 — форма речи
    'P1.question':    [('P1', 'question', 0.85)],
    'P1.directive':   [('P1', 'directive', 0.80)],
    # P4 — ответность
    'P4.avoidance':   [('P4', 'avoidance', 0.70)],
    # P8 — сомнение
    'P8.doubt':       [('P8', 'doubt', 0.65)],
    # P9 — уверенность
    'P9.confidence':  [('P9', 'confidence', 0.70)],
    # P12 — уязвимость
    'P12.vulnerability': [('P12', 'vulnerability', 0.75)],
    # P13 — нападение
    'P13.attack':     [('P13', 'attack', 0.75)],
    'P13.soft_attack': [('P13', 'soft_attack', 0.60)],
    # P21 — скрытая враждебность
    'P21.passive_aggressive': [('P21', 'passive_aggressive', 0.65)],
    # P24 — сарказм
    'P24.sarcasm':    [('P24', 'sarcasm', 0.70)],
    # P32 — сближение
    'P32.approach':   [('P32', 'approach', 0.65)],
    # P33 — дистанцирование
    'P33.distancing': [('P33', 'distancing', 0.65)],
    # P40 — искренность
    'P40.sincerity':  [('P40', 'sincerity', 0.65)],
    # P47 — скрытый подтекст
    'P47.hidden_reproach': [('P47', 'hidden_reproach', 0.70)],
    'P47.hidden_threat':   [('P47', 'hidden_threat', 0.75)],
    'P47.hidden_plea':     [('P47', 'hidden_plea', 0.65)],
}

_EN_KEYWORD_MAP: dict[str, str] = {
    # вопросы → P1.question
    'what ': 'P1.question', 'how ': 'P1.question', 'why ': 'P1.question',
    'when ': 'P1.question', 'where ': 'P1.question', 'who ': 'P1.question',
    'do you': 'P1.question', 'are you': 'P1.question', 'can you': 'P1.question',
    'would you': 'P1.question', 'could you': 'P1.question',
    # директивы → P1.directive
    'please ': 'P1.directive', 'tell me': 'P1.directive', 'show me': 'P1.directive',
    'give me': 'P1.directive', 'stop ': 'P1.directive',
    # сомнение → P8
    "i don't know": 'P8.doubt', "i'm not sure": 'P8.doubt', 'maybe ': 'P8.doubt',
    'perhaps ': 'P8.doubt', 'i guess': 'P8.doubt',
    # уверенность → P9
    "i know": 'P9.confidence', "i'm sure": 'P9.confidence', 'definitely': 'P9.confidence',
    'of course': 'P9.confidence', 'obviously': 'P9.confidence',
    # уязвимость → P12
    'i feel': 'P12.vulnerability', 'i am afraid': 'P12.vulnerability',
    "i'm scared": 'P12.vulnerability', "i'm hurt": 'P12.vulnerability',
    'i miss': 'P12.vulnerability',
    # нападение → P13
    "you're wrong": 'P13.attack', "you don't": 'P13.attack',
    "that's stupid": 'P13.attack', "that's ridiculous": 'P13.attack',
    "you never": 'P13.attack', "you always": 'P13.attack',
    # мягкая атака → P13.soft
    "are you sure": 'P13.soft_attack', "really?": 'P13.soft_attack',
    "seriously?": 'P13.soft_attack',
    # скрытый упрёк → P47
    "fine, whatever": 'P47.hidden_reproach', "if that's what you want": 'P47.hidden_reproach',
    "sure, go ahead": 'P47.hidden_reproach', "okay, fine": 'P47.hidden_reproach',
    # скрытая угроза → P47
    "we'll see": 'P47.hidden_threat', "just wait": 'P47.hidden_threat',
    "you'll regret": 'P47.hidden_threat',
    # скрытая просьба → P47
    "i'll be fine": 'P47.hidden_plea', "don't worry about me": 'P47.hidden_plea',
    "i can handle it": 'P47.hidden_plea',
    # пассивная агрессия → P21
    "whatever": 'P21.passive_aggressive', "if you say so": 'P21.passive_aggressive',
    "as always": 'P21.passive_aggressive', "typical": 'P21.passive_aggressive',
    # сарказм → P24
    "oh great": 'P24.sarcasm', "oh sure": 'P24.sarcasm',
    "how wonderful": 'P24.sarcasm', "right, because": 'P24.sarcasm',
    # сближение → P32
    "i love": 'P32.approach', "i like": 'P32.approach', "thank you": 'P32.approach',
    "i appreciate": 'P32.approach', "tell me more": 'P32.approach',
    # дистанцирование → P33
    "i need space": 'P33.distancing', "leave me alone": 'P33.distancing',
    "not now": 'P33.distancing', "i don't want to": 'P33.distancing',
    # искренность → P40
    "honestly": 'P40.sincerity', "to be honest": 'P40.sincerity',
    "i really": 'P40.sincerity', "the truth is": 'P40.sincerity',
}


def _apply_en_rules(text: str, source: str) -> None:
    """Быстрые английские keyword-правила без P-реестра."""
    low = text.lower()
    seen_keys: set[str] = set()
    for kw, rule_key in _EN_KEYWORD_MAP.items():
        if kw in low and rule_key not in seen_keys:
            seen_keys.add(rule_key)
            for p_id, variant, score in _EN_RULES.get(rule_key, []):
                _add(p_id, variant, text, score, source, 'en_rule')


def _apply_rules(text: str, source: str, registry: PSubsystemRegistry) -> None:
    """Запустить P-реестр (правила) и добавить результаты с высоким score."""
    # для коротких английских текстов — быстрые EN-правила
    _apply_en_rules(text, source)
    try:
        outputs = registry.compute(text, parallel=False)
        for p_id, out in outputs.items():
            for label, score in out.scores.items():
                if label != 'absent' and score >= 0.4:
                    _add(p_id, label, text, score, source, 'rule')
    except Exception:
        pass


# ─── Обработчики датасетов ────────────────────────────────────────────────────

def _parse_numpy_str_array(s: str) -> list[str]:
    """Парсит numpy-стиль строк: ['a' 'b' 'c'] без запятых."""
    s = s.strip()
    # убираем внешние скобки
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    # извлекаем строки в кавычках (одинарных или двойных)
    parts = re.findall(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", s)
    return [a or b for a, b in parts]


def _parse_numpy_int_array(s: str) -> list[int]:
    """Парсит numpy-стиль int: [3 4 2 2] без запятых."""
    return [int(x) for x in re.findall(r'\d+', s)]


def process_archive(registry: PSubsystemRegistry) -> int:
    """archive/train.csv: dialog (numpy str list), act (numpy int list), emotion."""
    path = DATASETS_DIR / 'archive' / 'train.csv'
    if not path.exists():
        return 0
    count = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                utterances = _parse_numpy_str_array(row['dialog'])
                acts = _parse_numpy_int_array(row['act'])
                emotions = _parse_numpy_int_array(row['emotion'])
            except Exception:
                continue

            for i, utt in enumerate(utterances):
                text = str(utt).strip().strip("'\"")
                if not text:
                    continue
                act = acts[i] if i < len(acts) else 0
                emotion = emotions[i] if i < len(emotions) else 0

                _apply_label_map(text, ARCHIVE_ACT_MAP.get(act, []), 'archive')
                _apply_label_map(text, ARCHIVE_EMOTION_MAP.get(emotion, []), 'archive')
                _apply_rules(text, 'archive', registry)
                count += 1
    return count


def process_empatheticdialogues(registry: PSubsystemRegistry) -> int:
    """empatheticdialogues/train.csv: conv_id, utterance_idx, context, utterance."""
    count = 0
    for split in ['train', 'valid']:
        path = DATASETS_DIR / 'empatheticdialogues' / f'{split}.csv'
        if not path.exists():
            continue
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = str(row.get('utterance') or '').strip()
                context = str(row.get('context') or '').lower().strip()
                if not text:
                    continue
                text = text.replace('_comma_', ',').replace('_period_', '.')
                label_list = EMPATHETIC_CONTEXT_MAP.get(context, [])
                _apply_label_map(text, label_list, f'empathetic_{split}')
                _apply_rules(text, f'empathetic_{split}', registry)
                count += 1
    return count


def process_dialogstudio(registry: PSubsystemRegistry) -> int:
    """DialogStudio JSONs: nested dialog format with log turns."""
    count = 0
    ds_dir = DATASETS_DIR / 'DialogStudio'
    for json_path in ds_dir.glob('*.json'):
        source = json_path.stem.replace('_converted_examples', '')
        try:
            data = json.load(open(json_path))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        for dialog_id, dialog in data.items():
            log = dialog.get('log', [])
            orig_info = dialog.get('original dialog info', {})
            context_label = str(orig_info.get('context') or '').lower().strip()

            for turn in log:
                for field in ('user utterance', 'system response'):
                    text = str(turn.get(field) or '').strip()
                    if not text:
                        continue

                    # emotion context если есть
                    label_list = EMPATHETIC_CONTEXT_MAP.get(context_label, [])
                    _apply_label_map(text, label_list, source)

                    # original side information может содержать emotion/safety теги
                    side_key = 'original user side information' if field == 'user utterance' \
                               else 'original system side information'
                    side = turn.get(side_key, {})
                    if isinstance(side, dict):
                        emotion_str = str(side.get('emotion') or '').lower()
                        if emotion_str in EMPATHETIC_CONTEXT_MAP:
                            _apply_label_map(text, EMPATHETIC_CONTEXT_MAP[emotion_str], source, 'label_mapped_side')
                        safety = str(side.get('safety_label') or side.get('safety') or '').lower()
                        for key, plist in PROSOCIAL_SAFETY_MAP.items():
                            if key in safety:
                                _apply_label_map(text, plist, source, 'label_mapped_safety')

                    _apply_rules(text, source, registry)
                    count += 1
    return count


def process_clariq(registry: PSubsystemRegistry) -> int:
    """ClariQ: clarification questions — P1.question, P4.counter_question, P30."""
    count = 0
    for filename in ['train.tsv', 'dev.tsv']:
        path = DATASETS_DIR / 'ClariQ' / filename
        if not path.exists():
            continue
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                question = str(row.get('question') or '').strip()
                answer = str(row.get('answer') or '').strip()
                if question:
                    # clarification question → P1.question, P4.counter_question
                    _add('P1', 'question', question, 0.85, 'clariq_question', 'label_mapped')
                    _add('P4', 'counter_question', question, 0.75, 'clariq_question', 'label_mapped')
                    _apply_rules(question, 'clariq_question', registry)
                    count += 1
                if answer:
                    _add('P4', 'answer', answer, 0.80, 'clariq_answer', 'label_mapped')
                    _apply_rules(answer, 'clariq_answer', registry)
                    count += 1
    return count


def process_multi_turn_clariq(registry: PSubsystemRegistry) -> int:
    """ClariQ multi-turn: question chains — давление через повторные уточнения."""
    path = DATASETS_DIR / 'ClariQ' / 'multi_turn_human_generated_data.tsv'
    if not path.exists():
        return 0
    count = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            text = str(row.get('question') or row.get('utterance') or '').strip()
            if not text:
                continue
            _add('P1', 'question', text, 0.85, 'clariq_multi', 'label_mapped')
            _add('P30', 'repetition_pressure', text, 0.55, 'clariq_multi', 'label_mapped')
            _apply_rules(text, 'clariq_multi', registry)
            count += 1
    return count


# ─── Everyday Conversations (English, multi-turn) ────────────────────────────

# Маппинг тем → P-подсказки для assistant-реплик
_TOPIC_P_MAP: dict[str, list[tuple[str, str, float]]] = {
    'Family':      [('P16', 'genuine_care', 0.60), ('P32', 'approach', 0.55)],
    'Health':      [('P16', 'genuine_care', 0.65), ('P32', 'problem_solving', 0.55)],
    'Work':        [('P9',  'confidence', 0.55),   ('P22', 'soft_dominance', 0.50)],
    'Shopping':    [('P4',  'answer', 0.60)],
    'Food':        [('P4',  'answer', 0.60)],
    'Cooking':     [('P4',  'answer', 0.60)],
    'Travel':      [('P4',  'answer', 0.60),        ('P32', 'approach', 0.50)],
    'Sports':      [('P4',  'answer', 0.60)],
    'Music':       [('P4',  'answer', 0.60)],
    'Technology':  [('P4',  'answer', 0.60),        ('P9', 'confidence', 0.50)],
    'Weather':     [('P4',  'answer', 0.60)],
}


def process_everyday_conversations(registry: PSubsystemRegistry) -> int:
    """
    DataSets/everyday_conversations.jsonl — нейтральные английские диалоги.

    Стратегия:
    - user turns  → P1 (question/statement/greeting)
    - assistant turns → P4.answer + topic trait + keyword rules
    - Все assistant turns → явные нейтральные негативы P47 (score=0.0)
      это главный источник чистых «absent» примеров для английского
    """
    path = DATASETS_DIR / 'everyday_conversations.jsonl'
    if not path.exists():
        return 0

    # P47 variants для нейтральных негативов
    _P47_VARIANTS = [
        'hidden_reproach', 'hidden_threat', 'hidden_contempt',
        'hidden_plea', 'hidden_affection',
    ]

    count = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            messages = row.get('messages') or []
            topic = str(row.get('topic') or '')
            topic_labels = _TOPIC_P_MAP.get(topic, [('P4', 'answer', 0.60)])

            for i, msg in enumerate(messages):
                role = str(msg.get('role') or '')
                text = str(msg.get('content') or '').strip()
                if not text or len(text) < 4:
                    continue

                if role == 'user':
                    # Первый ход — приветствие
                    if i == 0 and text.lower() in ('hi', 'hey', 'hello', 'hi!', 'hey!'):
                        _add('P32', 'approach', text, 0.80, 'everyday_conv', 'label_mapped')
                    elif '?' in text:
                        _add('P1', 'question', text, 0.85, 'everyday_conv', 'label_mapped')
                    else:
                        _add('P1', 'statement', text, 0.70, 'everyday_conv', 'label_mapped')
                    _apply_en_rules(text, 'everyday_conv')

                elif role == 'assistant':
                    # Topic-based labels
                    for p_id, variant, score in topic_labels:
                        _add(p_id, variant, text, score, 'everyday_conv', 'label_mapped')

                    # Keyword rules (EN)
                    _apply_en_rules(text, 'everyday_conv')

                    # Нейтральные негативы для P47 — ключевая ценность датасета
                    for variant in _P47_VARIANTS:
                        _add('P47', variant, text, 0.0, 'everyday_conv', 'neutral_negative')

            count += 1

    return count


# ─── Armenian Intonation Dataset ─────────────────────────────────────────────

def process_armenian_intonation(registry: PSubsystemRegistry) -> int:
    """
    DataSets/armenian_sentences.jsonl — армянские предложения.

    Содержит: короткие реальные армянские предложения (утверждения/вопросы).
    Используем как:
    - statements → P1.statement (армянский)
    - questions  → P1.question  (армянский)
    - Все        → нейтральные негативы P47 (score=0.0)
    """
    path = DATASETS_DIR / 'armenian_sentences.jsonl'
    if not path.exists():
        return 0

    _P47_VARIANTS = [
        'hidden_reproach', 'hidden_threat', 'hidden_contempt',
        'hidden_plea', 'hidden_affection',
    ]

    count = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            text = str(row.get('text') or '').strip()
            if not text or len(text) < 3:
                continue

            stype = str(row.get('type') or 'statement')
            score = 0.85 if stype == 'question' else 0.75
            variant = 'question' if stype == 'question' else 'statement'

            _add('P1', variant, text, score, 'armenian_intonation', 'label_mapped')

            # Нейтральные негативы для P47 (армянские предложения — без подтекста)
            for v in _P47_VARIANTS:
                _add('P47', v, text, 0.0, 'armenian_intonation', 'neutral_negative')

            count += 1
    return count


# ─── SiberianPersonaChat ─────────────────────────────────────────────────────

# Маппинг черт персоны → P-метки
_PERSONA_TRAIT_MAP: list[tuple[list[str], list[tuple[str, str, float]]]] = [
    (['умный', 'интеллект', 'эрудит', 'образован', 'учёный', 'профессор'],
     [('P22', 'intellectual_dominance', 0.60), ('P9', 'confidence', 0.65)]),
    (['помогать', 'помощь', 'забот', 'поддерж', 'добрый', 'добр'],
     [('P16', 'genuine_care', 0.65), ('P32', 'approach', 0.60)]),
    (['лидер', 'руковод', 'директор', 'начальник', 'менеджер'],
     [('P22', 'dominance', 0.65), ('P9', 'confidence', 0.70)]),
    (['уверен', 'решит', 'смел', 'напористый', 'дерзк'],
     [('P9', 'confidence', 0.70)]),
    (['застенч', 'скромн', 'тих', 'неуверен'],
     [('P12', 'controlled_vulnerability', 0.60), ('P8', 'doubt', 0.55)]),
    (['честный', 'открыт', 'прямолинейн', 'искренн'],
     [('P40', 'sincerity', 0.65), ('P9', 'quiet_certainty', 0.55)]),
    (['ироничн', 'саркасти', 'остроумн'],
     [('P24', 'sarcasm', 0.60), ('P25', 'playful_irony', 0.55)]),
    (['эмоцион', 'страстн', 'экспрессивн'],
     [('P15', 'tense', 0.55)]),
    (['аналитич', 'логичн', 'рациональн', 'системат'],
     [('P7', 'defined', 0.65), ('P9', 'overconfidence', 0.50)]),
    (['заботлив', 'нежн', 'ласков', 'тёплый', 'тёпл'],
     [('P16', 'genuine_care', 0.65), ('P47', 'hidden_affection', 0.50)]),
]


def _persona_trait_labels(persona_desc: str) -> list[tuple[str, str, float]]:
    """Извлекает P-метки из описания персоны."""
    low = persona_desc.lower()
    labels: list[tuple[str, str, float]] = []
    for keywords, p_labels in _PERSONA_TRAIT_MAP:
        if any(kw in low for kw in keywords):
            labels.extend(p_labels)
    return labels


def _parse_persona_dialog(input_text: str) -> tuple[str, list[tuple[str, str]], str]:
    """
    Возвращает (persona_desc, turns, last_user_msg).
    turns: [('user'|'persona', text), ...]
    """
    if 'Продолжи диалог:' in input_text:
        persona_part, dialog_part = input_text.split('Продолжи диалог:', 1)
        persona_desc = persona_part.strip()
    else:
        persona_desc = ''
        dialog_part = input_text

    turns: list[tuple[str, str]] = []
    for line in dialog_part.strip().split('\n'):
        line = line.strip()
        if line.startswith('Собеседник:'):
            turns.append(('user', line[len('Собеседник:'):].strip()))
        elif line.startswith('Ты:'):
            turns.append(('persona', line[len('Ты:'):].strip()))

    last_user = next((t[1] for t in reversed(turns) if t[0] == 'user'), '')
    return persona_desc, turns, last_user


def process_siberian_persona_chat(registry: PSubsystemRegistry,
                                  max_rows: int = 100_000) -> int:
    """
    DataSets/dataset.json — SiberianPersonaChat (русский).

    Стратегия:
    - dialog_personal_context: парсим персону + диалог
        - output → keyword-правила → P47/P41/P33/P13/...
        - persona_desc → trait маппинг → P9/P22/P16/...
        - user_msg → P1-варианты (вопрос/утверждение)
    - chitchat: output → P4/P1
    - reaction: output → P32/P16
    """
    path = DATASETS_DIR / 'dataset.json'
    if not path.exists():
        return 0

    import random
    rng = random.Random(42)

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    # Разбиваем по типам
    persona_rows = [r for r in data if r.get('name') == 'dialog_personal_context']
    chitchat_rows = [r for r in data if r.get('name') == 'chitchat']
    reaction_rows = [r for r in data if r.get('name') == 'reaction']

    # Сэмплируем если слишком много
    if len(persona_rows) > max_rows:
        persona_rows = rng.sample(persona_rows, max_rows)

    count = 0

    # ── dialog_personal_context ───────────────────────────────────────────────
    for row in persona_rows:
        output = str(row.get('output') or '').strip()
        if len(output) < 15:
            continue

        input_text = str(row.get('input') or '')
        persona_desc, turns, last_user = _parse_persona_dialog(input_text)

        # 1. Output → keyword rules (главный источник P-меток)
        _apply_rules(output, 'siberian_persona', registry)

        # 2. Черты персоны → метки для самого output
        for p_id, variant, score in _persona_trait_labels(persona_desc):
            _add(p_id, variant, output, score, 'siberian_persona', 'persona_trait')

        # 3. Последнее сообщение пользователя → P1
        if last_user and len(last_user) > 5:
            if '?' in last_user:
                _add('P1', 'question', last_user, 0.85, 'siberian_persona', 'label_mapped')
            else:
                _add('P1', 'statement', last_user, 0.70, 'siberian_persona', 'label_mapped')
            _apply_rules(last_user, 'siberian_persona', registry)

        count += 1

    # ── chitchat ──────────────────────────────────────────────────────────────
    for row in chitchat_rows:
        output = str(row.get('output') or '').strip()
        if len(output) < 10:
            continue
        # chitchat — нейтральный ответ
        _add('P4', 'answer', output, 0.65, 'siberian_chitchat', 'label_mapped')
        _apply_rules(output, 'siberian_chitchat', registry)

        user_msg = str(row.get('input') or '').replace('Собеседник:', '').replace('Ты:', '').strip()
        if user_msg and len(user_msg) > 5:
            label = 'question' if '?' in user_msg else 'statement'
            _add('P1', label, user_msg, 0.80, 'siberian_chitchat', 'label_mapped')
        count += 1

    # ── reaction ─────────────────────────────────────────────────────────────
    for row in reaction_rows:
        output = str(row.get('output') or '').strip()
        if len(output) < 10:
            continue
        _add('P32', 'problem_solving', output, 0.65, 'siberian_reaction', 'label_mapped')
        _apply_rules(output, 'siberian_reaction', registry)
        count += 1

    return count


# ─── Запись результатов ───────────────────────────────────────────────────────

def write_output(min_examples: int = 3) -> dict[str, int]:
    """
    Мержит новые примеры из _BUCKETS с существующими файлами.
    Приоритет: expert > label_mapped > rule.
    Дубли определяются по (text[:120], variant).
    """
    _PRIORITY = {'expert': 3, 'label_mapped': 2, 'persona_trait': 1, 'rule': 0, 'en_rule': 0}
    counts: dict[str, int] = {}

    all_p_ids = set(_BUCKETS.keys())
    # Также учитываем файлы, которые уже есть на диске (экспертные аннотации)
    for f in OUT_DIR.glob('P*.jsonl'):
        all_p_ids.add(f.stem)

    for p_id in sorted(all_p_ids):
        out_path = OUT_DIR / f'{p_id}.jsonl'

        # Загружаем существующие (экспертные, предыдущие)
        existing: dict[tuple[str, str], dict] = {}
        if out_path.exists():
            for line in out_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    key = (ex['text'][:120], ex['variant'])
                    prev = existing.get(key)
                    if prev is None or _PRIORITY.get(ex.get('confidence',''), 0) >= _PRIORITY.get(prev.get('confidence',''), 0):
                        existing[key] = ex
                except Exception:
                    pass

        # Мержим новые из _BUCKETS
        for ex in _BUCKETS.get(p_id, []):
            key = (ex['text'][:120], ex['variant'])
            prev = existing.get(key)
            if prev is None or _PRIORITY.get(ex.get('confidence',''), 0) > _PRIORITY.get(prev.get('confidence',''), 0):
                existing[key] = ex

        merged = list(existing.values())
        if len(merged) < min_examples:
            continue

        with open(out_path, 'w', encoding='utf-8') as f:
            for ex in merged:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        counts[p_id] = len(merged)

    return counts


# ─── Сводная статистика по вариантам ─────────────────────────────────────────

def variant_summary() -> dict[str, dict[str, int]]:
    """Сколько примеров накоплено по каждому P-варианту."""
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p_id, examples in _BUCKETS.items():
        for ex in examples:
            summary[p_id][ex['variant']] += 1
    return {k: dict(v) for k, v in summary.items()}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print('Строю P-реестр...')
    registry = PSubsystemRegistry.build()
    print(f'Реестр: {registry.status()["families"]} семей, {registry.status()["total_variants"]} вариантов\n')

    total = 0
    print('Обрабатываю archive/train.csv...')
    n = process_archive(registry)
    print(f'  {n} реплик')
    total += n

    print('Обрабатываю empatheticdialogues...')
    n = process_empatheticdialogues(registry)
    print(f'  {n} реплик')
    total += n

    print('Обрабатываю DialogStudio...')
    n = process_dialogstudio(registry)
    print(f'  {n} реплик')
    total += n

    print('Обрабатываю ClariQ...')
    n = process_clariq(registry)
    print(f'  {n} реплик')
    total += n

    print('Обрабатываю ClariQ multi-turn...')
    n = process_multi_turn_clariq(registry)
    print(f'  {n} реплик')
    total += n

    print('Обрабатываю Armenian Intonation Dataset...')
    n = process_armenian_intonation(registry)
    print(f'  {n} предложений')
    total += n

    print('Обрабатываю Everyday Conversations (английский)...')
    n = process_everyday_conversations(registry)
    print(f'  {n} диалогов')
    total += n

    print('Обрабатываю SiberianPersonaChat (dataset.json)...')
    n = process_siberian_persona_chat(registry)
    print(f'  {n} реплик')
    total += n

    print(f'\nВсего обработано: {total} реплик')
    print('Записываю training/p_examples/...')
    counts = write_output(min_examples=3)

    print(f'\nВыходные файлы: {len(counts)} P-семей')
    total_examples = sum(counts.values())
    print(f'Итого примеров: {total_examples}')
    print()

    # топ-10 P-семей по количеству примеров
    print('Топ P-семей по накопленным примерам:')
    for p_id, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f'  {p_id}: {cnt}')

    print()
    # детализация по вариантам для топ-5
    print('Детализация по вариантам (топ-5 семей):')
    vsummary = variant_summary()
    for p_id, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f'  {p_id}:')
        for variant, vcnt in sorted(vsummary.get(p_id, {}).items(), key=lambda x: x[1], reverse=True):
            print(f'    {variant}: {vcnt}')


if __name__ == '__main__':
    main()
