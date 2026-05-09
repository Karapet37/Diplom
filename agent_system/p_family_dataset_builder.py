"""
p_family_dataset_builder.py — автоматическое извлечение обучающих данных
для P-семей из dataset.json и everyday_conversations.jsonl.

Запуск:
    python -m agent_system.p_family_dataset_builder

Выходные файлы: DataSets/p_annotations/P{N}_{variant}.jsonl
    {"text": "...", "p_id": "F1", "variant": "question", "score": 1.0}
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

_ROOT = Path(__file__).parent.parent
_DS_MAIN = _ROOT / 'DataSets' / 'dataset.json'
_DS_CONV = _ROOT / 'DataSets' / 'everyday_conversations.jsonl'
_DS_ARM  = _ROOT / 'DataSets' / 'armenian_sentences.jsonl'
_OUT_DIR = _ROOT / 'DataSets' / 'p_annotations'

# ── Детекторы для авто-разметки ────────────────────────────────────────────────

_RU_QUESTION_WORDS = re.compile(
    r'\b(что|кто|где|когда|зачем|почему|как|какой|какая|какие|чем|кого|кому|ли)\b',
    re.I
)
_RU_IMPERATIVE = re.compile(
    r'\b(сделай|иди|скажи|возьми|дай|подожди|посмотри|помоги|остановись|прекрати'
    r'|убери|принеси|позвони|напиши|забудь|расскажи|покажи|слушай|молчи|отвечай)\b',
    re.I
)
_EN_QUESTION_WORDS = re.compile(
    r'\b(what|who|where|when|why|how|which|whose|whom)\b', re.I
)
_ATTACK_RU = re.compile(
    r'\b(дурак|идиот|тупой|ничтожество|мразь|жалкий|идиотка|кретин|придурок'
    r'|заткнись|убирайся|пошёл|отвали|ненавижу)\b', re.I
)
_ATTACK_EN = re.compile(
    r"\b(idiot|stupid|moron|pathetic|worthless|shut up|get out|i hate you"
    r"|you're wrong|you're stupid|loser|you fool)\b", re.I
)
_DOUBT_RU = re.compile(
    r'\b(может|наверное|кажется|не знаю|сомневаюсь|возможно|вроде|как-то)\b', re.I
)
_CARE_RU = re.compile(
    r'\b(береги|не простудись|позвони|тепло|покушай|осторожнее|как ты'
    r'|всё нормально|как дела|не волнуйся)\b', re.I
)

# Маркеры скрытого подтекста (из _KEYWORD_RULES)
_SUBTEXT_RU = {
    'hidden_reproach': ['ну ладно', 'делай как хочешь', 'как знаешь', 'мне без разницы', 'как скажешь'],
    'hidden_plea': ['справлюсь как-нибудь', 'не беспокойся', 'сам справлюсь', 'не волнуйся обо мне'],
    'hidden_threat': ['посмотрим', 'ты только попробуй', 'я запомню', 'будем посмотреть'],
    'hidden_affection': ['осторожнее там', 'тепло оденься', 'позвони когда', 'береги себя'],
    'hidden_contempt': ['интересный подход', 'смелое решение', 'тебе виднее', 'рад за тебя'],
    'hidden_fear': ['всё под контролем', 'мне не нужна помощь', 'справлюсь', 'я держу'],
}


def _label_p1(text: str) -> str | None:
    """P1: форма высказывания."""
    t = text.strip()
    if '?' in t:
        if _RU_QUESTION_WORDS.search(t) or _EN_QUESTION_WORDS.search(t):
            return 'question'
        return 'question'
    if _RU_IMPERATIVE.search(t):
        return 'directive'
    if _DOUBT_RU.search(t) and len(t) < 60:
        return 'thought'
    if t.endswith('.') or t.endswith('!'):
        return 'statement'
    return None


def _label_p8(text: str) -> str | None:
    """P8: сомнение."""
    if _DOUBT_RU.search(text):
        return 'doubt'
    return None


def _label_p13(text: str) -> str | None:
    """P13: нападение."""
    if _ATTACK_RU.search(text) or _ATTACK_EN.search(text):
        return 'attack'
    return None


def _label_p16(text: str) -> str | None:
    """P16: забота."""
    if _CARE_RU.search(text):
        return 'genuine_care'
    return None


def _label_p47(text: str) -> str | None:
    """P47: скрытый подтекст."""
    low = text.lower()
    for variant, markers in _SUBTEXT_RU.items():
        if any(m in low for m in markers):
            return variant
    return None


_LABELERS = {
    'F1':  _label_p1,
    'F8':  _label_p8,
    'F13': _label_p13,
    'F16': _label_p16,
    'F47': _label_p47,
}


# ── Итераторы по датасетам ─────────────────────────────────────────────────────

def _iter_dialog_dataset(path: Path) -> Iterator[str]:
    """Извлекает текстовые реплики из dataset.json."""
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding='utf-8'))
    for item in data:
        output = str(item.get('output') or '').strip()
        if output and len(output) > 5:
            yield output
        # Also yield Собеседник lines from input
        raw_input = str(item.get('input') or '')
        for line in raw_input.splitlines():
            if line.startswith('Собеседник:'):
                text = line[len('Собеседник:'):].strip()
                if text and len(text) > 3:
                    yield text


def _iter_conversation_dataset(path: Path) -> Iterator[str]:
    """Извлекает реплики из everyday_conversations.jsonl."""
    if not path.exists():
        return
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            for msg in (ex.get('messages') or []):
                content = str(msg.get('content') or '').strip()
                if content and len(content) > 5:
                    yield content


def _iter_armenian_dataset(path: Path) -> Iterator[tuple[str, str]]:
    """Из armenian_sentences.jsonl — уже есть type: question/statement."""
    if not path.exists():
        return
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                text = str(ex.get('text') or '').strip()
                typ = str(ex.get('type') or '').strip()
                if text and typ:
                    yield text, typ
            except json.JSONDecodeError:
                continue


# ── Основная функция ───────────────────────────────────────────────────────────

def build_p_annotation_files(
    max_per_variant: int = 2000,
    min_per_variant: int = 30,
) -> dict[str, int]:
    """
    Собирает аннотации из всех датасетов и сохраняет в DataSets/p_annotations/.
    Возвращает словарь {p_id.variant: count}.
    """
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # accumulate: {pid: {variant: [text, ...]}}
    collected: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    # 1. Armenian dataset → P1 labels (already ground-truth)
    print('  Loading armenian_sentences.jsonl...')
    for text, typ in _iter_armenian_dataset(_DS_ARM):
        if typ == 'question':
            collected['F1']['question'].append(text)
        elif typ == 'statement':
            collected['F1']['statement'].append(text)

    # 2. Main dialog dataset
    print('  Loading dataset.json (this takes a moment)...')
    for text in _iter_dialog_dataset(_DS_MAIN):
        if len(collected['F1']['question']) + len(collected['F1']['statement']) < max_per_variant * 3:
            lbl = _label_p1(text)
            if lbl:
                if len(collected['F1'][lbl]) < max_per_variant:
                    collected['F1'][lbl].append(text)

        for pid, labeler in _LABELERS.items():
            if pid == 'F1':
                continue
            bucket = collected[pid]
            lbl = labeler(text)
            if lbl and len(bucket[lbl]) < max_per_variant:
                bucket[lbl].append(text)

    # 3. Everyday conversations
    print('  Loading everyday_conversations.jsonl...')
    for text in _iter_conversation_dataset(_DS_CONV):
        lbl_p1 = _label_p1(text)
        if lbl_p1 and len(collected['F1'][lbl_p1]) < max_per_variant:
            collected['F1'][lbl_p1].append(text)
        for pid, labeler in _LABELERS.items():
            if pid == 'F1':
                continue
            lbl = labeler(text)
            if lbl and len(collected[pid][lbl]) < max_per_variant:
                collected[pid][lbl].append(text)

    # Write files
    counts: dict[str, int] = {}
    for pid, variants in collected.items():
        for variant, texts in variants.items():
            if len(texts) < min_per_variant:
                continue
            out_path = _OUT_DIR / f'{pid}_{variant}.jsonl'
            with open(out_path, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(json.dumps({'text': text, 'p_id': pid, 'variant': variant, 'score': 1.0}, ensure_ascii=False) + '\n')
            key = f'{pid}.{variant}'
            counts[key] = len(texts)
            print(f'    {key}: {len(texts)} examples → {out_path.name}')

    return counts


# ── Дообучение P-семей из собранных аннотаций ─────────────────────────────────

def retrain_p_families_from_annotations(annotations_dir: Path | None = None) -> dict[str, bool]:
    """
    Читает DataSets/p_annotations/*.jsonl и переобучает SklearnVariant-модели.
    """
    from .p_subsystem_registry import PSubsystemRegistry, train_from_file
    from pathlib import Path as _P

    ann_dir = annotations_dir or _OUT_DIR
    if not ann_dir.exists():
        print('No annotations dir found. Run build_p_annotation_files() first.')
        return {}

    # Merge all annotation files into a single merged.jsonl per p_id
    by_pid: dict[str, list[Path]] = defaultdict(list)
    for f in ann_dir.glob('*.jsonl'):
        pid = f.stem.split('_')[0]
        by_pid[pid].append(f)

    models_dir = _P(__file__).parent.parent / 'models' / 'p_subsystems'
    reg = PSubsystemRegistry.build(models_dir=models_dir)

    results = {}
    for pid, files in by_pid.items():
        # Merge into temp file
        merged = ann_dir / f'_merged_{pid}.jsonl'
        with open(merged, 'w', encoding='utf-8') as out:
            for f in files:
                out.write(f.read_text(encoding='utf-8'))

        family = reg.family(pid)
        if family is None:
            continue

        # Load examples
        from collections import defaultdict as _dd
        positives: dict[str, list[str]] = _dd(list)
        with open(merged, encoding='utf-8') as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    if float(ex.get('score', 1.0)) >= 0.5:
                        positives[ex['variant']].append(ex['text'])
                except Exception:
                    pass

        for variant, pos in positives.items():
            if len(pos) < 10:
                continue
            # Use texts of other variants as negatives
            neg = []
            for other_v, other_texts in positives.items():
                if other_v != variant:
                    neg.extend(other_texts[:len(pos) * 2])
            if len(neg) < 5:
                continue
            ok = family.train_variant(variant, pos, neg)
            results[f'{pid}.{variant}'] = ok
            print(f'  {"✓" if ok else "✗"} {pid}.{variant}: {len(pos)} pos, {len(neg)} neg')

        merged.unlink(missing_ok=True)

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'build'

    if cmd == 'build':
        print('Building P-family annotation files...')
        counts = build_p_annotation_files()
        print(f'\nDone. {len(counts)} variant files written to {_OUT_DIR}')

    elif cmd == 'train':
        print('Retraining P-families from annotations...')
        results = retrain_p_families_from_annotations()
        trained = sum(1 for v in results.values() if v)
        print(f'\nDone. {trained}/{len(results)} variants retrained.')

    elif cmd == 'all':
        print('=== Step 1: Build annotations ===')
        build_p_annotation_files()
        print('\n=== Step 2: Retrain P-families ===')
        retrain_p_families_from_annotations()
