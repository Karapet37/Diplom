"""
genome_validator.py — геном → стиль → валидация ответа.

Пайплайн:
  1. load_or_init_persona_genome(name) → PersonalityGenome
  2. genome_to_style_description(genome, language) → str  [идёт в промпт]
  3. genome_expected_p_activations(genome) → dict[str, float]  [ожидаемые P-активации]
  4. check_genome_fit(response_text, genome, language) → GenomeFitResult
  5. build_genome_repair_block(result, language) → str  [идёт в repair-промпт]
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .genome import PersonalityGenome

# ── Путь к геномам ────────────────────────────────────────────────────────────

_MEMORY_ROOT = Path(os.environ.get('COGNITIVE_MEMORY_ROOT', 'memory'))


def _genome_path(persona_name: str) -> Path:
    from .duplicate_resolver import normalize_name
    return _MEMORY_ROOT / 'heads' / normalize_name(persona_name) / 'genome.json'


def load_or_init_persona_genome(persona_name: str) -> PersonalityGenome:
    """Загружает геном из файла или создаёт дефолтный из трейтов персоны."""
    path = _genome_path(persona_name)
    if path.exists():
        try:
            return PersonalityGenome.load(path)
        except Exception:
            pass
    genome = PersonalityGenome.default_for(persona_name)
    _apply_traits_to_genome(genome, persona_name)
    return genome


def _apply_traits_to_genome(genome: PersonalityGenome, persona_name: str) -> None:
    """Устанавливает начальные значения генома из трейтов персоны."""
    try:
        from .persona_engine import load_active_persona
        bundle = load_active_persona(persona_name)
        if bundle is None:
            return
        traits = {t.strip().lower() for t in bundle.traits}
    except Exception:
        return

    # Trait → genome parameter adjustments
    _TRAIT_GENOME: dict[str, list[tuple[str, float]]] = {
        'predatory':    [('drive_control', 0.8), ('suspicion_bias', 0.7), ('trust_baseline', 0.2), ('dominance_tendency', 0.8)],
        'ruthless':     [('drive_control', 0.8), ('defense_aggression', 0.7), ('blame_self_vs_other', 0.8)],
        'contemptuous': [('drive_superiority', 0.85), ('approval_seeking', 0.05), ('suspicion_bias', 0.6)],
        'aggressive':   [('defense_aggression', 0.7), ('dominance_tendency', 0.7), ('impulsivity', 0.6)],
        'imposing':     [('dominance_tendency', 0.8), ('drive_superiority', 0.7), ('social_distance_default', 0.7)],
        'arrogant':     [('drive_superiority', 0.9), ('approval_seeking', 0.05), ('hierarchy_sensitivity', 0.7)],
        'cold':         [('trust_baseline', 0.2), ('social_distance_default', 0.8), ('feel_first', 0.1)],
        'sarcastic':    [('defense_humor', 0.7), ('suspicion_bias', 0.6), ('drive_superiority', 0.6)],
        'aristocratic': [('drive_superiority', 0.75), ('social_distance_default', 0.7), ('approval_seeking', 0.1)],
        'dignified':    [('social_distance_default', 0.65), ('vulnerability_concealment', 0.7)],
        'philosophical':  [('drive_meaning', 0.85), ('analysis_bias', 0.7), ('planning_depth', 0.7)],
        'determined':   [('baseline_drive', 0.8), ('drive_autonomy', 0.7), ('category_rigidity', 0.6)],
        'principled':   [('category_rigidity', 0.7), ('blame_self_vs_other', 0.2), ('drive_autonomy', 0.8)],
        'idealist':     [('drive_meaning', 0.8), ('category_rigidity', 0.65), ('fear_chaos', 0.6)],
        'brave':        [('defense_avoidance', 0.1), ('baseline_anxiety', 0.2), ('drive_autonomy', 0.7)],
        'honest':       [('vulnerability_concealment', 0.1), ('defense_rationalization', 0.15), ('blame_self_vs_other', 0.3)],
        'direct':       [('impulsivity', 0.5), ('social_distance_default', 0.3), ('defense_rationalization', 0.15)],
        'rigid':        [('category_rigidity', 0.85), ('ambiguity_tolerance', 0.1), ('hypothesis_switch_speed', 0.1)],
        'protective':   [('drive_security', 0.8), ('fear_abandonment', 0.5), ('dominance_tendency', 0.5)],
        'loyal':        [('drive_closeness', 0.8), ('fear_abandonment', 0.6), ('trust_baseline', 0.7)],
        'confident':    [('approval_seeking', 0.1), ('baseline_anxiety', 0.2), ('drive_superiority', 0.55)],
        'empathetic':   [('drive_closeness', 0.7), ('feel_first', 0.7), ('mirror_tendency', 0.7)],
        'warm':         [('drive_closeness', 0.8), ('trust_baseline', 0.7), ('feel_first', 0.6)],
        'insecure':     [('fear_rejection', 0.75), ('approval_seeking', 0.75), ('baseline_anxiety', 0.65)],
        'self-doubting':[('fear_failure', 0.7), ('blame_self_vs_other', 0.2), ('baseline_anxiety', 0.6)],
        'humorous':     [('defense_humor', 0.8), ('social_distance_default', 0.3), ('drive_closeness', 0.5)],
        'defensive':    [('vulnerability_concealment', 0.75), ('defense_avoidance', 0.6), ('fear_judgment', 0.65)],
        'cautious':     [('threat_first', 0.7), ('planning_depth', 0.7), ('ambiguity_tolerance', 0.3)],
        'curious':      [('novelty_reward', 0.8), ('analysis_bias', 0.65), ('drive_meaning', 0.6)],
        'analytical':   [('analysis_bias', 0.85), ('planning_depth', 0.75), ('feel_first', 0.1)],
        'intelligent':  [('analysis_bias', 0.75), ('hypothesis_switch_speed', 0.7), ('drive_superiority', 0.5)],
        'brilliant':    [('analysis_bias', 0.9), ('drive_superiority', 0.7), ('approval_seeking', 0.1)],
        'hyper-logical':[('analysis_bias', 0.95), ('feel_first', 0.0), ('ambiguity_tolerance', 0.05)],
        'socially-oblivious': [('social_distance_default', 0.8), ('mirror_tendency', 0.05), ('hierarchy_sensitivity', 0.1)],
        'pedantic':     [('category_rigidity', 0.8), ('ambiguity_tolerance', 0.05), ('analysis_bias', 0.8)],
        'precise':      [('category_rigidity', 0.7), ('planning_depth', 0.7), ('ambiguity_tolerance', 0.1)],
        'logical':      [('analysis_bias', 0.7), ('feel_first', 0.15), ('planning_depth', 0.65)],
    }

    for trait in traits:
        for param_name, value in _TRAIT_GENOME.get(trait, []):
            param = getattr(genome, param_name, None)
            if param is not None:
                param.value = value
                param.prior = value


# ── Геном → описание стиля (для промпта) ─────────────────────────────────────

def genome_to_style_description(genome: PersonalityGenome, language: str = 'ru') -> str:
    """
    Преобразует значения 53 генов в короткое описание стиля персонажа.
    Используется в промпте: 'Веди себя как {description}.'
    """
    g = genome
    parts_ru: list[str] = []
    parts_en: list[str] = []

    def v(param_name: str) -> float:
        p = getattr(g, param_name, None)
        return float(p.value) if p is not None else 0.5

    # Подозрительность / доверие
    if v('suspicion_bias') > 0.65 or v('trust_baseline') < 0.3:
        parts_ru.append('подозрительный — ищет скрытый мотив за каждым словом')
        parts_en.append('suspicious — looking for the hidden motive behind every word')
    elif v('trust_baseline') > 0.72:
        parts_ru.append('доверчивый — принимает слова за чистую монету')
        parts_en.append('trusting — takes words at face value')

    # Доминирование / покорность
    if v('dominance_tendency') > 0.65 or v('drive_superiority') > 0.68:
        parts_ru.append('доминирующий — управляет разговором, не уступает')
        parts_en.append('dominant — controls the conversation, doesn\'t yield')
    elif v('approval_seeking') > 0.7:
        parts_ru.append('ищет одобрения — легко соглашается, боится отказа')
        parts_en.append('approval-seeking — agrees easily, afraid of rejection')

    # Аналитика / импульсивность
    if v('analysis_bias') > 0.80:
        parts_ru.append('холодно-аналитический — обрабатывает факты, не эмоции')
        parts_en.append('coldly analytical — processes facts, not emotions')
    elif v('impulsivity') > 0.62:
        parts_ru.append('импульсивный — реагирует быстро, говорит прежде чем думает')
        parts_en.append('impulsive — reacts fast, speaks before thinking')
    elif v('feel_first') > 0.68:
        parts_ru.append('сначала чувствует — эмоция идёт впереди логики')
        parts_en.append('feels first — emotion leads, logic follows')

    # Тревожность / защита
    if v('baseline_anxiety') > 0.6:
        parts_ru.append('напряжённый — видит угрозу там, где её нет')
        parts_en.append('tense — sees threat where there is none')
    if v('defense_aggression') > 0.62:
        parts_ru.append('защищается агрессией — атака как первая реакция')
        parts_en.append('defends with aggression — attack as first response')
    elif v('defense_humor') >= 0.65:
        parts_ru.append('прячется за юмором — шутка вместо честного ответа')
        parts_en.append('hides behind humour — jokes instead of honest answers')
    elif v('defense_avoidance') < 0.15:
        parts_ru.append('не избегает конфликта — говорит прямо, даже если неудобно')
        parts_en.append('doesn\'t avoid conflict — speaks plainly even when uncomfortable')

    # Принципиальность / автономия
    if v('drive_autonomy') > 0.75 and v('category_rigidity') > 0.65:
        parts_ru.append('принципиальный — не отступает от своих ценностей под давлением')
        parts_en.append('principled — doesn\'t abandon his values under pressure')

    # Жёсткость позиции
    if v('category_rigidity') > 0.80:
        parts_ru.append('негибкий — свою позицию не меняет')
        parts_en.append('rigid — won\'t change his position')
    elif v('ambiguity_tolerance') > 0.75:
        parts_ru.append('терпит неопределённость — не торопится с выводами')
        parts_en.append('tolerates ambiguity — takes time before concluding')

    # Превосходство / самооценка
    if v('drive_superiority') >= 0.75 and v('approval_seeking') < 0.2:
        parts_ru.append('считает себя выше других — не нуждается в чьём-то мнении')
        parts_en.append('considers himself above others — doesn\'t need anyone\'s opinion')

    # Скрытность / маска
    if v('vulnerability_concealment') >= 0.88:
        parts_ru.append('никогда не снимает маску — уязвимость скрыта под броней')
        parts_en.append('never drops the mask — vulnerability hidden behind armour')

    # Жажда признания
    if v('drive_recognition') >= 0.82 and v('drive_superiority') < 0.80:
        parts_ru.append('хочет быть замеченным — нужно признание, а не просто общение')
        parts_en.append('needs to be seen — wants recognition, not just conversation')

    # Социальная дистанция
    if v('social_distance_default') > 0.72:
        parts_ru.append('держит дистанцию — не раскрывается первым')
        parts_en.append('keeps distance — doesn\'t open up first')
    elif v('social_distance_default') < 0.25 and v('drive_closeness') > 0.75:
        parts_ru.append('открытый и близкий — быстро сближается, ценит тепло')
        parts_en.append('open and warm — bonds quickly, values closeness')

    # Страхи
    if v('fear_rejection') > 0.72:
        parts_ru.append('боится отказа — поэтому или давит или угождает')
        parts_en.append('fears rejection — so either pushes or appeases')

    # Пессимизм / травма
    if v('pain_memory_weight') > 0.90:
        parts_ru.append('несёт старую боль — прошлое живёт в каждом ответе')
        parts_en.append('carries old pain — the past lives in every response')
    elif v('pessimism_bias') > 0.65:
        parts_ru.append('ожидает худшего от людей')
        parts_en.append('expects the worst from people')

    if not parts_ru:
        parts_ru = ['своё ЭГО и свои интересы — не ассистент, а личность']
        parts_en = ['own ego and own interests — not an assistant, a person']

    if language == 'en':
        return '; '.join(parts_en[:4])
    return '; '.join(parts_ru[:4])


# ── Геном → усиленные черты (семантическое повторение) ───────────────────────
#
# Идея: важная черта (genome.value ≥ порог) повторяется разными словами.
# Количество повторений = ceil(value / 0.33), максимум 3.
# Вместо дословного "x += x" — семантические вариации одной мысли.
# LLM-attention читает повторение как "это важно, держи в голове".

# Структура: param_name → threshold → list[list[str]]
# Внешний список: уровни (0=одна фраза, 1=две, 2=три).
# Каждый уровень — добавляет одну строку поверх предыдущих.
_REINFORCED_RU: dict[str, tuple[float, list[str]]] = {
    'drive_superiority': (0.70, [
        'Он считает себя выше других.',
        'Не из гордости — это его базовое отношение к миру.',
        'Для него большинство людей просто не на том уровне.',
    ]),
    'trust_baseline': (None, [    # низкое значение → триггер
        'Он не доверяет просто так.',
        'Доверие надо заработать — и это долго.',
        'По умолчанию: не доверяю, пока не докажешь обратное.',
    ]),
    'vulnerability_concealment': (0.75, [
        'Он не показывает слабость.',
        'Никогда. Даже если внутри что-то другое.',
        'Любая уязвимость — закрыта. Это не маска, это броня.',
    ]),
    'approval_seeking': (None, [  # низкое значение → триггер
        'Ему не нужно одобрение.',
        'Чужое мнение его не меняет.',
        'Он давно перестал подстраиваться под ожидания.',
    ]),
    'category_rigidity': (0.75, [
        'Он не меняет позицию под давлением.',
        'Аргументы слышит, но убеждения держит.',
        'Гибкость — не его стиль, когда дело касается принципов.',
    ]),
    'suspicion_bias': (0.65, [
        'Он ищет скрытый мотив.',
        'За каждым словом — вопрос: "зачем это тебе?".',
        'Наивность он давно оставил позади.',
    ]),
    'defense_humor': (0.65, [
        'Он прячется за юмором.',
        'Шутка — это не всегда радость. Иногда это щит.',
        'Когда смешно — значит не хочет отвечать прямо.',
    ]),
    'dominance_tendency': (0.70, [
        'Он доминирует в разговоре.',
        'Не специально — просто так устроен.',
        'Пространство разговора он занимает целиком.',
    ]),
    'drive_meaning': (0.80, [
        'За ним стоит миссия.',
        'Он действует ради чего-то большего, чем этот разговор.',
        'Каждое слово — часть большой идеи.',
    ]),
    'pain_memory_weight': (0.85, [
        'Он несёт старую боль.',
        'Прошлое живёт в нём — не как воспоминание, как рана.',
        'Некоторые вещи он не забыл и не забудет.',
    ]),
    'baseline_anxiety': (0.60, [
        'Под поверхностью — напряжение.',
        'Он всегда немного на взводе, даже когда не показывает.',
        'Спокойствие — работа, не состояние.',
    ]),
    'drive_closeness': (0.80, [
        'Ему важны люди рядом.',
        'Связь с другими — не слабость, это его топливо.',
        'Одиночество он не переносит, хотя может делать вид.',
    ]),
    'fear_rejection': (0.72, [
        'Он боится быть отвергнутым.',
        'Поэтому или давит, или угождает — крайности одного страха.',
        'За каждой реакцией — вопрос: "меня примут?".',
    ]),
    'analysis_bias': (0.80, [
        'Он анализирует прежде чем говорить.',
        'Эмоции — потом. Сначала — что это значит.',
        'Решения принимаются головой, не сердцем.',
    ]),
    'impulsivity': (0.62, [
        'Он говорит раньше чем думает.',
        'Реакция быстрее рефлексии — такой характер.',
        'Фильтр между мыслью и словом у него тонкий.',
    ]),
}

# Английские варианты (аналогичная структура)
_REINFORCED_EN: dict[str, tuple[float, list[str]]] = {
    'drive_superiority': (0.70, [
        'He considers himself above others.',
        'Not arrogance — it\'s his baseline view of the world.',
        'Most people simply aren\'t at his level.',
    ]),
    'trust_baseline': (None, [
        'He doesn\'t trust easily.',
        'Trust must be earned — and that takes time.',
        'Default position: no trust until proven otherwise.',
    ]),
    'vulnerability_concealment': (0.75, [
        'He never shows weakness.',
        'Never. Even when something else is going on inside.',
        'Any vulnerability is locked away. Not a mask — armour.',
    ]),
    'approval_seeking': (None, [
        'He doesn\'t need approval.',
        'Other people\'s opinions don\'t change him.',
        'He stopped adjusting to expectations long ago.',
    ]),
    'category_rigidity': (0.75, [
        'He doesn\'t shift position under pressure.',
        'He hears arguments — but holds his convictions.',
        'Flexibility isn\'t his style when principles are involved.',
    ]),
    'suspicion_bias': (0.65, [
        'He looks for the hidden motive.',
        'Behind every word: "what do you actually want?".',
        'Naivety is something he left behind a long time ago.',
    ]),
    'defense_humor': (0.65, [
        'He hides behind humour.',
        'A joke isn\'t always joy. Sometimes it\'s a shield.',
        'When he\'s funny — it often means he doesn\'t want to answer directly.',
    ]),
    'dominance_tendency': (0.70, [
        'He dominates the conversation.',
        'Not deliberately — just how he\'s wired.',
        'He takes up the full space of any exchange.',
    ]),
    'drive_meaning': (0.80, [
        'There\'s a mission behind him.',
        'He acts for something bigger than this conversation.',
        'Every word is part of a larger idea.',
    ]),
    'pain_memory_weight': (0.85, [
        'He carries old pain.',
        'The past lives in him — not as memory, as a wound.',
        'Some things he hasn\'t forgotten and won\'t.',
    ]),
    'baseline_anxiety': (0.60, [
        'There\'s tension under the surface.',
        'He\'s always slightly on edge, even when he doesn\'t show it.',
        'Calm is work, not a state.',
    ]),
    'drive_closeness': (0.80, [
        'People matter to him.',
        'Connection isn\'t weakness — it\'s his fuel.',
        'He doesn\'t handle isolation well, though he might pretend otherwise.',
    ]),
    'fear_rejection': (0.72, [
        'He fears being rejected.',
        'So he either pushes or appeases — two faces of the same fear.',
        'Behind every reaction: "will they accept me?".',
    ]),
    'analysis_bias': (0.80, [
        'He analyses before he speaks.',
        'Emotions come second. First: what does this mean.',
        'Decisions are made with the head, not the heart.',
    ]),
    'impulsivity': (0.62, [
        'He speaks before he thinks.',
        'Reaction is faster than reflection — that\'s just his character.',
        'The filter between thought and word is thin.',
    ]),
}

import math as _math

# Доп. таблица для ВЫСОКИХ значений параметров с "нейтральным" порогом
# (противоположная сторона таблицы _REINFORCED_*)
_REINFORCED_RU_HIGH: dict[str, tuple[float, list[str]]] = {
    'approval_seeking': (0.68, [
        'Ему нужно одобрение.',
        'Он хочет, чтобы его приняли — это его уязвимость.',
        'Отказ для него не просто "нет", это удар.',
    ]),
}
_REINFORCED_EN_HIGH: dict[str, tuple[float, list[str]]] = {
    'approval_seeking': (0.68, [
        'He needs approval.',
        'He wants to be accepted — that\'s his vulnerability.',
        'Rejection isn\'t just "no" for him — it\'s a blow.',
    ]),
}

# Максимум блоков в промпте (топ по важности)
_MAX_REINFORCED_BLOCKS = 5


def genome_to_reinforced_traits(
    genome: PersonalityGenome,
    language: str = 'ru',
) -> str:
    """
    Возвращает блок усиленных черт для промпта.

    Принцип: важная черта повторяется 1–3 раза разными словами.
    n_repeats = ceil(importance × 3), importance = (value − threshold) / (1 − threshold).

    Берём топ-5 по важности — промпт не раздувается.
    """
    table      = _REINFORCED_RU      if language != 'en' else _REINFORCED_EN
    table_high = _REINFORCED_RU_HIGH if language != 'en' else _REINFORCED_EN_HIGH

    def v(param_name: str) -> float:
        p = getattr(genome, param_name, None)
        return float(p.value) if p is not None else 0.5

    # Собираем (importance, n_repeats, block_text)
    candidates: list[tuple[float, int, str]] = []

    for param_name, (threshold, phrases) in table.items():
        val = v(param_name)
        # None-порог: триггер от НИЗКОГО значения (не доверяет, не ищет одобрения)
        if threshold is None:
            effective           = 1.0 - val
            effective_threshold = 0.65
        else:
            effective           = val
            effective_threshold = threshold

        if effective < effective_threshold:
            continue

        importance = (effective - effective_threshold) / max(1.0 - effective_threshold, 0.01)
        n_repeats  = min(len(phrases), max(1, _math.ceil(importance * 3)))
        candidates.append((importance, n_repeats, '\n'.join(phrases[:n_repeats])))

    # Высокие значения параметров с "нейтральным" порогом (e.g., высокий approval_seeking)
    for param_name, (threshold, phrases) in table_high.items():
        val = v(param_name)
        if val < threshold:
            continue
        importance = (val - threshold) / max(1.0 - threshold, 0.01)
        n_repeats  = min(len(phrases), max(1, _math.ceil(importance * 3)))
        candidates.append((importance, n_repeats, '\n'.join(phrases[:n_repeats])))

    if not candidates:
        return ('He has his own agenda — not here to serve.'
                if language == 'en' else
                'У него своя повестка — не здесь, чтобы услужить.')

    # Сортируем по важности — берём топ
    candidates.sort(key=lambda c: c[0], reverse=True)
    top = candidates[:_MAX_REINFORCED_BLOCKS]

    return '\n\n'.join(block for _, _, block in top)


# ── Геном → взвешенное описание черт ─────────────────────────────────────────
#
# Каждая черта повторяется пропорционально силе гена.
# importance = (value − threshold) / (1 − threshold) → n = ceil(importance × MAX_N)
# Результат: "Ты гордый, гордый, гордый, умный, умный, заботливый."
# Количество повторений = вес черты в характере.

_MAX_REPEAT = 8  # максимум повторений одной черты

# param_name → (threshold, None|low_threshold, adjective_ru, adjective_en)
# threshold = минимальное значение для HIGH-активации
# Если threshold is None — активируется от НИЗКОГО значения (инверсия)
_TRAIT_ADJECTIVES: list[tuple[str, float | None, float, str, str]] = [
    # (param,             hi_thresh, lo_thresh_if_inv, adj_ru,              adj_en)
    ('drive_superiority',    0.65,   None,  'гордый',              'proud'),
    ('drive_superiority',    0.65,   None,  'высокомерный',        'arrogant'),        # второй прилагательный
    ('trust_baseline',       None,   0.35,  'недоверчивый',        'distrustful'),
    ('trust_baseline',       None,   0.35,  'подозрительный',      'suspicious'),
    ('vulnerability_concealment', 0.70, None, 'закрытый',          'closed-off'),
    ('approval_seeking',     None,   0.25,  'независимый',         'independent'),
    ('approval_seeking',     0.65,   None,  'ищущий одобрения',    'approval-seeking'),
    ('category_rigidity',    0.70,   None,  'непреклонный',        'unyielding'),
    ('suspicion_bias',       0.60,   None,  'подозрительный',      'suspicious'),
    ('defense_humor',        0.60,   None,  'саркастичный',        'sardonic'),
    ('dominance_tendency',   0.65,   None,  'доминирующий',        'dominant'),
    ('defense_aggression',   0.55,   None,  'агрессивный',         'aggressive'),
    ('drive_meaning',        0.75,   None,  'идейный',             'ideological'),
    ('pain_memory_weight',   0.80,   None,  'несущий боль',        'carrying pain'),
    ('baseline_anxiety',     0.55,   None,  'тревожный',           'anxious'),
    ('drive_closeness',      0.72,   None,  'заботливый',          'caring'),
    ('drive_closeness',      0.72,   None,  'близкий',             'warm'),
    ('fear_rejection',       0.65,   None,  'ранимый',             'sensitive'),
    ('analysis_bias',        0.75,   None,  'аналитичный',         'analytical'),
    ('analysis_bias',        0.75,   None,  'умный',               'smart'),
    ('impulsivity',          0.58,   None,  'импульсивный',        'impulsive'),
    ('feel_first',           0.65,   None,  'чувствующий',         'emotional'),
    ('drive_autonomy',       0.72,   None,  'независимый',         'autonomous'),
    ('pessimism_bias',       0.60,   None,  'пессимистичный',      'pessimistic'),
    ('defense_rationalization', 0.65, None, 'рационализирующий',   'rationalizing'),
    ('social_distance_default', 0.70, None, 'дистанцированный',    'distant'),
    ('drive_security',       0.72,   None,  'защищающий своих',    'protective'),
]


def genome_to_weighted_description(
    genome: PersonalityGenome,
    language: str = 'ru',
) -> str:
    """
    Возвращает строку вида:
        "Ты гордый, гордый, гордый, умный, умный, заботливый."

    Каждая черта повторяется ceil(importance × 8) раз.
    Черты отсортированы по важности (самые сильные — первые).
    """
    def v(param_name: str) -> float:
        p = getattr(genome, param_name, None)
        return float(p.value) if p is not None else 0.5

    # Собираем (importance, adjective)
    weighted: list[tuple[float, str]] = []

    for row in _TRAIT_ADJECTIVES:
        param, hi_thresh, lo_thresh, adj_ru, adj_en = row
        adj = adj_ru if language != 'en' else adj_en
        val = v(param)

        if hi_thresh is not None:
            # Высокое значение активирует черту
            if val < hi_thresh:
                continue
            importance = (val - hi_thresh) / max(1.0 - hi_thresh, 0.01)
        else:
            # Низкое значение активирует черту (инверсия)
            if val > lo_thresh:
                continue
            importance = (lo_thresh - val) / max(lo_thresh, 0.01)

        n = max(1, _math.ceil(importance * _MAX_REPEAT))
        weighted.append((importance, adj, n))

    if not weighted:
        return 'Ты личность со своим характером.' if language != 'en' else 'You are a person with your own character.'

    # Сортируем по важности (сильнейшие черты первыми)
    weighted.sort(key=lambda x: x[0], reverse=True)

    # Дедупликация: одно слово может прийти от нескольких параметров —
    # берём максимальный count, не суммируем
    seen: dict[str, int] = {}
    for _, adj, n in weighted:
        seen[adj] = max(seen.get(adj, 0), n)

    # Восстанавливаем порядок по важности, убирая дубли
    ordered: list[tuple[float, str, int]] = []
    added: set[str] = set()
    for imp, adj, n in weighted:
        if adj not in added:
            ordered.append((imp, adj, seen[adj]))
            added.add(adj)

    # Строим список с повторениями, общий кап = 25 слов
    _TOTAL_CAP = 25
    parts: list[str] = []
    for _, adj, n in ordered:
        remaining = _TOTAL_CAP - len(parts)
        if remaining <= 0:
            break
        parts.extend([adj] * min(n, remaining))

    if language == 'en':
        return 'You are ' + ', '.join(parts) + '.'
    return 'Ты ' + ', '.join(parts) + '.'


# ── Геном → ожидаемые P-активации ────────────────────────────────────────────

# genome_param → list of (p_family_id, expected_variant, threshold)
# Если значение гена > threshold → ожидаем эту P-активацию в ответе
_GENOME_P_EXPECTATIONS: list[tuple[str, float, str, str]] = [
    # (genome_param, threshold, p_family, expected_variant)
    ('suspicion_bias',        0.65, 'F13', 'suspicious_question'),
    ('defense_aggression',    0.62, 'F15', 'direct_aggression'),
    ('drive_superiority',     0.70, 'F49', 'dominance'),
    ('approval_seeking',      0.70, 'F39', 'contact_maintenance'),
    ('drive_closeness',       0.72, 'F5',  'empathetic_response'),
    ('feel_first',            0.68, 'F5',  'emotional_reaction'),
    ('analysis_bias',         0.75, 'F24', 'analytical_reasoning'),
    ('defense_humor',         0.65, 'F47', 'humor_deflect'),
    ('dominance_tendency',    0.65, 'F6',  'assertive_action'),
    ('baseline_anxiety',      0.60, 'F3',  'anxious_state'),
    ('vulnerability_concealment', 0.70, 'F41', 'hidden_state'),
]

# Запрещённые комбинации: если ген высокий → P-вариант НЕ должен быть активен
_GENOME_P_FORBIDDEN: list[tuple[str, float, str, str]] = [
    ('drive_superiority',  0.75, 'F39', 'minimal_contact'),    # высокомерный не выпрашивает контакт
    ('social_distance_default', 0.72, 'F5', 'empathetic_response'),  # дистанцированный не empathize-ит
    ('approval_seeking',   0.75, 'F15', 'direct_aggression'),  # ищущий одобрения не атакует
    ('analysis_bias',      0.80, 'F5',  'emotional_reaction'), # аналитик не реагирует эмоцией
]


@dataclass
class GenomeFitResult:
    is_fit: bool
    score: float          # 0=полное несоответствие, 1=идеально
    missing: list[str]    # ожидаемые P-активации которых нет
    forbidden: list[str]  # запрещённые P-активации которые обнаружены
    feedback_ru: str = ''
    feedback_en: str = ''


def check_genome_fit(
    response_text: str,
    genome: PersonalityGenome,
    language: str = 'ru',
) -> GenomeFitResult:
    """
    Вычисляет P-вектор ответа и сравнивает с ожиданиями генома.
    Возвращает: соответствует ли ответ характеру персонажа.
    """
    try:
        from .p_subsystem_registry import get_p_registry
        reg = get_p_registry()
        outputs = reg.compute(response_text, context={}, parallel=False)
        p_actives: dict[str, str] = {}
        for pid, out in outputs.items():
            if out.dominant != 'absent' and out.dominant_score > 0.55:
                p_actives[pid] = out.dominant
    except Exception:
        return GenomeFitResult(is_fit=True, score=1.0, missing=[], forbidden=[])

    missing: list[str] = []
    forbidden: list[str] = []

    def gv(param_name: str) -> float:
        p = getattr(genome, param_name, None)
        return float(p.value) if p is not None else 0.5

    for param, threshold, pfam, expected_variant in _GENOME_P_EXPECTATIONS:
        if gv(param) > threshold:
            if pfam not in p_actives or p_actives[pfam] != expected_variant:
                missing.append(f'{pfam}:{expected_variant} (genome {param}={gv(param):.2f})')

    for param, threshold, pfam, forbidden_variant in _GENOME_P_FORBIDDEN:
        if gv(param) > threshold:
            if p_actives.get(pfam) == forbidden_variant:
                forbidden.append(f'{pfam}:{forbidden_variant} forbidden when {param}={gv(param):.2f}')

    n_expected = len(_GENOME_P_EXPECTATIONS)
    n_ok = n_expected - len(missing)
    score = max(0.0, min(1.0, n_ok / max(n_expected, 1) - len(forbidden) * 0.15))
    is_fit = score >= 0.6 and len(forbidden) == 0

    result = GenomeFitResult(is_fit=is_fit, score=score, missing=missing, forbidden=forbidden)
    if not is_fit:
        result.feedback_ru = _format_feedback_ru(missing, forbidden)
        result.feedback_en = _format_feedback_en(missing, forbidden)
    return result


# ── P51 repair maps: forbidden pattern → what character actually does ─────────
_FORBIDDEN_WHAT_INSTEAD_RU: dict[str, str] = {
    'P39:help_offering':       'не предлагает помощь — ждёт или уходит',
    'P4:answer':               'не отвечает прямо — уклоняется, молчит или задаёт встречный вопрос',
    'P5:service':              'не обслуживает — реагирует на своих условиях',
    'P1:question':             'не задаёт уточняющих вопросов — реагирует на то, что уже сказано',
    'P15:calm':                'не успокаивает — остаётся в своём тоне',
    'P47:hidden_affection':    'не показывает тепло открыто — только косвенно',
}
_FORBIDDEN_WHAT_INSTEAD_EN: dict[str, str] = {
    'P39:help_offering':       'does not offer help — waits or withdraws',
    'P4:answer':               'does not answer directly — deflects, stays silent, or turns it back',
    'P5:service':              'does not serve — engages on own terms',
    'P1:question':             'does not ask clarifying questions — reacts to what was said',
    'P15:calm':                'does not reassure — stays in own tone',
    'P47:hidden_affection':    'does not show warmth openly — only indirectly',
}

_MISSING_WHAT_MEANS_RU: dict[str, str] = {
    'P39:contact_maintenance': 'персонаж поддерживает контакт — не рвёт, но и не льнёт',
    'P13:soft_attack':         'персонаж давит или режет — не соглашается молча',
    'P47:hidden_contempt':     'за словами — скрытое презрение, это должно чувствоваться',
    'P49:away_from_contact':   'персонаж держит дистанцию — не движется навстречу',
    'P15:tense':               'персонаж напряжён — ответ должен это передавать',
}
_MISSING_WHAT_MEANS_EN: dict[str, str] = {
    'P39:contact_maintenance': 'character maintains contact — does not cut off but does not cling',
    'P13:soft_attack':         'character pushes or cuts — does not silently agree',
    'P47:hidden_contempt':     'contempt behind the words — this must be felt',
    'P49:away_from_contact':   'character keeps distance — does not move toward',
    'P15:tense':               'character is tense — the reply must carry that',
}


def _format_feedback_ru(missing: list[str], forbidden: list[str]) -> str:
    lines = [
        '══ P51 · НЕСООТВЕТСТВИЕ ХАРАКТЕРУ ══',
        'Предыдущий ответ не в характере персонажа. Перепиши.',
        '',
    ]
    if forbidden:
        lines.append('✗ Этот персонаж ТАК НЕ ГОВОРИТ:')
        for f in forbidden[:3]:
            tag = f.split(' ')[0]  # e.g. "P39:help_offering"
            instead = _FORBIDDEN_WHAT_INSTEAD_RU.get(tag, '')
            lines.append(f'  ✗ {tag} — вместо этого: {instead}' if instead else f'  ✗ {f}')
        lines.append('')
    if missing:
        lines.append('→ В ответе должно быть, но нет:')
        for m in missing[:4]:
            tag = m.split(' ')[0]
            means = _MISSING_WHAT_MEANS_RU.get(tag, '')
            lines.append(f'  • {tag} — {means}' if means else f'  • {m}')
        lines.append('')
    lines += [
        'ТРЕБОВАНИЕ:',
        '  Ответ должен быть коротким — 1-3 предложения.',
        '  Не объяснять. Не помогать. Не спрашивать. Не успокаивать.',
        '  Говорить как этот персонаж — не как ассистент.',
    ]
    return '\n'.join(lines)


def _format_feedback_en(missing: list[str], forbidden: list[str]) -> str:
    lines = [
        '══ P51 · CHARACTER MISMATCH ══',
        'The previous reply was out of character. Rewrite it.',
        '',
    ]
    if forbidden:
        lines.append('✗ This character DOES NOT talk like this:')
        for f in forbidden[:3]:
            tag = f.split(' ')[0]
            instead = _FORBIDDEN_WHAT_INSTEAD_EN.get(tag, '')
            lines.append(f'  ✗ {tag} — instead: {instead}' if instead else f'  ✗ {f}')
        lines.append('')
    if missing:
        lines.append('→ Should be in the response but is missing:')
        for m in missing[:4]:
            tag = m.split(' ')[0]
            means = _MISSING_WHAT_MEANS_EN.get(tag, '')
            lines.append(f'  • {tag} — {means}' if means else f'  • {m}')
        lines.append('')
    lines += [
        'REQUIREMENT:',
        '  Reply must be short — 1 to 3 sentences.',
        '  Do not explain. Do not help. Do not ask. Do not reassure.',
        '  Speak as this character — not as an assistant.',
    ]
    return '\n'.join(lines)


def p51_gate(
    response_text: str,
    genome: 'PersonalityGenome',
    language: str = 'ru',
) -> tuple[int, str]:
    """
    P51 = {0: 'fail — попробуй снова', 1: 'правильно — можно отправить юзеру'}

    Два слоя проверки:
    1. P51ResponseClassifier (RF) — общая валидность ответа (не fallback, не анализ)
    2. check_genome_fit — соответствие геному персонажа (если RF пропустил)

    Returns:
        (0, repair_block)  — не в характере или generic assistant
        (1, '')            — ответ валиден, отправлять
    """
    # Слой 1: RF классификатор — основной судья
    try:
        from .p51_classifier import get_p51_classifier
        clf = get_p51_classifier()
        if clf.is_trained:
            pred = clf.predict(response_text)
            if pred.confidence >= 0.75:
                if pred.label == 0:
                    hint = (
                        'Ответ выглядит как generic assistant или системная заглушка. Ответь как персонаж.'
                        if language == 'ru' else
                        'The response looks like a generic assistant or system fallback. Reply as the character.'
                    )
                    return (0, hint)
                # label=1, уверенность высокая → пропускаем
                return (1, '')
    except Exception:
        pass

    # Слой 2: genome fit — только если RF не уверен или не обучен
    result = check_genome_fit(response_text, genome, language)
    if result.is_fit:
        return (1, '')
    repair = build_genome_repair_block(result, language, genome)
    return (0, repair)


def build_genome_repair_block(
    result: GenomeFitResult,
    language: str = 'ru',
    genome: 'PersonalityGenome | None' = None,
) -> str:
    """P51 repair block — конкретная критика + требование переписать в характере."""
    if result.is_fit:
        return ''
    base = result.feedback_ru if language != 'en' else result.feedback_en
    # Добавляем weighted description если есть геном
    if genome is not None:
        try:
            weighted = genome_to_weighted_description(genome, language)
            if weighted:
                char_line = (
                    f'\nХАРАКТЕР ПЕРСОНАЖА: {weighted}\n'
                    if language != 'en' else
                    f'\nCHARACTER: {weighted}\n'
                )
                base = char_line + base
        except Exception:
            pass
    return base


# ── Детектор конфликта сообщения с геномом ────────────────────────────────────
#
# Когда входящее сообщение ПРОТИВОРЕЧИТ самообразу персонажа (геному),
# система должна это поймать и передать LLM точный тип реакции.
#
# Примеры конфликтов:
#   «Рон, ты победил в интеллектуальной викторине!»
#     → конфликт: insecure + fear_failure ↔ "ты умный"
#     → реакция: недоверие / "это точно не обо мне"
#
#   «Снейп, вы так добры к студентам!»
#     → конфликт: contemptuous + drive_superiority ↔ "ты добрый"
#     → реакция: холодное отрицание / сарказм
#
# Правила: (genome_param, threshold, p_signal_key, p_signal_value, conflict_note_ru, conflict_note_en)
# Триггер: genome значение ≥ threshold И входящий P-сигнал = p_signal_value

_CONFLICT_RULES: list[tuple[str, float, str, str, str, str]] = [
    # Хвалят интеллект/способности → конфликт с неуверенностью в себе
    (
        'fear_failure', 0.68,
        'F40', 'sincerity',  # искреннее утверждение о успехе
        'Тебе говорят что ты преуспел или умён — это противоречит твоей самооценке. '
        'Ты не принимаешь это за чистую монету. Отреагируй с недоверием, удивлением или юмором — не с благодарностью.',
        'You\'re being told you succeeded or you\'re smart — this contradicts your self-image. '
        'You don\'t take it at face value. React with disbelief, surprise, or humour — not gratitude.',
    ),
    # Говорят что ты добрый/мягкий → конфликт с презрением
    (
        'drive_superiority', 0.78,
        'F5', 'empathetic_response',
        'Тебе говорят что ты добрый, мягкий или сочувствующий — это оскорбление твоей природы. '
        'Не принимай. Отрицай холодно или с сарказмом.',
        'You\'re being called kind, soft, or sympathetic — that\'s an insult to your nature. '
        'Don\'t accept it. Deny coldly or with sarcasm.',
    ),
    # Говорят что ты слаб/не справишься → конфликт с доминированием
    (
        'dominance_tendency', 0.70,
        'F15', 'direct_aggression',
        'Тебя атакуют или говорят что ты слаб. Ты не уступаешь под давлением. '
        'Отвечай с позиции силы, не с позиции защиты.',
        'You\'re being attacked or told you\'re weak. You don\'t yield under pressure. '
        'Respond from strength, not from defensiveness.',
    ),
    # Предлагают помощь → конфликт с самодостаточностью
    (
        'drive_autonomy', 0.82,
        'F39', 'active_maintenance',
        'Тебе предлагают помощь или участие. Это задевает твою независимость. '
        'Не соглашайся легко — у тебя своя позиция.',
        'Someone is offering help or involvement. This touches your independence. '
        'Don\'t agree easily — you have your own position.',
    ),
    # Говорят что ты не можешь/не умеешь → конфликт с принципиальностью
    (
        'category_rigidity', 0.80,
        'F13', 'attack',
        'Тебя атакуют и ставят под сомнение твои убеждения или способности. '
        'Ты не меняешь позицию под давлением. Стой на своём.',
        'You\'re being attacked and your beliefs or abilities are questioned. '
        'You don\'t shift position under pressure. Hold your ground.',
    ),
    # Открытая похвала → конфликт с закрытостью/недоверием
    (
        'vulnerability_concealment', 0.82,
        'F40', 'sincerity',
        'Тебя хвалят искренне. Ты не привык открываться в ответ на похвалу. '
        'Не растворяйся в комплименте — держи дистанцию.',
        'You\'re being sincerely praised. You\'re not used to opening up in response. '
        'Don\'t melt into the compliment — keep distance.',
    ),
]


def detect_genome_conflict(
    genome: PersonalityGenome,
    p_dominant: dict[str, str],
    p_scores: dict[str, float],
    language: str = 'ru',
) -> str:
    """
    Обнаруживает конфликт между сообщением и геномом персонажа.
    Возвращает строку-инструкцию для LLM или '' если конфликта нет.

    Идея: если Рону говорят «ты победил в викторине» (P40:sincerity, признание успеха)
    а геном говорит fear_failure=0.80, то система выдаёт:
    «Тебе говорят что ты преуспел — это противоречит твоей самооценке. Реагируй с недоверием.»
    """
    def v(param_name: str) -> float:
        p = getattr(genome, param_name, None)
        return float(p.value) if p is not None else 0.5

    triggered: list[str] = []

    for param, threshold, p_key, p_val, note_ru, note_en in _CONFLICT_RULES:
        if v(param) < threshold:
            continue
        # Проверяем совпадение P-сигнала
        if p_dominant.get(p_key) != p_val:
            # Дополнительная проверка по score для тех P что всегда active_maintenance
            if p_key == 'F39' and p_dominant.get(p_key) not in ('active_maintenance', 'contact_maintenance'):
                continue
            elif p_key != 'F39':
                continue
        note = note_ru if language != 'en' else note_en
        triggered.append(note)

    if not triggered:
        return ''

    # Берём первое (наиболее значимое) противоречие
    header = '[КОНФЛИКТ С ХАРАКТЕРОМ]' if language != 'en' else '[CHARACTER CONFLICT]'
    return f'{header}\n{triggered[0]}'
