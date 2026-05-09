"""
user_affect_model.py — модель эмоционального состояния пользователя.

Отвечает на вопрос: «Что сейчас с пользователем?»
  — злится, грустит, провоцирует, скучает, тестирует систему,
    прячет что-то за "да ладно", или просто спрашивает?

Архитектура: PRIMARY → variant-based (p_dominant),
             FALLBACK  → score-based (p_scores),
             CONTEXT   → история P-векторов из DialogContextMatrix.

Реальные наблюдения из P-семей:
  • P15 ('overloaded') активен почти всегда — не информативен
  • P13:attack   → прямая агрессия/нападение
  • P33:protective_distancing → отгораживается, закрывается
  • P21:passive_aggressive → пассивная агрессия ("да ладно", "хорошо")
  • P47:hidden_contempt     → презрение под поверхностью
  • P47:hidden_reproach     → обида/претензия под поверхностью
  • P47:hidden_affection    → скрытая симпатия/интерес
  • P39:active_maintenance  → хочет продолжать
  • P39:minimal_contact     → еле держит контакт
  • P35:slow_escalation     → нарастающее напряжение
  • P40:sincerity           → искренность, говорит прямо
  • P41:masking             → скрывает истинное
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ─── Состояния ────────────────────────────────────────────────────────────────

# (ru_label, en_label, valence [-1..+1])
_STATE_META: dict[str, tuple[str, str, float]] = {
    'hostile':          ('агрессивный/враждебный',      'hostile',              -1.0),
    'passive_agressive':('пассивно-агрессивный',        'passive-aggressive',   -0.7),
    'contemptuous':     ('презрительный',               'contemptuous',         -0.7),
    'disappointed':     ('обиженный/разочарованный',    'disappointed',         -0.6),
    'frustrated':       ('раздражённый/на грани',       'frustrated',           -0.5),
    'withdrawn':        ('закрылся/уходит',             'withdrawn',            -0.5),
    'stressed':         ('напряжён/тревожен',           'stressed',             -0.4),
    'bored':            ('скучает/безразличен',         'bored',                -0.2),
    'testing':          ('проверяет/провоцирует',       'testing',              -0.1),
    'neutral':          ('нейтральный',                 'neutral',               0.0),
    'curious':          ('любопытен/вовлечён',          'curious',              +0.6),
    'seeking_help':     ('ищет ответа/помощи',          'seeking help',         +0.5),
    'warm':             ('тёплый/дружелюбный',          'warm/friendly',        +0.8),
}


# ─── Классификация по P-dominant variants (PRIMARY) ──────────────────────────

def _classify_from_variants(
    p_dominant: dict[str, str],
    p_scores: dict[str, float],
    text: str = '',
) -> tuple[str, float] | None:
    """
    Использует конкретные P-варианты как прямые улики.
    Возвращает (state, intensity) или None → fallback к score-based.
    """
    p13  = p_dominant.get('F13', '')
    p21  = p_dominant.get('F21', '')
    p33  = p_dominant.get('F33', '')
    p35  = p_dominant.get('F35', '')
    p39  = p_dominant.get('F39', '')
    p40  = p_dominant.get('F40', '')
    p47  = p_dominant.get('F47', '')

    # ── Прямые сигналы ────────────────────────────────────────────────────────

    # P13:attack = нападение. Самый сильный сигнал враждебности.
    if p13 == 'attack':
        # Если при этом высокий контакт — скорее проверяет/провоцирует, не уходит
        if p39 in ('active_maintenance', 'contact_maintenance'):
            return 'testing', 0.82
        return 'hostile', 0.90

    # P33:protective_distancing = отгородился физически/эмоционально
    if p33 == 'protective_distancing':
        if p39 in ('active_maintenance', 'contact_maintenance'):
            return 'testing', 0.72    # закрылся, но хочет реакции
        return 'withdrawn', 0.78

    # P21:passive_aggressive = "да ладно", "хорошо, понял" с напряжением
    if p21 == 'passive_aggressive':
        return 'passive_agressive', 0.75

    # ── P47 variants — что скрыто под словами ────────────────────────────────

    # Презрение под нейтральным/вопросительным текстом
    if p47 == 'hidden_contempt':
        # Если при этом P39 активен — провоцирует, а не просто презирает
        if p39 in ('active_maintenance',):
            return 'contemptuous', 0.72
        return 'contemptuous', 0.68

    # Обида / скрытая претензия ("ты же знаешь", "всё понятно")
    if p47 == 'hidden_reproach':
        if p35 == 'slow_escalation':
            return 'frustrated', 0.70
        return 'disappointed', 0.65

    # Скрытая симпатия + активный контакт → искренний интерес
    if p47 == 'hidden_affection':
        if p39 in ('active_maintenance', 'contact_maintenance'):
            return 'curious', 0.82
        return 'warm', 0.70

    # ── P39 variants — уровень контакта ──────────────────────────────────────

    # Активное поддержание диалога
    if p39 == 'active_maintenance':
        if p35 == 'slow_escalation':
            return 'frustrated', 0.58   # хочет продолжать но напряжён
        if p40 == 'sincerity':
            return 'seeking_help', 0.72
        return 'curious', 0.65

    # Еле держит контакт
    if p39 == 'minimal_contact':
        if p35 == 'slow_escalation':
            return 'frustrated', 0.62
        # Короткий текст + минимальный контакт = скучает или уходит
        if len(text.strip()) <= 8:
            return 'bored', 0.55
        return 'withdrawn', 0.50

    # ── P40:sincerity ─────────────────────────────────────────────────────────
    # Прямая искренность без агрессии = говорит что думает
    if p40 == 'sincerity' and p13 != 'attack' and p33 != 'protective_distancing':
        return 'seeking_help', 0.60

    return None   # → fallback


# ─── Fallback: score-based (когда variants не дали чёткого сигнала) ──────────

# Игнорируем P15 ('overloaded') и P4 ('avoidance'), P22 ('dominance') —
# они почти всегда активны и не несут дифференцирующей информации.
_IGNORED_FOR_SCORING = {'F15', 'F4', 'F22', 'F41', 'F42', 'F43', 'F32', 'F9', 'F1'}

def _classify_from_scores(
    p_scores: dict[str, float],
    p_dominant: dict[str, str],
) -> tuple[str, float]:
    """Fallback: взвешенная сумма информативных P-сигналов."""

    def s(pid: str) -> float:
        if pid in _IGNORED_FOR_SCORING:
            return 0.0
        return float(p_scores.get(pid, 0.0))

    anx  = (s('F3') + s('F2')) / 2
    fear = s('F37')
    cont = s('F39')
    empa = s('F5')
    hidd = (s('F47') + s('F41')) / 2
    ques = (s('F8') + s('F38')) / 2
    escl = s('F35')
    aggr = s('F13')   # P13 score (not variant) as backup
    dist = s('F33')

    # Score каждого состояния
    scores: dict[str, float] = {
        'hostile':           aggr * 0.6 + dist * 0.3 - cont * 0.4,
        'frustrated':        escl * 0.45 + anx * 0.30 + aggr * 0.15,
        'stressed':          anx * 0.55 + fear * 0.30 - aggr * 0.20,
        'withdrawn':         dist * 0.50 + fear * 0.25 - cont * 0.35,
        'curious':           cont * 0.45 + ques * 0.35 + empa * 0.15,
        'seeking_help':      ques * 0.50 + cont * 0.25,
        'warm':              empa * 0.50 + cont * 0.35,
        'bored':             max(0, 0.25 - cont - aggr),
        'neutral':           0.20,
    }

    best_state = max(scores, key=lambda k: scores[k])
    best_score = scores[best_state]
    intensity = max(0.0, min(1.0, best_score + 0.30))  # смещение вверх

    if best_score < 0.12:
        return 'neutral', 0.25

    return best_state, round(intensity, 3)


# ─── Детектор скрытого состояния ─────────────────────────────────────────────

def _detect_hidden(
    state: str,
    p_dominant: dict[str, str],
    p_scores: dict[str, float],
    text: str,
    gap_seconds: float = 0.0,
) -> str:
    """
    Ищет противоречие: поверхностный тон vs скрытый P-сигнал.
    Возвращает '' или метку скрытого состояния.
    """
    p47  = p_dominant.get('F47', '')
    p21  = p_dominant.get('F21', '')
    p35  = p_dominant.get('F35', '')
    p40  = p_dominant.get('F40', '')
    p39  = p_dominant.get('F39', '')
    p13  = p_dominant.get('F13', '')

    text_lower = text.lower().strip()
    text_len   = len(text_lower)

    # "Да ладно" / "хорошо" / "понял" + hidden_reproach или passive_aggressive
    # = согласие словами, но злость под поверхностью
    appeasement = any(m in text_lower for m in (
        'ладно', 'хорошо', 'понял', 'ок', 'окей', 'ясно', 'ok', 'fine', 'sure', 'whatever', 'noted',
    ))
    if appeasement and (p47 in ('hidden_reproach', 'hidden_contempt') or p21 == 'passive_aggressive'):
        return 'masked_anger'

    # Шутит / саркастичен, но P35 escalation или P47:hidden_contempt
    if state in ('testing', 'curious') and (p35 == 'slow_escalation' or p47 == 'hidden_contempt'):
        return 'contempt_behind_humor'

    # Задаёт вопрос (P1:question) но P39:minimal_contact + P47 present → не за ответом
    p1 = p_dominant.get('F1', '')
    if p1 == 'question' and p39 == 'minimal_contact' and p47:
        return 'rhetorical_exit'   # "ты вообще слышишь?" — не вопрос, а уход

    # Активный контакт + P13:attack = провокация, не реальная злость
    if p13 == 'attack' and p39 in ('active_maintenance',):
        return 'provocative_testing'

    # Очень короткий ответ + нет aggression + нет contact = пассивный выход
    if text_len <= 5 and p39 not in ('active_maintenance', 'contact_maintenance') and p13 != 'attack':
        return 'passive_exit'

    # Долгая пауза (> 5 мин) перед ответом + P35 escalation
    if gap_seconds > 300 and p35 == 'slow_escalation':
        return 'slow_boil'         # долго копилось — теперь выходит

    return ''


# ─── Тренд ───────────────────────────────────────────────────────────────────

def _compute_trend(states: list[str]) -> str:
    if len(states) < 2:
        return 'stable'
    valences = [_STATE_META.get(s, ('', '', 0.0))[2] for s in states[-6:]]
    if len(valences) < 2:
        return 'stable'

    slope = (valences[-1] - valences[0]) / max(len(valences) - 1, 1)
    spread = max(valences) - min(valences)

    if spread > 1.0:
        return 'erratic'
    if slope > 0.25:
        return 'de_escalating'
    if slope < -0.25:
        return 'escalating'
    if sum(v for v in valences if v < -0.3) / len(valences) < -0.4:
        return 'persistently_negative'
    return 'stable'


# ─── Рекомендация режима ответа ──────────────────────────────────────────────

def _recommend_mode(
    state: str,
    hidden: str,
    trend: str,
    turns_in_state: int,
) -> str:
    # Скрытые состояния имеют приоритет
    if hidden == 'masked_anger':
        return 'name_the_subtext'
    if hidden == 'contempt_behind_humor':
        return 'stay_in_character'   # не реагировать на ловушку
    if hidden == 'rhetorical_exit':
        return 'create_opening'
    if hidden == 'provocative_testing':
        return 'stay_in_character'
    if hidden == 'passive_exit':
        return 'create_opening'
    if hidden == 'slow_boil':
        return 'acknowledge_then_hold'

    if state == 'hostile' and turns_in_state >= 2:
        return 'firm_boundary'
    if state == 'hostile':
        return 'cold_mirror'
    if state == 'passive_agressive':
        return 'name_the_subtext'
    if state == 'contemptuous' and turns_in_state >= 2:
        return 'stay_in_character'   # не меняться под давлением
    if state == 'contemptuous':
        return 'cold_mirror'
    if state == 'disappointed':
        return 'gentle_probe'
    if state == 'frustrated':
        if trend == 'escalating':
            return 'acknowledge_then_hold'
        return 'steady_presence'
    if state == 'withdrawn':
        return 'create_opening'
    if state == 'stressed':
        return 'steady_presence'
    if state == 'testing':
        return 'stay_in_character'
    if state == 'bored':
        return 'raise_stakes'
    if state in ('curious', 'seeking_help', 'warm'):
        return 'natural_response'

    if trend == 'escalating':
        return 'de_escalate'
    return 'natural_response'


# ─── Результат ───────────────────────────────────────────────────────────────

@dataclass
class UserAffectSnapshot:
    current_state: str = 'neutral'
    current_state_ru: str = 'нейтральный'
    current_state_en: str = 'neutral'
    current_intensity: float = 0.3
    current_confidence: float = 0.5

    session_baseline_state: str = 'neutral'
    session_volatility: float = 0.0

    trend: str = 'stable'
    turns_in_current_state: int = 1

    surface_vs_hidden: str = ''
    recommended_response_mode: str = 'natural_response'

    context_note_ru: str = ''
    context_note_en: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'current_state':      self.current_state,
            'current_state_ru':   self.current_state_ru,
            'current_intensity':  round(self.current_intensity, 3),
            'current_confidence': round(self.current_confidence, 3),
            'session_baseline':   self.session_baseline_state,
            'session_volatility': round(self.session_volatility, 3),
            'trend':              self.trend,
            'turns_in_state':     self.turns_in_current_state,
            'hidden':             self.surface_vs_hidden,
            'response_mode':      self.recommended_response_mode,
            'context_note_ru':    self.context_note_ru,
        }


# ─── Главная функция ─────────────────────────────────────────────────────────

def analyze_user_affect(
    matrix_rows: list[Any],
    current_text: str = '',
    current_p_scores: dict[str, float] | None = None,
    current_p_dominant: dict[str, str] | None = None,
    current_ts: float | None = None,
    language: str = 'ru',
) -> UserAffectSnapshot:
    user_rows = [r for r in matrix_rows if getattr(r, 'speaker', '') == 'user']

    if current_p_scores is not None:
        cur_scores   = current_p_scores
        cur_dominant = current_p_dominant or {}
        cur_text     = current_text
        cur_ts       = current_ts or time.time()
    elif user_rows:
        last         = user_rows[-1]
        cur_scores   = last.p_scores
        cur_dominant = last.p_dominant
        cur_text     = last.text
        cur_ts       = last.timestamp
    else:
        return UserAffectSnapshot()

    prev_ts     = user_rows[-2].timestamp if len(user_rows) >= 2 else cur_ts
    gap_seconds = max(0.0, cur_ts - prev_ts)

    # ── Классификация текущего хода ───────────────────────────────────────────
    result = _classify_from_variants(cur_dominant, cur_scores, cur_text)
    if result is None:
        result = _classify_from_scores(cur_scores, cur_dominant)
    state, intensity = result

    hidden = _detect_hidden(state, cur_dominant, cur_scores, cur_text, gap_seconds)

    # ── История ───────────────────────────────────────────────────────────────
    history_states: list[str] = []
    for row in user_rows[:-1]:
        r = _classify_from_variants(row.p_dominant, row.p_scores, row.text)
        if r is None:
            r = _classify_from_scores(row.p_scores, row.p_dominant)
        history_states.append(r[0])

    all_states = history_states + [state]
    trend      = _compute_trend(all_states)

    baseline = Counter(history_states).most_common(1)[0][0] if history_states else state

    turns_in_state = 1
    for s in reversed(history_states):
        if s == state:
            turns_in_state += 1
        else:
            break

    volatility = round(
        sum(1 for s in all_states if s != baseline) / max(len(all_states), 1), 2
    )

    # Уверенность: выше когда несколько ходов подряд + высокая интенсивность
    confidence = min(1.0, 0.35 + 0.12 * turns_in_state + intensity * 0.35)

    mode = _recommend_mode(state, hidden, trend, turns_in_state)

    note_ru, note_en = _build_context_note(
        state=state, hidden=hidden, trend=trend,
        turns_in_state=turns_in_state, baseline=baseline,
        intensity=intensity, mode=mode,
    )

    meta = _STATE_META.get(state, ('неизвестно', 'unknown', 0.0))
    return UserAffectSnapshot(
        current_state=state,
        current_state_ru=meta[0],
        current_state_en=meta[1],
        current_intensity=round(intensity, 3),
        current_confidence=round(confidence, 3),
        session_baseline_state=baseline,
        session_volatility=volatility,
        trend=trend,
        turns_in_current_state=turns_in_state,
        surface_vs_hidden=hidden,
        recommended_response_mode=mode,
        context_note_ru=note_ru,
        context_note_en=note_en,
    )


# ─── Контекстная заметка ─────────────────────────────────────────────────────

_MODE_HINT: dict[str, tuple[str, str]] = {
    'firm_boundary':       ('Установи чёткую границу — не агрессивно, но твёрдо.',
                            'Set a clear boundary — firm, not aggressive.'),
    'cold_mirror':         ('Отрази тон без оправданий и без escalation.',
                            'Mirror the tone without justifying or escalating.'),
    'stay_in_character':   ('Не меняйся. Реагируй как этот персонаж.',
                            "Don't break. React as this persona would."),
    'acknowledge_then_hold':('Признай напряжение — но позицию не меняй.',
                            "Acknowledge the tension — don't concede."),
    'steady_presence':     ('Оставайся стабильным. Не давить, не успокаивать.',
                            'Stay stable. Don\'t push, don\'t over-soothe.'),
    'gentle_probe':        ('Что-то за словами. Осторожно — не форсировать.',
                            'Something is behind the words. Don\'t force it.'),
    'name_the_subtext':    ('Назови то, что чувствуется под словами.',
                            'Name what\'s felt beneath the words.'),
    'create_opening':      ('Дай выход — не гнаться, оставить дверь открытой.',
                            'Leave a door open without chasing.'),
    'raise_stakes':        ('Скучает — добавь неожиданность или вызов.',
                            'They\'re bored — add surprise or a challenge.'),
    'de_escalate':         ('Снизь напряжение — не уступай, не нагнетай.',
                            'Reduce tension — don\'t concede, don\'t aggravate.'),
    'natural_response':    ('', ''),
}

_HIDDEN_LABEL: dict[str, tuple[str, str]] = {
    'masked_anger':         ('Слова нейтральные, злость под поверхностью.',
                             'Words are neutral — anger is felt underneath.'),
    'contempt_behind_humor':('Шутит, но за этим презрение или вызов.',
                             'Joking, but contempt or a challenge is behind it.'),
    'rhetorical_exit':      ('Вопрос не за ответом — это маскировка ухода.',
                             'The question isn\'t for an answer — it\'s a veiled exit.'),
    'provocative_testing':  ('Атакует, но хочет реакции — проверяет персонажа.',
                             'Attacks, but wants a reaction — testing the persona.'),
    'passive_exit':         ('Отстраняется — короткие ответы, контакт падает.',
                             'Withdrawing — short replies, contact is dropping.'),
    'slow_boil':            ('Долго копилось. Сейчас выходит.',
                             'Long buildup. It\'s coming out now.'),
}


def _build_context_note(
    state: str,
    hidden: str,
    trend: str,
    turns_in_state: int,
    baseline: str,
    intensity: float,
    mode: str,
) -> tuple[str, str]:
    meta    = _STATE_META.get(state, ('неизвестно', 'unknown', 0.0))
    i_label = 'слабо' if intensity < 0.4 else ('умеренно' if intensity < 0.7 else 'сильно')
    i_lbl_e = 'mildly' if intensity < 0.4 else ('moderately' if intensity < 0.7 else 'strongly')

    ru: list[str] = [f'Пользователь: {meta[0]} ({i_label})']
    en: list[str] = [f'User: {meta[1]} ({i_lbl_e})']

    if hidden and hidden in _HIDDEN_LABEL:
        h = _HIDDEN_LABEL[hidden]
        ru.append(h[0])
        en.append(h[1])

    _trend_ru = {
        'escalating':           f'Нарастает — {turns_in_state} хода(ов) подряд.',
        'de_escalating':        'Спадает.',
        'erratic':              'Скачет — нестабильно.',
        'persistently_negative':'Устойчивый негатив в сессии.',
    }
    _trend_en = {
        'escalating':           f'Escalating — {turns_in_state} turn(s) in a row.',
        'de_escalating':        'Dropping.',
        'erratic':              'Erratic — unstable.',
        'persistently_negative':'Persistent negative throughout session.',
    }
    if trend in _trend_ru:
        ru.append(_trend_ru[trend])
        en.append(_trend_en[trend])

    if state != baseline and baseline != 'neutral':
        bm = _STATE_META.get(baseline, ('норма', 'baseline', 0.0))
        ru.append(f'Обычно в сессии: {bm[0]}. Сейчас — отклонение.')
        en.append(f'Session baseline: {bm[1]}. This is a departure.')

    hint_ru, hint_en = _MODE_HINT.get(mode, ('', ''))
    if hint_ru:
        ru.append(f'→ {hint_ru}')
        en.append(f'→ {hint_en}')

    return '\n'.join(ru), '\n'.join(en)


# ─── Рендер для промпта ──────────────────────────────────────────────────────

def render_user_affect_block(snapshot: UserAffectSnapshot, language: str = 'ru') -> str:
    """Блок для промпта. Пустой если состояние нейтральное и стабильное."""
    if (snapshot.current_state == 'neutral'
            and not snapshot.surface_vs_hidden
            and snapshot.trend == 'stable'):
        return ''
    note = snapshot.context_note_ru if language != 'en' else snapshot.context_note_en
    if not note:
        return ''
    header = '[СОСТОЯНИЕ ПОЛЬЗОВАТЕЛЯ]' if language != 'en' else '[USER STATE]'
    return f'{header}\n{note}'
