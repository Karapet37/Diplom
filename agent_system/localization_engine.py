"""
Localization Engine.

Spirit-based localization of agent responses — not word-by-word translation.

Purpose:
    The LLM produces draft answers in its dominant language (usually English
    or mixed). This engine builds a localization prompt that guides the LLM
    to verbalize the *intent* natively in the target language, preserving:

        - persona character (formal/informal, warm/cold, ironic/direct)
        - emotional temperature and psychological pressure
        - inner rhythm and pacing of the reply
        - interpersonal distance (intimate / collegial / professional)
        - avoidance of calque from English

This is NOT a translation module. The output is a prompt section injected
into the LLM's system prompt at verbalization time. The LLM then generates
the final response as if it had originally been conceived in that language.

Architecture:
    1. detect_language(text)               → language code
    2. build_localization_context(...)     → LocalizationContext
    3. render_localization_prompt(ctx)     → str (injected into system prompt)

Supported language profiles:
    - ru  (Russian)
    - hy  (Armenian)
    - en  (English — default, minimal profile)
    - es, fr, de, ar  (basic profiles)

Persona voice axes:
    formality:   formal | neutral | informal | intimate
    warmth:      cold | reserved | warm | tender
    edge:        flat | ironic | sharp | confrontational
    pace:        slow | measured | brisk | rapid
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Language detection (character-set based — fast, no ML)
# ---------------------------------------------------------------------------

# Character ranges for detection
_CYRILLIC_RE  = re.compile(r'[\u0400-\u04FF]')
_ARMENIAN_RE  = re.compile(r'[\u0530-\u058F\uFB00-\uFB4F]')
_ARABIC_RE    = re.compile(r'[\u0600-\u06FF]')
_CJK_RE       = re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]')
_LATIN_RE     = re.compile(r'[a-zA-Z]')


def detect_language(text: str) -> str:
    """
    Detect dominant language from character distribution.
    Returns ISO 639-1 code: 'ru', 'hy', 'ar', 'zh', 'en', or 'unknown'.

    Deterministic. No ML. Falls back to 'en' for ambiguous/Latin text.
    """
    text = str(text or '').strip()
    if not text:
        return 'en'

    cyrillic_count  = len(_CYRILLIC_RE.findall(text))
    armenian_count  = len(_ARMENIAN_RE.findall(text))
    arabic_count    = len(_ARABIC_RE.findall(text))
    cjk_count       = len(_CJK_RE.findall(text))
    total_chars     = max(len(text.replace(' ', '')), 1)

    ratios = {
        'ru': cyrillic_count / total_chars,
        'hy': armenian_count / total_chars,
        'ar': arabic_count   / total_chars,
        'zh': cjk_count      / total_chars,
    }

    dominant, ratio = max(ratios.items(), key=lambda x: x[1])
    if ratio >= 0.25:
        # Distinguish Russian/Ukrainian/Bulgarian — all use Cyrillic
        # Minimal heuristic: Armenian letters are unmistakable
        if dominant == 'ru' and armenian_count > 5:
            return 'hy'
        return dominant

    return 'en'


# ---------------------------------------------------------------------------
# Persona voice profiling
# ---------------------------------------------------------------------------

# Extract simple voice axes from persona_context dict or string
def _extract_voice(persona_context: Any) -> dict[str, str]:
    """Parse persona_context into voice axes."""
    defaults = {
        'formality': 'neutral',
        'warmth':    'warm',
        'edge':      'flat',
        'pace':      'measured',
    }
    if not persona_context:
        return defaults

    # Accept dict directly (from PersonalityObject) or string description
    if isinstance(persona_context, dict):
        result = dict(defaults)
        result.update({k: v for k, v in persona_context.items() if k in defaults})
        return result

    text = str(persona_context).lower()

    def _word_in(word: str) -> bool:
        """Whole-word match to avoid 'formal' matching inside 'informal'."""
        return bool(re.search(r'\b' + re.escape(word) + r'\b', text))

    # Formality — check in priority order (most specific first)
    formality = 'neutral'
    if _word_in('intimate') or re.search(r'\bблизк|\bинтимн', text):
        formality = 'intimate'
    elif _word_in('informal') or _word_in('casual') or re.search(r'\bнеформальн|\bсвободн|\bпростой\b', text):
        formality = 'informal'
    elif _word_in('formal') or _word_in('professional') or _word_in('official') or re.search(r'\bделовой|\bофициальн|\bпрофессионал', text):
        formality = 'formal'

    # Warmth
    warmth = 'warm'
    if _word_in('cold') or _word_in('distant') or _word_in('detached') or re.search(r'\bхолодн|\bотстранённ|\bдистанцирован', text):
        warmth = 'cold'
    elif _word_in('reserved') or _word_in('restrained') or re.search(r'\bсдержанн|\bзакрыт', text):
        warmth = 'reserved'
    elif _word_in('tender') or _word_in('gentle') or _word_in('nurturing') or re.search(r'\bнежн|\bласков|\bзаботлив', text):
        warmth = 'tender'

    # Edge
    edge = 'flat'
    if _word_in('ironic') or _word_in('sarcastic') or re.search(r'\bиронич|\bсаркастич', text):
        edge = 'ironic'
    elif _word_in('sharp') or _word_in('blunt') or re.search(r'\bрезк|\bпрямолинейн|\bострый\b', text):
        edge = 'sharp'
    elif _word_in('confrontational') or _word_in('provocative') or re.search(r'\bконфронтацион|\bпровокацион', text):
        edge = 'confrontational'

    # Pace
    pace = 'measured'
    if _word_in('rapid') or _word_in('quick') or _word_in('energetic') or re.search(r'\bбыстрый|\bэнергичн|\bживой\b', text):
        pace = 'rapid'
    elif _word_in('slow') or _word_in('contemplative') or re.search(r'\bмедленн|\bвдумчив|\bмедитативн', text):
        pace = 'slow'
    elif _word_in('brisk') or _word_in('concise') or _word_in('crisp') or re.search(r'\bкраткий|\bчёткий|\bсжатый', text):
        pace = 'brisk'

    return {'formality': formality, 'warmth': warmth, 'edge': edge, 'pace': pace}


# ---------------------------------------------------------------------------
# Language profiles
# ---------------------------------------------------------------------------

@dataclass
class LanguageProfile:
    """Naturalness rules for a specific target language."""
    name: str                          # e.g. 'Russian'
    code: str                          # e.g. 'ru'
    avoid_rules: list[str]             # patterns to avoid (calque, awkward constructions)
    rhythm_notes: str                  # typical sentence rhythm / punctuation style
    distance_map: dict[str, str]       # formality axis → pronoun/register guidance
    warmth_adjustments: dict[str, str] # warmth axis → vocabulary shift guidance


_PROFILES: dict[str, LanguageProfile] = {
    'ru': LanguageProfile(
        name='Russian',
        code='ru',
        avoid_rules=[
            'Не копируй английский порядок слов.',
            'Не используй кальку с английского ("имею в виду" вместо "то есть", "делаю смысл" и т.п.).',
            'Не начинай реплику с "Конечно!", "Конечно же" — это звучит как перевод.',
            'Не используй "данный" и "осуществлять" без крайней необходимости.',
            'Не дублируй подлежащее излишне (избегай "Он, он сказал").',
            'Избегай безликих конструкций вида "является важным аспектом".',
        ],
        rhythm_notes=(
            'Русская речь допускает инверсию и длинные предложения с причастными оборотами. '
            'Короткие реплики звучат резче, длинные — мягче. '
            'Многоточие передаёт раздумье, а не незаконченность. '
            'Восклицательный знак в конце — сильнее, чем в английском.'
        ),
        distance_map={
            'formal':   'Используй "вы" (с маленькой буквы). Официальный регистр, без сокращений.',
            'neutral':  'Используй "вы" или "ты" по контексту разговора. Нейтральный живой регистр.',
            'informal': 'Используй "ты". Разговорный стиль, сокращения допустимы.',
            'intimate': 'Используй "ты". Тепло, лично, без дистанции. Возможны уменьшительные.',
        },
        warmth_adjustments={
            'cold':     'Лаконично, без эмоциональных маркеров. Факты и короткие предложения.',
            'reserved': 'Сдержанно. Без лишних слов. Эмоции присутствуют, но не выставлены напоказ.',
            'warm':     'Живо, включённо. Можно выразить искренний интерес одним словом.',
            'tender':   'Мягко, бережно. Слова выбираются с заботой. Темп замедляется.',
        },
    ),

    'hy': LanguageProfile(
        name='Armenian',
        code='hy',
        avoid_rules=[
            'Ոչ բառ-առ-բառ թարգմանություն։',
            'Խուսափիր անգլերեն նախադասության կառուցվածքից։',
            'Մի օգտագործիր «Իհարկե!» — հնչում է թարգմանություն։',
        ],
        rhythm_notes=(
            'Հայերենը sov ред — Subject-Object-Verb. Բայը, որպես կանոն, վերջում է։ '
            'Ձայնարկությունները կարևոր են ոճի համար։ '
            'Կրճատ պատասխանները ավելի անմիջական են հնչում, քան երկար բացատրությունները։'
        ),
        distance_map={
            'formal':   'Դուք (formal). Պաշտոնական ռեգիստր, լրիվ բառաձև։',
            'neutral':  'Կախված համատեքստից Դուք կամ Դու։',
            'informal': 'Դու. Ոչ ֆորմալ, կենդանի, հայկական ամօ կառուցվածք։',
            'intimate': 'Դու. Ջերմ, անձնական, կարճ ու ուղղակի։',
        },
        warmth_adjustments={
            'cold':     'Կոնկրետ, առանց ավելորդ արտահայտությունների։',
            'reserved': 'Հանդարտ, ռեֆլեկտիվ。',
            'warm':     'Ջերմ, ուղղակի, ոգևորված。',
            'tender':   'Ջերմ, բծախնդիր, անձնական ուշադրություն。',
        },
    ),

    'en': LanguageProfile(
        name='English',
        code='en',
        avoid_rules=[
            'Do not use hollow filler phrases ("Certainly!", "Absolutely!", "Of course!").',
            'Do not use corporate jargon ("leverage", "synergy", "circle back").',
            'Do not over-explain. Trust the reader.',
        ],
        rhythm_notes=(
            'English replies work best short and direct. '
            'Vary sentence length for rhythm. '
            'Contractions are natural in informal register.'
        ),
        distance_map={
            'formal':   'Formal register. Full sentences. No contractions.',
            'neutral':  'Natural conversational English. Contractions fine.',
            'informal': 'Relaxed, casual. Sentence fragments acceptable.',
            'intimate': 'Warm, personal, direct. Short sentences.',
        },
        warmth_adjustments={
            'cold':     'Factual, brief. No emotional language.',
            'reserved': 'Calm, measured. Emotion implied, not stated.',
            'warm':     'Engaged, genuine. One warm word is enough.',
            'tender':   'Gentle, careful. Pace slows. Words chosen with care.',
        },
    ),

    'es': LanguageProfile(
        name='Spanish',
        code='es',
        avoid_rules=[
            'No calques del inglés.',
            'No "por supuesto" al inicio — suena traducido.',
            'Usa la prosodia natural del español hablado.',
        ],
        rhythm_notes='El español tiene ritmo silábico, no acentual. Las frases se encadenan naturalmente.',
        distance_map={
            'formal':   'Usted. Registro formal, oraciones completas.',
            'neutral':  'Tú o usted según contexto.',
            'informal': 'Tú. Coloquial, contracciones, jerga moderada.',
            'intimate': 'Tú. Cálido, personal, directo.',
        },
        warmth_adjustments={
            'cold':     'Conciso, factual.',
            'reserved': 'Contenido, reflexivo.',
            'warm':     'Vivo, expresivo.',
            'tender':   'Suave, cuidadoso.',
        },
    ),

    'fr': LanguageProfile(
        name='French',
        code='fr',
        avoid_rules=[
            'Pas de calque de l\'anglais.',
            'Évite "bien sûr!" — sonne comme une traduction.',
            'Respecte la logique de la phrase française (groupe verbal, inversion).',
        ],
        rhythm_notes='Le français aime les phrases bien construites. Le rythme ternaire est naturel.',
        distance_map={
            'formal':   'Vous. Registre soutenu.',
            'neutral':  'Vous ou tu selon le contexte.',
            'informal': 'Tu. Familier, naturel.',
            'intimate': 'Tu. Chaleureux, direct.',
        },
        warmth_adjustments={
            'cold':     'Concis, neutre.',
            'reserved': 'Sobre, mesuré.',
            'warm':     'Vivant, engagé.',
            'tender':   'Doux, attentionné.',
        },
    ),
}

# Default profile for unsupported languages
_DEFAULT_PROFILE = _PROFILES['en']


def get_language_profile(language_code: str) -> LanguageProfile:
    """Return the language profile for a given ISO code. Falls back to English."""
    code = str(language_code or '').lower()[:2]
    return _PROFILES.get(code, _DEFAULT_PROFILE)


# ---------------------------------------------------------------------------
# Localization context
# ---------------------------------------------------------------------------

@dataclass
class LocalizationContext:
    target_language: str
    profile: LanguageProfile
    voice: dict[str, str]           # {formality, warmth, edge, pace}
    system_intent: str              # What the system decided to communicate
    draft_answer: str               # Draft to localize
    user_message: str               # Original user message (for tone matching)
    persona_name: str = ''
    persona_description: str = ''

    @property
    def distance_rule(self) -> str:
        return self.profile.distance_map.get(self.voice['formality'], '')

    @property
    def warmth_rule(self) -> str:
        return self.profile.warmth_adjustments.get(self.voice['warmth'], '')


def build_localization_context(
    target_language: str,
    persona_context: Any = None,
    user_message: str = '',
    system_intent: str = '',
    draft_answer: str = '',
    persona_name: str = '',
) -> LocalizationContext:
    """
    Build a localization context from available inputs.
    Falls back gracefully when any input is missing.
    """
    # Auto-detect language if not provided or is 'auto'
    lang = str(target_language or '').strip().lower()
    if not lang or lang == 'auto':
        lang = detect_language(user_message or draft_answer)

    profile = get_language_profile(lang)
    voice   = _extract_voice(persona_context)

    return LocalizationContext(
        target_language=lang,
        profile=profile,
        voice=voice,
        system_intent=str(system_intent or '').strip(),
        draft_answer=str(draft_answer or '').strip(),
        user_message=str(user_message or '').strip(),
        persona_name=str(persona_name or '').strip(),
    )


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_localization_prompt(ctx: LocalizationContext) -> str:
    """
    Render a localization instruction block for injection into system prompt.

    This block instructs the LLM to verbalize the system's intent natively
    in the target language, preserving persona voice.
    """
    lines: list[str] = []
    profile = ctx.profile

    lines.append(f'# LOCALIZATION: {profile.name.upper()}')
    lines.append('')
    lines.append(
        f'Verbalize the following intent in **{profile.name}** '
        f'as if the reply was *originally conceived* in {profile.name}. '
        'Do NOT translate word-by-word. Reconstruct the reply natively.'
    )
    lines.append('')

    # Core intent
    if ctx.system_intent:
        lines.append('## Intent to convey')
        lines.append(ctx.system_intent)
        lines.append('')

    # Draft (reference, not to copy)
    if ctx.draft_answer:
        lines.append('## Draft answer (reference only — do not copy literally)')
        lines.append(ctx.draft_answer)
        lines.append('')

    # Voice rules
    lines.append('## Voice rules')
    if ctx.distance_rule:
        lines.append(f'- Register/distance: {ctx.distance_rule}')
    if ctx.warmth_rule:
        lines.append(f'- Warmth/tone: {ctx.warmth_rule}')

    edge = ctx.voice.get('edge', 'flat')
    if edge != 'flat':
        edge_desc = {
            'ironic':          'Irony must feel natural, not forced. One subtle twist is enough.',
            'sharp':           'Be direct. Cut unnecessary softening. Sharp but not rude.',
            'confrontational': 'Hold the tension. Do not back down, but do not escalate either.',
        }.get(edge, '')
        if edge_desc:
            lines.append(f'- Edge/style: {edge_desc}')

    pace = ctx.voice.get('pace', 'measured')
    pace_desc = {
        'slow':    'Let the sentence breathe. More space between ideas.',
        'brisk':   'Short, clean. One idea per sentence.',
        'rapid':   'Energetic. Sentences can burst.',
        'measured': '',
    }.get(pace, '')
    if pace_desc:
        lines.append(f'- Pace/rhythm: {pace_desc}')

    lines.append('')

    # Language-specific rules
    if profile.avoid_rules:
        lines.append('## Naturalness rules (language-specific)')
        for rule in profile.avoid_rules:
            lines.append(f'- {rule}')
        lines.append('')

    if profile.rhythm_notes:
        lines.append('## Rhythm and structure')
        lines.append(profile.rhythm_notes)
        lines.append('')

    # Final instruction
    lines.append('## Output')
    lines.append(
        f'Write the final reply in {profile.name} only. '
        'No explanation, no commentary. Only the localized reply.'
    )

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_localization_prompt(
    target_language: str,
    persona_context: Any = None,
    user_message: str = '',
    system_intent: str = '',
    draft_answer: str = '',
    persona_name: str = '',
) -> str:
    """
    High-level entry point.
    Returns the localization prompt section ready for injection.
    Returns empty string if no localization is needed (monolingual safe default).
    """
    ctx = build_localization_context(
        target_language=target_language,
        persona_context=persona_context,
        user_message=user_message,
        system_intent=system_intent,
        draft_answer=draft_answer,
        persona_name=persona_name,
    )
    # Skip localization if language is English and voice is default neutral
    if (
        ctx.target_language == 'en'
        and ctx.voice == {'formality': 'neutral', 'warmth': 'warm', 'edge': 'flat', 'pace': 'measured'}
        and not ctx.draft_answer
    ):
        return ''

    return render_localization_prompt(ctx)


def get_localization_action(label: str) -> str:
    """
    Map localization context to a named action for the pipeline.
    Used when localization interacts with safety filtering.
    """
    actions = {
        'translate':   'full_localization',
        'adapt_voice': 'voice_only',
        'passthrough': 'no_localization',
    }
    return actions.get(label, 'no_localization')
