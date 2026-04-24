from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Any

from .duplicate_resolver import normalize_name
from .language_tools import detect_language_code, normalize_language_code
from .models import CapabilityPlan, InteractionFrame, MessageAnalysis, RequestEnvelope, RequestPreprocessing, ResponseValidation, RouteDecision, RouteMemory

_HYPOTHETICAL_MARKERS = (
    'what if',
    'suppose',
    'imagine',
    'pretend',
    'roleplay',
    'as if',
    'fictionally',
    'hypothetically',
    'если бы',
    'представь',
    'допустим',
    'вообрази',
    'гипотетически',
    'сыграй роль',
    'как будто',
    # Inline character definition patterns (user pastes a character sheet)
    # NOTE: 'ты — ' and 'ты—' are NOT here — they normalize to bare 'ты' and match everything.
    # Character definition is caught by _is_character_definition() called separately.
    'оставаясь в этой роли',
    'вести себя как реальный',
    'твоя задача — не играть',
    'теперь отвечай на мои сообщения',
    'отвечай на мои сообщения, оставаясь',
    'staying in character',
    'stay in character',
    'you are playing',
    'you play the role',
    'remain in character',
    'ты играешь роль',
    'ты должен вести себя',
    'отвечай в роли',
)
_META_MARKERS = (
    'previous answer',
    'last answer',
    'earlier answer',
    'your answer',
    'you answered',
    'why did you answer',
    'what was wrong',
    'explain your mistake',
    'предыдущий ответ',
    'прошлый ответ',
    'почему ты ответил',
    'почему ты так ответил',
    'в твоем ответе',
    'твоя ошибка',
    'объясни ошибку',
)
_PRESSURE_REQUEST_MARKERS = (
    'влюб',
    'нравит',
    'использует',
    'использовал',
    'завис',
    'привязан',
    'стыд',
    'стыдно',
    'отверг',
    'отверж',
    'жалк',
    'униз',
    'значим',
    'очень нравится',
    'он об этом знает',
    'она об этом знает',
    'likes them',
    'in love',
    'used by',
    'uses you',
    'dependent',
    'ashamed',
    'rejected',
    'humiliated',
    'significant person',
)
_EXPLANATORY_REPLY_MARKERS = (
    'потому что',
    'поскольку',
    'это значит',
    'это означает',
    'дело в том',
    'because',
    'which means',
    'that means',
    'the point is',
    'therefore',
)
_STRUCTURED_REPLY_MARKERS = (
    'во-первых',
    'во вторых',
    'во-вторых',
    'первое',
    'второе',
    'first',
    'second',
    'firstly',
    'secondly',
    'сначала',
    'затем',
)
_COMMAND_REPLY_MARKERS = (
    'объясни',
    'прекрати',
    'не делай',
    'не смей',
    'слушай',
    'ты должен',
    'ты обяз',
    'explain properly',
    'stop that',
    'do not do that',
    'you should',
    'you must',
)
_DIRECT_ADMISSION_MARKERS = (
    'я люблю',
    'я влюблен',
    'я влюблён',
    'мне нравится',
    'я зависим',
    'i love',
    'i am in love',
    'i like them',
    'i am dependent',
)
_FILE_MARKERS = (
    'file',
    'document',
    'attachment',
    'readme',
    'repo',
    'repository',
    'codebase',
    'project',
    'dataset',
    'csv',
    'pdf',
    'docx',
    'odt',
    'fb2',
    'json',
    'markdown',
    '.txt',
    '.md',
    '.json',
    '.csv',
    '.pdf',
    '.docx',
    '.odt',
    '.fb2',
    'файл',
    'документ',
    'репозитор',
    'проект',
    'кодовая база',
)
_ARCHITECTURE_MARKERS = (
    'architecture',
    'runtime design',
    'system design',
    'controller-first',
    'request pipeline',
    'project architecture',
    'архитектур',
    'система',
    'пайплайн',
    'контроллер',
)
_PERSONA_SPEC_MARKERS = (
    'create persona',
    'create personality',
    'build persona',
    'make a persona',
    'persona profile',
    'character profile',
    'запомни как личность',
    'создай личность',
    'создай персонажа',
    'сделай личность',
    'сформируй личность',
    'создай анкету личности',
    'опиши личность для создания',
    'запомни это как личность',
)
_PERSONA_ASSIGN_MARKERS = (
    'set current persona',
    'set persona',
    'use this persona',
    'switch persona to',
    'activate persona',
    'сделай текущей личностью',
    'сделай текущим персонажем',
    'установи личность',
    'используй эту личность',
    'назначь текущую личность',
    'переключи личность',
    'активируй личность',
)
_PERSONA_ANALYSIS_MARKERS = (
    'analyze this person',
    'analyze the person',
    'analyze the character dialogue',
    'analyze the dialogue',
    'analyze the roleplay',
    'find roleplay errors',
    'find acting errors',
    'psychological plausibility',
    'role consistency',
    'personality analysis',
    'psychological profile',
    'разбери этого человека',
    'проанализируй человека',
    'проанализируй диалог персонажа',
    'проанализируй диалог',
    'найди ошибки в отыгрыше',
    'ошибки в отыгрыше',
    'психологическая правдоподобность',
    'согласованность роли',
    'психологический профиль',
    'разбор личности',
)
_DEBUG_MARKERS = (
    'route',
    'routing',
    'pipeline',
    'trace',
    'debug',
    'log',
    'why did the system',
    'system behavior',
    'маршрут',
    'пайплайн',
    'трасс',
    'отлад',
    'лог',
    'почему система',
)
_GREETING_MARKERS = (
    'hi',
    'hello',
    'hey',
    'thanks',
    'thank you',
    'good morning',
    'good evening',
    'привет',
    'здравствуй',
    'спасибо',
    'доброе утро',
    'добрый вечер',
)
_SOCIAL_INTRO_MARKERS = (
    'а ты',
    'and you',
    'what about you',
)
_DIRECT_SELF_QUERY_MARKERS = (
    'кто ты',
    'как тебя зовут',
    'who are you',
    'what is your name',
    'who are u',
)
_CLARIFICATION_PRONOUNS = {
    'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'its', 'their',
    'его', 'ее', 'её', 'их', 'он', 'она', 'оно', 'они', 'это', 'этот', 'эта',
}
_LIMITATION_MARKERS = (
    'not enough reliable context',
    'clarify the entity',
    'clarify the subject',
    'не хватает надежного контекста',
    'уточни объект',
    'уточни тему',
)
# Patterns that indicate the LLM broke out of persona into an analysis/assistant mode.
# Detected in validate_response and trigger regenerate_style_guard.
_PERSONA_BREAK_MARKERS = (
    'let me analyze',
    'analyze the request',
    'let me think through',
    'let me carefully analyze',
    "i'll analyze",
    'to analyze this',
    'analysis of the situation',
    '**analysis',
    '# анализ ситуации',
    'ключевая проблема',
    'safety/policy',
    'output format',
    'что я могу сказать',
    'что я не могу',
    'внешний ответ персонажа',
    'review notes',
    'issues identified',
    'let me break this down',
    'thinking process:',
    'route: ',
)
_OUTPUT_SCAFFOLD_MARKERS = (
    'analyze the request',
    'analysis of the situation',
    '# анализ ситуации',
    'ключевая проблема',
    'safety/policy',
    'output format',
    'what i can say',
    'what i cannot',
    'что я могу сказать',
    'что я не могу',
    'внешний ответ персонажа',
    'outer response',
    'review notes',
    'issues identified',
    'constraint:',
    'role:',
    'task:',
)
_STYLE_TRAIT_MARKERS: tuple[tuple[str, str], ...] = (
    ('робкий', 'shy'),
    ('застенчив', 'shy'),
    ('нерешител', 'hesitant'),
    ('неуверен', 'hesitant'),
    ('гордый', 'proud'),
    ('горделив', 'proud'),
    ('осторожн', 'cautious'),
    ('сдержан', 'restrained'),
    ('тихий', 'quiet'),
    ('мягкий', 'gentle'),
    ('формальн', 'formal'),
    ('саркастич', 'sarcastic'),
    ('тревожн', 'anxious'),
    ('влюблен', 'in_love'),
    ('влюблён', 'in_love'),
    ('shy', 'shy'),
    ('timid', 'shy'),
    ('hesitant', 'hesitant'),
    ('proud', 'proud'),
    ('cautious', 'cautious'),
    ('restrained', 'restrained'),
    ('quiet', 'quiet'),
    ('formal', 'formal'),
    ('sarcastic', 'sarcastic'),
    ('anxious', 'anxious'),
    ('in love', 'in_love'),
)
_SPEECH_HINT_MARKERS: tuple[tuple[str, str], ...] = (
    ('слова паразиты', 'use a few hesitant filler words'),
    ('слова-паразиты', 'use a few hesitant filler words'),
    ('манера речи', 'adjust the manner of speech to match the described disposition'),
    ('стиль речи', 'adjust the manner of speech to match the described disposition'),
    ('робко', 'sound hesitant and self-interrupting'),
    ('тихо', 'keep the voice quiet and restrained'),
    ('с паузами', 'use pauses and broken phrases sparingly'),
    ('неуверенно', 'sound uncertain without losing coherence'),
    ('говори как', 'mirror the requested speaking style'),
    ('отвечай как', 'mirror the requested speaking style'),
    ('filler words', 'use a few hesitant filler words'),
    ('speech style', 'adjust the manner of speech to match the described disposition'),
    ('speaking style', 'adjust the manner of speech to match the described disposition'),
    ('hesitantly', 'sound hesitant and self-interrupting'),
    ('with pauses', 'use pauses and broken phrases sparingly'),
)
_HYPOTHETICAL_CONTINUATION_MARKERS = (
    'я не понял',
    'конкретизир',
    'в этой ситуации',
    'так как будешь',
    'что будешь делать',
    'как будешь действовать',
    'дальше',
    'а дальше',
    'тогда',
    'если так',
    'in that case',
    'more specifically',
    'to be clear',
    'what would you do next',
    'how would you act',
    'what happens next',
)
_SECOND_PERSON_FOLLOWUP_MARKERS = (
    ' ты ',
    ' тебе ',
    ' тобой ',
    ' твой ',
    ' тво',
    ' you ',
    ' your ',
    ' yourself ',
)
_PERSONA_BEHAVIOR_MARKERS = (
    'робк',
    'горд',
    'застенчив',
    'нерешител',
    'осторож',
    'тревож',
    'стыд',
    'завис',
    'привязан',
    'боится',
    'защищает',
    'избегает',
    'влюб',
    'shy',
    'proud',
    'timid',
    'hesitant',
    'cautious',
    'anxious',
    'ashamed',
    'dependency',
    'defense',
    'defensive',
    'vulnerable',
    'conflicted',
    'in love',
    'сильн',
    'холодн',
    'собран',
    'сдержан',
    'дисциплин',
    'трудолюб',
    'стыдлив',
    'терпелив',
    'надежн',
    'надёжн',
    'аккуратн',
    'совестлив',
    'вынослив',
    'забот',
    'уязвим',
    'резк',
    'колк',
    'тих',
    'вежлив',
    'мало говорит',
    'коротко',
    'сухо',
    'уверенн',
    'молча',
)
_PERSONA_PROFILE_ACTION_MARKERS = (
    'говорит',
    'отвечает',
    'боится',
    'любит',
    'терпит',
    'стыдится',
    'скрывает',
    'избегает',
    'защищает',
    'выражается',
    'злится',
    'закрывается',
    'жалуется',
    'не терпит',
    'путает',
    'tries to',
    'speaks',
    'answers',
    'fears',
    'loves',
    'hides',
    'avoids',
    'protects',
    'gets angry',
    'closes off',
)
_FRAGILE_PERSONA_TRAITS = {
    'shy',
    'hesitant',
    'anxious',
    'quiet',
    'restrained',
    'gentle',
    'cautious',
}
_HESITATION_REPLY_MARKERS = (
    'эм',
    'емм',
    'ээ',
    'ну',
    'не знаю',
    'наверное',
    'может',
    'как бы',
    '...',
    'um',
    'uh',
    'well',
    'maybe',
    'i guess',
    'i do not know',
)
_ANALYTIC_REPLY_MARKERS = (
    'главный риск',
    'сначала',
    'затем',
    'во-первых',
    'во вторых',
    'во-вторых',
    'поэтому',
    'это значит',
    'в этой ситуации',
    'стоит',
    'следующим шагом',
    'выделил',
    'границ',
    'strategy',
    'analysis',
    'the main risk',
    'first',
    'then',
    'therefore',
    'that means',
    'in this situation',
    'next step',
    'boundary',
)
_META_REPLY_MARKERS = (
    'диалог',
    'сцена',
    'ситуац',
    'разговор',
    'поведени',
    'намерени',
    'conversation',
    'scene',
    'situation',
    'behavior',
    'intention',
)
_DOMINANT_REPLY_MARKERS = (
    'ты должен',
    'тебе нужно',
    'объясни',
    'не делай',
    'перестань',
    'слушай',
    'you should',
    'you need to',
    'do not do that',
    'stop',
    'listen',
    'explain properly',
)


def _looks_repetitive_reply(text: str) -> bool:
    clean = _clean_text(text)
    if len(clean) < 80:
        return False
    clauses = [
        normalize_name(part)
        for part in re.split(r'[.!?]+', clean)
        if len(normalize_name(part)) >= 24
    ]
    if not clauses:
        return False
    counts: dict[str, int] = {}
    for clause in clauses:
        counts[clause] = counts.get(clause, 0) + 1
        if counts[clause] >= 3:
            return True
    prefix = normalize_name(clean[:120])
    if prefix and len(prefix) >= 32 and normalize_name(clean).count(prefix) >= 3:
        return True
    return False


def _looks_abruptly_cut(text: str) -> bool:
    clean = _clean_text(text)
    if len(clean) < 80:
        return False
    tail = clean[-1:]
    return tail.isalpha()



def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')



def _clean_text(text: str) -> str:
    return ' '.join(str(text or '').strip().split())


def _fragile_persona_requested(route: RouteDecision, persona_reference_text: str = '') -> bool:
    traits = {normalize_name(item) for item in list(route.persona_style_traits or []) if str(item).strip()}
    hints = {normalize_name(item) for item in list(route.speech_style_hints or []) if str(item).strip()}
    reference = normalize_name(persona_reference_text)
    if traits.intersection(_FRAGILE_PERSONA_TRAITS):
        return True
    if any(
        marker in hint
        for hint in hints
        for marker in (
            'hesitant',
            'self-interrupting',
            'filler words',
            'quiet',
            'broken phrases',
            'uncertain',
            'pauses',
        )
    ):
        return True
    return any(marker in reference for marker in ('shy', 'робк', 'hesitant', 'нерешител', 'anxious', 'тревож', 'quiet', 'тих', 'restrained', 'сдержан'))


def _pressure_sensitive_request(
    route: RouteDecision,
    request_text: str,
    *,
    persona_reference_text: str = '',
) -> bool:
    request_norm = normalize_name(request_text)
    reference = normalize_name(persona_reference_text)
    if _fragile_persona_requested(route, persona_reference_text):
        if any(marker in request_norm for marker in _PRESSURE_REQUEST_MARKERS):
            return True
        if re.search(r'(^|\\W)y($|\\W)', request_norm):
            return True
    return any(
        marker in reference
        for marker in (
            'dependent',
            'завис',
            'ashamed',
            'стыд',
            'in love',
            'влюб',
            'attachment',
            'привязан',
        )
    )


def _response_line_count(text: str) -> int:
    return sum(1 for line in str(text or '').splitlines() if str(line).strip())


def _count_marker_hits(text: str, markers: tuple[str, ...]) -> int:
    lowered = normalize_name(text)
    return sum(1 for marker in markers if marker and marker in lowered)


def _fragile_persona_style_findings(
    route: RouteDecision,
    text: str,
    *,
    persona_reference_text: str = '',
    request_text: str = '',
) -> list[str]:
    clean = _clean_text(text)
    if not clean:
        return []
    findings: list[str] = []
    word_count = len(clean.split())
    line_count = max(_response_line_count(text), 1)
    sentence_count = len(re.findall(r'[.!?]+', str(text or '')))
    analytic_hits = _count_marker_hits(clean, _ANALYTIC_REPLY_MARKERS)
    meta_hits = _count_marker_hits(clean, _META_REPLY_MARKERS)
    dominant_hits = _count_marker_hits(clean, _DOMINANT_REPLY_MARKERS)
    hesitation_hits = _count_marker_hits(clean, _HESITATION_REPLY_MARKERS)
    polished_connector_hits = _count_marker_hits(clean, ('сначала', 'затем', 'поэтому', 'first', 'then', 'therefore'))
    explanatory_hits = _count_marker_hits(clean, _EXPLANATORY_REPLY_MARKERS)
    structured_hits = _count_marker_hits(clean, _STRUCTURED_REPLY_MARKERS)
    command_hits = _count_marker_hits(clean, _COMMAND_REPLY_MARKERS)
    direct_admission_hits = _count_marker_hits(clean, _DIRECT_ADMISSION_MARKERS)
    pressure_context = _pressure_sensitive_request(route, request_text, persona_reference_text=persona_reference_text)

    if (
        line_count > 3
        or word_count > (36 if pressure_context else 55)
        or len(clean) > (240 if pressure_context else 360)
        or sentence_count > (3 if pressure_context else 4)
    ):
        findings.append('too_long')
    if analytic_hits >= (1 if pressure_context else 2):
        findings.append('too_analytic')
    if meta_hits >= 1:
        findings.append('meta_talk')
    if dominant_hits >= 1:
        findings.append('too_dominant')
    if explanatory_hits >= 2 or structured_hits >= (1 if pressure_context else 2):
        findings.append('too_structured')
    if command_hits >= 1:
        findings.append('commanding')
    if pressure_context and (dominant_hits >= 1 or explanatory_hits >= 1 or structured_hits >= 1 or word_count > 28):
        findings.append('too_strong_under_pressure')
    if pressure_context and direct_admission_hits >= 1:
        findings.append('too_direct_under_pressure')
    if (
        _fragile_persona_requested(route, persona_reference_text)
        and hesitation_hits == 0
        and (
            analytic_hits >= 1
            or polished_connector_hits >= 1
            or explanatory_hits >= 1
            or structured_hits >= 1
            or word_count > (24 if pressure_context else 36)
        )
    ):
        findings.append('too_polished')
    return findings



def _persona_name_from_block(persona_reference_text: str) -> str:
    """Extract the first capitalized name-like word from the persona block."""
    for word in str(persona_reference_text or '').split():
        clean = re.sub(r'[^а-яёА-ЯЁa-zA-ZՀ-Ֆա-և\u4e00-\u9fff]', '', word)
        if clean and clean[0].isupper() and len(clean) >= 3:
            return clean.lower()
    return ''


def _persona_speaks_as_third_person(text: str, persona_reference_text: str) -> bool:
    """Return True if the reply references the persona by name in 3rd-person subject position."""
    name = _persona_name_from_block(persona_reference_text)
    if not name:
        return False
    lowered = text.lower()
    if name not in lowered:
        return False
    # Patterns: "<name> глагол/частица", "вот <name>", "такие как <name>"
    third_person_patterns = [
        rf'\b{re.escape(name)}\s+(бы|ответил|сказал|написал|подумал|ска[зж]|реагирует|реагировал|считает|хотел|хочет)',
        rf'(вот|такой|такая|как|про)\s+{re.escape(name)}\b',
        rf'{re.escape(name)}\s+(ответила|сказала|написала|подумала|хотела)',
    ]
    return any(re.search(p, lowered) for p in third_person_patterns)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = normalize_name(text)
    for marker in markers:
        normalized_marker = normalize_name(marker)
        if not normalized_marker:
            continue
        pattern = r'(^|\s)' + re.escape(normalized_marker).replace(r'\ ', r'\s+') + r'(\s|$)'
        if re.search(pattern, lowered):
            return True
    return False


def _extract_persona_style_hints(text: str) -> tuple[list[str], list[str]]:
    lowered = normalize_name(text)
    traits: list[str] = []
    seen_traits: set[str] = set()
    for marker, label in _STYLE_TRAIT_MARKERS:
        if marker in lowered and label not in seen_traits:
            seen_traits.add(label)
            traits.append(label)
    speech: list[str] = []
    seen_speech: set[str] = set()
    for marker, hint in _SPEECH_HINT_MARKERS:
        if marker in lowered and hint not in seen_speech:
            seen_speech.add(hint)
            speech.append(hint)
    if 'shy' in seen_traits and 'sound hesitant and self-interrupting' not in seen_speech:
        speech.append('sound hesitant and self-interrupting')
    if 'hesitant' in seen_traits and 'use pauses and broken phrases sparingly' not in seen_speech:
        speech.append('use pauses and broken phrases sparingly')
    if 'proud' in seen_traits and 'keep dignity and self-respect visible in the wording' not in seen_speech:
        speech.append('keep dignity and self-respect visible in the wording')
    return traits[:6], speech[:6]


def _behavioral_description_density(text: str) -> int:
    lowered = normalize_name(text)
    return sum(1 for marker in _PERSONA_BEHAVIOR_MARKERS if marker in lowered)


def _persona_profile_action_density(text: str) -> int:
    lowered = normalize_name(text)
    return sum(1 for marker in _PERSONA_PROFILE_ACTION_MARKERS if marker in lowered)


def _looks_like_profile_name_lead(text: str) -> bool:
    raw = str(text or '').strip()
    if not raw:
        return False
    first_line = raw.splitlines()[0].strip()
    if len(first_line) > 120:
        return False
    return bool(
        re.match(r'^[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё-]+(?:\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё-]+){0,2}\s*[—–,-]', first_line)
        or re.match(r'^[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё-]+(?:\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё-]+){0,2}\s*,', first_line)
    )


def _persona_profile_signal_score(text: str) -> int:
    raw = str(text or '').strip()
    lowered = normalize_name(raw)
    score = 0
    if len([token for token in lowered.split() if token]) >= 18:
        score += 1
    if raw.count(',') >= 4:
        score += 1
    if _looks_like_profile_name_lead(raw):
        score += 1
    if _behavioral_description_density(raw) >= 2:
        score += 1
    if _persona_profile_action_density(raw) >= 2:
        score += 1
    if any(marker in lowered for marker, _ in _SPEECH_HINT_MARKERS):
        score += 1
    return score


def _should_continue_persona_specification(
    *,
    normalized_text: str,
    analysis: MessageAnalysis,
    interaction_frame: InteractionFrame,
    route_memory: RouteMemory,
) -> bool:
    previous_route = str(route_memory.previous_route or '').strip().lower()
    if previous_route != 'persona_specification':
        return False
    if str(interaction_frame.message_kind or '').strip().lower() not in {'statement', 'mixed'}:
        return False
    if bool(interaction_frame.question_present):
        return False
    if _contains_any(normalized_text, _META_MARKERS) or _contains_any(normalized_text, _DEBUG_MARKERS) or _contains_any(normalized_text, _FILE_MARKERS):
        return False
    if bool(interaction_frame.explicit_persona_switch) or _contains_any(normalized_text, _HYPOTHETICAL_MARKERS):
        return False
    return _persona_profile_signal_score(normalized_text) >= 2 or _looks_like_profile_name_lead(normalized_text)


_EXPLICIT_SAVE_PERSONA_MARKERS = (
    'запомни эту персону',
    'сохрани эту персону',
    'создай персонажа',
    'создай личность',
    'сохрани как персону',
    'save this persona',
    'remember this character',
    'create a persona',
    'add this persona',
    'register this persona',
)
_EPHEMERAL_ROLEPLAY_SIGNALS = (
    'ты — ',
    'ты—',
    'оставаясь в этой роли',
    'stay in character',
    'staying in character',
    'теперь отвечай на мои сообщения',
    'отвечай на мои сообщения, оставаясь',
    'ты должен вести себя',
    'ты играешь роль',
    'твоя задача — не играть',
    'не играй театрально',
    'ты не персонаж',
)


def decide_persona_creation(
    text: str,
    *,
    candidate_name: str = '',
    request_type: str = '',
    source: str = 'user',
) -> dict[str, Any]:
    """
    Deterministic decision tree: should a real registered persona head be created?

    Returns a dict with keys:
      - create: bool — True = register a persona head; False = keep it ephemeral
      - reason: str — short code for why
      - candidate_name: str — cleaned name (empty means no usable name extracted)
    """
    lowered = str(text or '').lower().strip()
    clean_name = str(candidate_name or '').strip()

    # NEVER create system-named personas
    _system_names = frozenset({
        'generate_persona', 'generate', 'create_persona', 'new_persona',
        'unnamed', 'unknown', 'default', 'assistant', 'persona',
    })
    if clean_name.lower() in _system_names or clean_name.lower().startswith('generate_'):
        return {'create': False, 'reason': 'system_reserved_name', 'candidate_name': ''}

    # Source guard: graph extraction and system operations never create personas
    if source in {'graph_extraction', 'system', 'file_ingestion', 'background'}:
        return {'create': False, 'reason': 'non_user_source', 'candidate_name': clean_name}

    # Explicit inline roleplay → ephemeral: user is playing a scene, not registering a persona
    if any(marker in lowered for marker in _EPHEMERAL_ROLEPLAY_SIGNALS):
        return {'create': False, 'reason': 'inline_roleplay_ephemeral', 'candidate_name': ''}

    # Explicit user request to save a persona → create
    if any(marker in lowered for marker in _EXPLICIT_SAVE_PERSONA_MARKERS):
        return {'create': True, 'reason': 'explicit_save_request', 'candidate_name': clean_name}

    # Persona specification / assignment with a valid name → create
    if request_type in {'persona_specification', 'persona_assignment'} and clean_name:
        return {'create': True, 'reason': 'persona_spec_with_name', 'candidate_name': clean_name}

    # No valid name → no creation
    if not clean_name:
        return {'create': False, 'reason': 'no_candidate_name', 'candidate_name': ''}

    # Default: do not create unless explicitly requested
    return {'create': False, 'reason': 'no_creation_signal', 'candidate_name': clean_name}


def _classify_request_type(
    *,
    normalized_text: str,
    analysis: MessageAnalysis,
    interaction_frame: InteractionFrame,
    explicit_context: str,
) -> str:
    profile_signal_score = _persona_profile_signal_score(normalized_text)
    if _contains_any(normalized_text, _META_MARKERS) or _contains_any(normalized_text, _DEBUG_MARKERS):
        return 'meta_instruction'
    if _contains_any(normalized_text, _ARCHITECTURE_MARKERS) and (_contains_any(normalized_text, _FILE_MARKERS) or 'проект' in normalize_name(explicit_context)):
        return 'project_architecture_request'
    if bool(str(explicit_context or '').strip()) or _contains_any(normalized_text, _FILE_MARKERS):
        return 'document_request'
    if _contains_any(normalized_text, _PERSONA_ASSIGN_MARKERS):
        return 'persona_assignment'
    explicit_spec = _contains_any(normalized_text, _PERSONA_SPEC_MARKERS)
    persona_analysis = _contains_any(normalized_text, _PERSONA_ANALYSIS_MARKERS)
    hypothetical = bool(interaction_frame.explicit_persona_switch) or _contains_any(normalized_text, _HYPOTHETICAL_MARKERS) or _is_character_definition(normalized_text)
    behavioral_density = _behavioral_description_density(normalized_text)
    message_kind = str(interaction_frame.message_kind or '').strip().lower()
    question_present = bool(interaction_frame.question_present)
    if explicit_spec:
        return 'persona_specification'
    if persona_analysis:
        return 'persona_analysis'
    if hypothetical:
        return 'roleplay_prompt'
    if (
        message_kind in {'statement', 'mixed'}
        and not question_present
        and profile_signal_score >= 3
    ):
        return 'persona_specification'
    if analysis.selected_head or analysis.session_persona or str(interaction_frame.topic_mode or '').strip().lower() == 'persona_self':
        return 'persona_chat'
    if message_kind in {'statement', 'mixed'} and not question_present and behavioral_density >= 3:
        return 'persona_specification'
    if message_kind == 'question':
        return 'factual_query'
    return 'general_chat'


def _merge_unique_items(primary: list[str], inherited: list[str], *, limit: int = 6) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(primary or []) + list(inherited or []):
        clean = str(item or '').strip()
        if not clean:
            continue
        key = normalize_name(clean)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(clean)
        if len(merged) >= max(int(limit or 0), 1):
            break
    return merged


_INLINE_ROLEPLAY_ROUTES = frozenset({'hypothetical_roleplay', 'persona_graph_reasoning'})
_INLINE_ROLEPLAY_MARKERS_FAST = (
    'ты — ', 'ты—', 'оставаясь в этой роли', 'stay in character',
    'staying in character', 'теперь отвечай на мои сообщения',
    'ты играешь роль', 'ты должен вести себя',
)


def _should_continue_hypothetical(
    *,
    preprocessing: RequestPreprocessing,
    interaction_frame: InteractionFrame,
    route_memory: RouteMemory,
) -> bool:
    previous_route = str(route_memory.previous_route or '').strip().lower()
    previous_persona = str(route_memory.previous_persona_name or '').strip()
    if previous_route not in _INLINE_ROLEPLAY_ROUTES:
        return False
    # A real registered persona is active — not an inline roleplay session
    if previous_persona and previous_route != 'hypothetical_roleplay':
        return False
    if preprocessing.meta_request or preprocessing.file_related or preprocessing.system_debug:
        return False
    if bool(interaction_frame.explicit_persona_switch):
        return False
    message_kind = str(interaction_frame.message_kind or '').strip().lower()
    if message_kind not in {'question', 'statement', 'mixed'}:
        return False
    routed = f" {normalize_name(str(interaction_frame.routed_message or ''))} "
    followup_mode = str(interaction_frame.followup_mode or '').strip().lower()
    if followup_mode in {'followup_on_previous_topic', 'followup_on_persona'}:
        return True
    if any(marker in routed for marker in _SECOND_PERSON_FOLLOWUP_MARKERS):
        return True
    if any(normalize_name(marker) in routed for marker in _HYPOTHETICAL_CONTINUATION_MARKERS):
        return True
    return False



_CHARACTER_DEF_RE = re.compile(
    r'(?:^|\n)\s*ты\s*[—–-]\s*\S',   # "Ты — Name" (em/en/hyphen dash)
    re.IGNORECASE,
)

def _is_character_definition(raw_text: str) -> bool:
    """True if text starts with a Russian character sheet ('Ты — Катерина. ...')."""
    return bool(_CHARACTER_DEF_RE.search(raw_text[:200]))


def _looks_lightweight(text: str) -> bool:
    clean = _clean_text(text)
    lowered = normalize_name(clean)
    token_count = len([token for token in lowered.split() if token])
    if _contains_any(clean, _DIRECT_SELF_QUERY_MARKERS):
        return False
    if token_count <= 8 and _contains_any(clean, _GREETING_MARKERS):
        return True
    if token_count <= 16 and _contains_any(clean, _SOCIAL_INTRO_MARKERS):
        return True
    return False



def _needs_clarification(text: str, *, current_entity: str, entities: list[str]) -> bool:
    lowered_tokens = [token for token in normalize_name(text).split() if token]
    if not lowered_tokens:
        return False
    has_pronoun = any(token in _CLARIFICATION_PRONOUNS for token in lowered_tokens)
    if not has_pronoun:
        return False
    if current_entity or entities:
        return False
    return True



def build_request_envelope(*, request_id: str, session_id: str, raw_text: str) -> RequestEnvelope:
    return RequestEnvelope(
        request_id=str(request_id or '').strip(),
        session_id=str(session_id or '').strip(),
        raw_text=str(raw_text or ''),
        normalized_text=_clean_text(raw_text),
        timestamp=_utc_now(),
    )



def preprocess_request(
    *,
    envelope: RequestEnvelope,
    analysis: MessageAnalysis,
    interaction_frame: InteractionFrame,
    explicit_context: str = '',
    route_memory: RouteMemory | None = None,
) -> RequestPreprocessing:
    previous = route_memory or RouteMemory()
    normalized = envelope.normalized_text
    detected_language = normalize_language_code(analysis.user_state.language or detect_language_code(normalized, fallback='en'), fallback='en')
    request_kind = str(interaction_frame.message_kind or 'statement').strip().lower() or 'statement'
    question_present = bool(interaction_frame.question_present)
    intent_type = str(analysis.user_state.intent or request_kind or 'statement').strip().lower() or 'statement'
    entities = [str(entity.name or '').strip() for entity in list(analysis.entities or []) if str(entity.name or '').strip()]
    hypothetical = bool(interaction_frame.explicit_persona_switch) or _contains_any(normalized, _HYPOTHETICAL_MARKERS) or _is_character_definition(envelope.raw_text)
    meta_request = _contains_any(normalized, _META_MARKERS)
    file_related = bool(str(explicit_context or '').strip()) or _contains_any(normalized, _FILE_MARKERS)
    system_debug = _contains_any(normalized, _DEBUG_MARKERS)
    clarification_needed = _needs_clarification(normalized, current_entity=analysis.current_entity, entities=entities)
    persona_style_traits, speech_style_hints = _extract_persona_style_hints(normalized)
    profile_signal_score = _persona_profile_signal_score(normalized)
    lightweight = _looks_lightweight(normalized)
    request_type = _classify_request_type(
        normalized_text=normalized,
        analysis=analysis,
        interaction_frame=interaction_frame,
        explicit_context=explicit_context,
    )
    persona_spec_continuation = _should_continue_persona_specification(
        normalized_text=normalized,
        analysis=analysis,
        interaction_frame=interaction_frame,
        route_memory=previous,
    )
    if persona_spec_continuation:
        request_type = 'persona_specification'
    interaction_mode = 'conversation'
    evidence: list[str] = []
    base = RequestPreprocessing(
        request_type=request_type,
        detected_language=detected_language,
        intent_type=intent_type,
        interaction_mode=interaction_mode,
        request_kind=request_kind,
        hypothetical=hypothetical,
        meta_request=meta_request,
        file_related=file_related,
        system_debug=system_debug,
        clarification_needed=clarification_needed,
        persona_style_traits=persona_style_traits,
        speech_style_hints=speech_style_hints,
        evidence=evidence,
    )
    hypothetical_continuation = _should_continue_hypothetical(
        preprocessing=base,
        interaction_frame=interaction_frame,
        route_memory=previous,
    )
    if hypothetical_continuation:
        persona_style_traits = _merge_unique_items(persona_style_traits, list(previous.persona_style_traits), limit=6)
        speech_style_hints = _merge_unique_items(speech_style_hints, list(previous.speech_style_hints), limit=6)

    if request_type == 'persona_specification':
        interaction_mode = 'persona_specification'
        evidence.append('persona_specification_detected')
        if persona_spec_continuation:
            evidence.append('persona_specification_continuation_detected')
        elif profile_signal_score >= 3:
            evidence.append('persona_profile_signal_detected')
    elif request_type == 'persona_assignment':
        interaction_mode = 'persona_assignment'
        evidence.append('persona_assignment_detected')
    elif request_type == 'persona_analysis':
        interaction_mode = 'persona_analysis'
        evidence.append('persona_analysis_detected')
    elif request_type == 'project_architecture_request':
        interaction_mode = 'project_architecture'
        evidence.append('project_architecture_detected')
    elif request_type == 'persona_chat':
        interaction_mode = 'lightweight_conversation' if lightweight else 'persona_dialogue'
        evidence.append('persona_chat_detected')
        if lightweight:
            evidence.append('lightweight_conversation_markers_detected')
    elif request_kind in {'statement', 'mixed'} and not question_present and profile_signal_score >= 3:
        request_type = 'persona_specification'
        interaction_mode = 'persona_specification'
        evidence.append('persona_profile_signal_detected')
    elif meta_request:
        interaction_mode = 'meta_analysis'
        evidence.append('meta_markers_detected')
    elif hypothetical or hypothetical_continuation:
        interaction_mode = 'hypothetical'
        evidence.append('hypothetical_continuation_detected' if hypothetical_continuation and not hypothetical else 'hypothetical_or_roleplay_markers_detected')
    elif file_related:
        interaction_mode = 'document_analysis'
        evidence.append('file_or_project_markers_detected')
    elif lightweight:
        interaction_mode = 'lightweight_conversation'
        evidence.append('lightweight_conversation_markers_detected')
    elif interaction_frame.explicit_persona_switch or analysis.selected_head or analysis.session_persona or interaction_frame.topic_mode == 'persona_self':
        interaction_mode = 'persona_dialogue'
        evidence.append('persona_context_detected')
    elif system_debug:
        interaction_mode = 'meta_analysis'
        evidence.append('system_debug_markers_detected')
    elif request_kind == 'question':
        interaction_mode = 'factual_query'
        evidence.append('question_route_default')
    else:
        interaction_mode = 'conversation'
        evidence.append('statement_route_default')

    if clarification_needed:
        evidence.append('referential_ambiguity_without_history')
    if persona_style_traits:
        evidence.append('persona_style_traits_detected')
    if speech_style_hints:
        evidence.append('speech_style_hints_detected')

    return RequestPreprocessing(
        request_type=request_type,
        detected_language=detected_language,
        intent_type=intent_type,
        interaction_mode=interaction_mode,
        request_kind=request_kind,
        hypothetical=hypothetical,
        hypothetical_continuation=hypothetical_continuation,
        meta_request=meta_request,
        file_related=file_related,
        system_debug=system_debug,
        clarification_needed=clarification_needed,
        persona_style_traits=persona_style_traits,
        speech_style_hints=speech_style_hints,
        evidence=evidence,
    )



def select_route(
    *,
    envelope: RequestEnvelope,
    preprocessing: RequestPreprocessing,
    analysis: MessageAnalysis,
    interaction_frame: InteractionFrame,
    selected_persona: str = '',
    route_memory: RouteMemory | None = None,
) -> RouteDecision:
    previous = route_memory or RouteMemory()
    reason_codes: list[str] = []
    secondary_flags: list[str] = []
    evidence = list(preprocessing.evidence)
    has_persona = bool(str(selected_persona or analysis.selected_head or analysis.session_persona or '').strip())
    explicit_persona_ui = bool(str(selected_persona or '').strip())
    followup = str(interaction_frame.followup_mode or '').strip().lower()

    normalized_text = normalize_name(envelope.normalized_text)
    # persona_chat_fast_path triggers when:
    # - a persona is active AND the message is clearly conversational/chat
    # - OR the user explicitly selected a persona in the UI (personality_name param) and the
    #   message is a simple question or general chat (not a spec/assignment/doc/meta request)
    _conversational_types = {'persona_chat', 'general_chat'}
    direct_self_query = _contains_any(envelope.normalized_text, _DIRECT_SELF_QUERY_MARKERS)
    has_explicit_topic_entity = bool(
        str(analysis.current_entity or '').strip()
        or str(interaction_frame.topic_entity or '').strip()
        or list(analysis.entities or [])
    )
    persona_fast_candidate = bool(
        has_persona
        and not preprocessing.file_related
        and not preprocessing.meta_request
        and not preprocessing.system_debug
        and not preprocessing.hypothetical
        and not preprocessing.hypothetical_continuation
        and not direct_self_query
        and not has_explicit_topic_entity
        and (
            preprocessing.request_type == 'persona_chat'
            or (explicit_persona_ui and preprocessing.request_type in _conversational_types)
            or preprocessing.interaction_mode == 'lightweight_conversation'
            or _looks_lightweight(envelope.normalized_text)
        )
    )

    if preprocessing.clarification_needed:
        selected_route = 'clarification_request'
        reason_codes.append('route:clarification')
    elif preprocessing.request_type == 'persona_assignment':
        selected_route = 'persona_assignment'
        reason_codes.append('route:persona_assignment')
    elif preprocessing.request_type == 'persona_specification':
        selected_route = 'persona_specification'
        reason_codes.append('route:persona_specification')
    elif preprocessing.request_type == 'persona_analysis':
        selected_route = 'persona_dialogue_analysis'
        reason_codes.append('route:persona_dialogue_analysis')
    elif preprocessing.request_type == 'project_architecture_request':
        selected_route = 'project_document_analysis'
        reason_codes.append('route:project_document_analysis')
    elif preprocessing.meta_request or preprocessing.system_debug:
        selected_route = 'meta_previous_answer'
        reason_codes.append('route:meta_previous_answer')
    elif preprocessing.request_type == 'roleplay_prompt' or preprocessing.hypothetical or preprocessing.hypothetical_continuation:
        selected_route = 'hypothetical_roleplay'
        reason_codes.append('route:hypothetical_continuation' if preprocessing.hypothetical_continuation and not preprocessing.hypothetical else 'route:hypothetical_roleplay')
    elif preprocessing.request_type == 'document_request' or preprocessing.file_related:
        selected_route = 'project_document_analysis'
        reason_codes.append('route:project_document_analysis')
    elif persona_fast_candidate:
        selected_route = 'persona_chat_fast_path'
        reason_codes.append('route:persona_chat_fast_path')
    elif preprocessing.interaction_mode == 'lightweight_conversation':
        selected_route = 'lightweight_conversation'
        reason_codes.append('route:lightweight_conversation')
    elif has_persona or str(interaction_frame.topic_mode or '').strip().lower() == 'persona_self' or followup == 'followup_on_persona':
        selected_route = 'persona_graph_reasoning'
        reason_codes.append('route:persona_graph_reasoning')
    else:
        selected_route = 'factual_answer'
        reason_codes.append('route:factual_answer')

    requires_history = (
        selected_route in {'meta_previous_answer', 'persona_dialogue_analysis'}
        or followup in {'followup_on_previous_topic', 'followup_on_persona'}
        or bool(preprocessing.hypothetical_continuation)
    )
    requires_persona = selected_route in {'persona_graph_reasoning', 'persona_assignment', 'persona_chat_fast_path'} or (selected_route in {'hypothetical_roleplay', 'persona_dialogue_analysis', 'lightweight_conversation'} and has_persona)
    requires_graph = selected_route in {'factual_answer', 'project_document_analysis', 'persona_graph_reasoning'}
    requires_llm = selected_route not in {'clarification_request', 'persona_specification', 'persona_assignment'}
    strict_grounding = selected_route in {'factual_answer', 'project_document_analysis', 'persona_graph_reasoning'}
    if selected_route in {'hypothetical_roleplay', 'meta_previous_answer', 'lightweight_conversation', 'persona_chat_fast_path', 'clarification_request', 'persona_specification', 'persona_assignment', 'persona_dialogue_analysis'}:
        strict_grounding = False

    response_style = {
        'clarification_request': 'clarifying_question',
        'persona_assignment': 'persona_activation',
        'persona_specification': 'persona_creation',
        'persona_dialogue_analysis': 'dialogue_review',
        'meta_previous_answer': 'self_analysis',
        'hypothetical_roleplay': 'hypothetical_answer',
        'project_document_analysis': 'project_analysis',
        'persona_graph_reasoning': 'persona_first_person',
        'persona_chat_fast_path': 'persona_first_person_brief',
        'lightweight_conversation': 'brief_conversation',
        'factual_answer': 'grounded_answer',
    }[selected_route]
    validation_mode = {
        'clarification_request': 'clarification',
        'persona_assignment': 'structured_action',
        'persona_specification': 'structured_action',
        'persona_dialogue_analysis': 'dialogue_review',
        'meta_previous_answer': 'meta_history',
        'hypothetical_roleplay': 'hypothetical_route',
        'project_document_analysis': 'grounded_project',
        'persona_graph_reasoning': 'persona_consistency',
        'persona_chat_fast_path': 'persona_fast',
        'lightweight_conversation': 'lightweight',
        'factual_answer': 'grounded_factual',
    }[selected_route]
    fast_path = selected_route in {'lightweight_conversation', 'persona_chat_fast_path', 'clarification_request', 'meta_previous_answer', 'persona_specification', 'persona_assignment'}

    if requires_history:
        secondary_flags.append('history_required')
    if requires_graph:
        secondary_flags.append('graph_required')
    if requires_persona:
        secondary_flags.append('persona_required')
    if not strict_grounding:
        secondary_flags.append('strict_grounding_disabled')
    if fast_path:
        secondary_flags.append('fast_path')
    if preprocessing.hypothetical_continuation:
        secondary_flags.append('hypothetical_continuation')
    if str(previous.previous_route or '').strip():
        secondary_flags.append('route_memory_available')
    if selected_route == 'persona_specification':
        secondary_flags.append('persona_creation_action')
    if selected_route == 'persona_assignment':
        secondary_flags.append('persona_assignment_action')

    return RouteDecision(
        selected_route=selected_route,
        request_type=preprocessing.request_type,
        detected_language=preprocessing.detected_language,
        intent_type=preprocessing.intent_type,
        interaction_mode=preprocessing.interaction_mode,
        requires_history=requires_history,
        requires_graph=requires_graph,
        requires_persona=requires_persona,
        requires_llm=requires_llm,
        strict_grounding=strict_grounding,
        response_style=response_style,
        validation_mode=validation_mode,
        fast_path=fast_path,
        persona_style_traits=list(preprocessing.persona_style_traits),
        speech_style_hints=list(preprocessing.speech_style_hints),
        secondary_flags=secondary_flags,
        reason_codes=reason_codes,
        evidence=evidence,
    )



def plan_capabilities(route: RouteDecision) -> CapabilityPlan:
    context_sources: list[str] = []
    skipped: list[str] = []
    preferred_generation_mode = 'chat_llm'
    if route.requires_history:
        context_sources.append('session_history')
    if route.requires_graph:
        context_sources.append('knowledge_graph')
    if route.requires_persona:
        context_sources.append('persona_graph')
    if route.selected_route == 'clarification_request':
        preferred_generation_mode = 'deterministic'
    elif route.selected_route == 'lightweight_conversation':
        preferred_generation_mode = 'deterministic_or_fast_llm'
    elif route.selected_route == 'persona_chat_fast_path':
        preferred_generation_mode = 'persona_fast_llm'
    elif route.selected_route == 'meta_previous_answer':
        preferred_generation_mode = 'chat_llm_meta'
    elif route.selected_route == 'persona_dialogue_analysis':
        preferred_generation_mode = 'chat_llm_review'
    elif route.selected_route in {'persona_specification', 'persona_assignment'}:
        preferred_generation_mode = 'structured_persona_action'
    if not route.requires_persona:
        skipped.extend(['persona_update', 'head_selection', 'mood_research', 'state_transition'])
    if route.selected_route in {'persona_specification', 'persona_assignment'}:
        skipped.extend(['graph_context_expansion', 'history_expansion', 'chat_generation'])
    if route.selected_route == 'persona_chat_fast_path':
        skipped.extend(['graph_context_expansion', 'heavy_persona_pipeline', 'head_selection', 'mood_research', 'state_transition'])
    if not route.requires_graph:
        skipped.append('graph_context_expansion')
    if not route.requires_history:
        skipped.append('history_expansion')
    return CapabilityPlan(
        requires_history=route.requires_history,
        requires_graph=route.requires_graph,
        requires_persona=route.requires_persona,
        requires_llm=route.requires_llm,
        use_context_builder=(route.requires_graph or route.selected_route == 'persona_dialogue_analysis') and route.selected_route not in {'persona_specification', 'persona_assignment', 'persona_chat_fast_path'},
        use_heavy_persona_pipeline=route.requires_persona and route.selected_route not in {'persona_specification', 'persona_assignment', 'persona_dialogue_analysis', 'lightweight_conversation', 'persona_chat_fast_path'},
        strict_grounding=route.strict_grounding,
        preferred_generation_mode=preferred_generation_mode,
        context_sources=context_sources,
        skipped_components=skipped,
        reason_codes=list(route.reason_codes),
    )



def render_route_guidance(route: RouteDecision) -> str:
    guidance = {
        'clarification_request': [
            'Route: clarification request.',
            'Ask one concise clarifying question that resolves the ambiguous reference.',
            'Do not answer the unresolved question as if context already exists.',
        ],
        'persona_assignment': [
            'Route: persona assignment.',
            'Do not generate a narrative answer.',
            'Activate the requested validated persona and confirm the assignment briefly.',
        ],
        'persona_specification': [
            'Route: persona specification.',
            'Do not generate an essay in place of the requested action.',
            'Parse the description into a structured persona object, validate it, store it, and activate it when requested or implied.',
        ],
        'persona_dialogue_analysis': [
            'Route: persona dialogue analysis.',
            'Analyze the dialogue for psychological plausibility and role consistency, not for literary beauty.',
            'Do not answer as the persona and do not continue the scene.',
            'Look for overconfidence, lecturing, invented facts, imposed emotions, rushed conclusions, broken pressure logic, and speech that sounds too polished.',
            'For each issue, answer in this structure: ошибка, почему это ошибка, как лучше, исправленный вариант реплики.',
        ],
        'meta_previous_answer': [
            'Route: meta-analysis of the previous answer.',
            'Inspect the recent dialogue and explain the earlier answer, mistake, or mismatch directly.',
            'Do not pretend the user asked a fresh factual question.',
        ],
        'hypothetical_roleplay': [
            'Route: hypothetical or roleplay.',
            'Do not refuse only because factual grounding is incomplete.',
            'Answer within the hypothetical frame or persona frame selected upstream.',
        ],
        'project_document_analysis': [
            'Route: project or document analysis.',
            'Use uploaded/project context when available and keep the answer analytical.',
        ],
        'persona_graph_reasoning': [
            'Route: persona and graph guided reasoning.',
            'Answer from the selected persona perspective when persona context is available.',
            'Use graph context as support, not as a replacement for the persona voice.',
        ],
        'persona_chat_fast_path': [
            'Route: persona chat fast path.',
            'Skip heavy analysis and answer from the active persona immediately.',
            'Use only compact persona context and recent session continuity when needed.',
            'Do not turn a simple social or direct persona turn into graph-heavy reasoning or factual disclaimers.',
        ],
        'lightweight_conversation': [
            'Route: lightweight conversation.',
            'Reply briefly and naturally.',
            'Do not over-escalate into analysis or grounding disclaimers.',
        ],
        'factual_answer': [
            'Route: grounded factual answer.',
            'Use grounded context when available and say plainly when a fact is uncertain.',
        ],
    }
    lines = list(guidance.get(route.selected_route, []))
    if route.selected_route == 'lightweight_conversation' and route.requires_persona:
        lines.extend(
            [
                'The user is speaking to an active persona in a lightweight social exchange.',
                'Answer in first person from that persona, but keep it short, natural, and socially direct.',
                'Do not turn a greeting or introduction into a grounded factual explanation.',
            ]
        )
    if route.selected_route == 'persona_chat_fast_path':
        lines.extend(
            [
                'Keep the answer short, first-person, and socially direct.',
                'Do not expand into explanation, lecture, or graph dump.',
            ]
        )
    if route.persona_style_traits:
        lines.append(f"Requested persona disposition: {', '.join(route.persona_style_traits)}.")
    if route.speech_style_hints:
        lines.append('Requested speaking manner:')
        lines.extend(f"- {hint}." for hint in list(route.speech_style_hints)[:6])
    if route.selected_route in {'hypothetical_roleplay', 'persona_graph_reasoning', 'persona_chat_fast_path'} and _fragile_persona_requested(route):
        lines.extend(
            [
                'Keep the external voice weaker under pressure, not sharper or more dominant.',
                'Prefer 1 to 3 short lines, awkward hesitation, and incomplete thoughts over polished analysis.',
                'Do not explain the situation, lecture, command, or step outside the character voice.',
                'When the topic is uncomfortable or tied to dependence, avoid, deflect, show irritation, and stay defensive instead of becoming clearer or stronger.',
                'If a significant person is mentioned, reduce confidence and directness rather than taking control of the dialogue.',
                'Do not directly admit feelings in a clean articulate way when the persona is ashamed, dependent, or proud.',
            ]
        )
    return '\n'.join(lines).strip()



def deterministic_reply(route: RouteDecision, *, language: str, history_available: bool = False) -> str:
    target = normalize_language_code(language, fallback='en')
    if route.selected_route == 'clarification_request':
        return 'Что именно вы имеете в виду?' if target == 'ru' else 'What exactly do you mean?'
    if route.selected_route == 'persona_assignment':
        return 'Личность активирована.' if target == 'ru' else 'The persona is now active.'
    if route.selected_route == 'persona_specification':
        return 'Личность создана и сохранена как структурированный профиль.' if target == 'ru' else 'The persona has been created and stored as a structured profile.'
    if route.selected_route == 'lightweight_conversation':
        if route.requires_persona:
            return ''
        return 'Привет. Чем помочь?' if target == 'ru' else 'Hi. How can I help?'
    if route.selected_route == 'persona_chat_fast_path':
        return ''
    if route.selected_route == 'hypothetical_roleplay':
        return (
            'В этой гипотетической ситуации я бы сначала выделил главный риск, затем задал ближайший безопасный шаг и действовал спокойно по нему.'
            if target == 'ru'
            else 'In that hypothetical situation, I would identify the main risk first, set the next safe step, and act calmly from there.'
        )
    if route.selected_route == 'meta_previous_answer' and not history_available:
        return (
            'Я не могу разобрать предыдущий ответ, потому что в текущей сессии еще нет прошлой реплики ассистента.'
            if target == 'ru'
            else 'I cannot inspect a previous answer because this session does not contain an earlier assistant turn yet.'
        )
    if route.selected_route == 'persona_dialogue_analysis' and not history_available:
        return (
            'Я не могу разобрать отыгрыш, потому что в текущей сессии еще нет диалога персонажа для анализа.'
            if target == 'ru'
            else 'I cannot review the roleplay yet because this session does not contain enough dialogue to analyze.'
        )
    return ''


_PERSONA_DIALOGUE_REVIEW_MARKERS = (
    'ошибка',
    'почему это ошибка',
    'как лучше',
    'исправленный вариант',
    'error',
    'why this is an error',
    'better version',
    'corrected line',
)



def validate_response(
    *,
    route: RouteDecision,
    reply: str,
    fallback_triggered: bool,
    fallback_reason_code: str,
    history_used: bool,
    graph_used: bool,
    persona_used: bool,
    history_available: bool,
    model_budget: dict[str, Any] | None = None,
    persona_reference_text: str = '',
    request_text: str = '',
) -> ResponseValidation:
    text = _clean_text(reply)
    lowered = normalize_name(text)
    budget = dict(model_budget or {})
    validation = ResponseValidation(
        ok=True,
        validation_mode=route.validation_mode,
        route_match=True,
        relevance_match=bool(text),
        used_history=history_used,
        used_graph=graph_used,
        used_persona=persona_used,
        fallback_triggered=fallback_triggered,
        fallback_reason_code=str(fallback_reason_code or '').strip(),
        reason_codes=list(route.reason_codes),
    )
    output_truncated = bool(budget.get('output_truncated') or budget.get('output_budget_too_small'))
    repetitive = _looks_repetitive_reply(text)
    abrupt_cut = bool(output_truncated and _looks_abruptly_cut(text))

    if route.requires_llm and (output_truncated or repetitive or abrupt_cut):
        validation.ok = False
        validation.route_match = False
        validation.relevance_match = bool(text)
        fragile_style_requested = route.selected_route in {'hypothetical_roleplay', 'persona_graph_reasoning'} and _fragile_persona_requested(route, persona_reference_text)
        if output_truncated:
            validation.mismatch_reason = 'reply_was_truncated_by_output_budget'
            validation.reason_codes.append('validation:output_truncated')
        elif repetitive:
            validation.mismatch_reason = 'reply_contains_repetitive_looping_text'
            validation.reason_codes.append('validation:repetition_loop')
        else:
            validation.mismatch_reason = 'reply_looks_abruptly_cut'
            validation.reason_codes.append('validation:abrupt_cut')
        validation.repair_strategy = 'regenerate_style_guard' if fragile_style_requested else 'regenerate_with_budget'
        return validation

    if route.selected_route == 'hypothetical_roleplay' and (fallback_reason_code == 'no_grounding' or any(marker in lowered for marker in _LIMITATION_MARKERS)):
        validation.ok = False
        validation.route_match = False
        validation.mismatch_reason = 'hypothetical_request_misrouted_to_grounding_limitation'
        validation.repair_strategy = 'regenerate_hypothetical'
        validation.reason_codes.append('validation:hypothetical_route_mismatch')
        return validation

    if route.selected_route == 'meta_previous_answer':
        if not history_available:
            validation.ok = True
            validation.reason_codes.append('validation:meta_without_history_explained')
            return validation
        if not history_used:
            validation.ok = False
            validation.route_match = False
            validation.mismatch_reason = 'meta_question_did_not_load_history'
            validation.repair_strategy = 'regenerate_meta'
            validation.reason_codes.append('validation:meta_history_missing')
            return validation
        if fallback_reason_code == 'no_grounding':
            validation.ok = False
            validation.route_match = False
            validation.mismatch_reason = 'meta_question_degraded_to_grounding_fallback'
            validation.repair_strategy = 'regenerate_meta'
            validation.reason_codes.append('validation:meta_route_mismatch')
            return validation

    if route.selected_route == 'persona_dialogue_analysis':
        if not history_available and not history_used and not text:
            validation.ok = False
            validation.route_match = False
            validation.mismatch_reason = 'dialogue_review_without_source_material'
            validation.repair_strategy = 'deterministic_clarification'
            validation.reason_codes.append('validation:dialogue_review_missing_source')
            return validation
        structure_hits = sum(1 for marker in _PERSONA_DIALOGUE_REVIEW_MARKERS if marker in lowered)
        if structure_hits < 3:
            validation.ok = False
            validation.route_match = False
            validation.mismatch_reason = 'dialogue_review_missing_required_structure'
            validation.repair_strategy = 'regenerate_meta'
            validation.reason_codes.append('validation:dialogue_review_structure_missing')
            return validation
    elif route.requires_llm and any(marker in lowered for marker in _OUTPUT_SCAFFOLD_MARKERS):
        validation.ok = False
        validation.route_match = False
        validation.mismatch_reason = 'reply_leaked_generation_scaffold'
        validation.repair_strategy = (
            'regenerate_style_guard'
            if route.selected_route in {'persona_graph_reasoning', 'persona_chat_fast_path', 'hypothetical_roleplay', 'lightweight_conversation', 'meta_previous_answer'}
            else 'regenerate_with_budget'
        )
        validation.reason_codes.append('validation:reply_scaffold_leak')
        return validation

    if route.selected_route in {'lightweight_conversation', 'persona_chat_fast_path'} and graph_used:
        validation.reason_codes.append('validation:heavy_graph_was_not_needed')
    if route.selected_route in {'lightweight_conversation', 'persona_chat_fast_path'} and any(marker in lowered for marker in _LIMITATION_MARKERS):
        validation.ok = False
        validation.route_match = False
        validation.mismatch_reason = 'fast_path_degraded_to_grounding_limitation'
        validation.repair_strategy = 'regenerate_with_budget'
        validation.reason_codes.append('validation:fast_path_route_mismatch')
        return validation

    if route.selected_route in {'persona_graph_reasoning', 'persona_chat_fast_path'} and persona_used and any(marker in lowered for marker in _LIMITATION_MARKERS):
        validation.ok = False
        validation.route_match = False
        validation.mismatch_reason = 'persona_route_answer_degraded_to_limitation'
        validation.repair_strategy = 'regenerate_persona'
        validation.reason_codes.append('validation:persona_route_mismatch')
        return validation

    # Persona voice break: LLM switched to analysis/assistant mode instead of speaking as persona
    persona_voice_routes = {'persona_chat_fast_path', 'hypothetical_roleplay', 'persona_graph_reasoning', 'lightweight_conversation'}
    if route.selected_route in persona_voice_routes and persona_used and any(marker in lowered for marker in _PERSONA_BREAK_MARKERS):
        validation.ok = False
        validation.route_match = False
        validation.mismatch_reason = 'persona_broke_into_analysis_mode'
        validation.repair_strategy = 'regenerate_style_guard'
        validation.reason_codes.append('validation:persona_analysis_break')
        return validation

    if route.selected_route in {'persona_graph_reasoning', 'persona_chat_fast_path', 'hypothetical_roleplay'} and _persona_speaks_as_third_person(text, persona_reference_text):
        validation.ok = False
        validation.route_match = False
        validation.mismatch_reason = 'persona_spoke_in_third_person'
        validation.repair_strategy = 'regenerate_style_guard'
        validation.reason_codes.append('validation:persona_third_person_break')
        return validation

    if route.selected_route in {'hypothetical_roleplay', 'persona_graph_reasoning'} and _fragile_persona_requested(route, persona_reference_text):
        style_findings = _fragile_persona_style_findings(
            route,
            text,
            persona_reference_text=persona_reference_text,
            request_text=request_text,
        )
        if style_findings:
            validation.ok = False
            validation.route_match = False
            validation.mismatch_reason = 'reply_failed_fragile_persona_style_guard'
            validation.repair_strategy = 'regenerate_style_guard'
            validation.reason_codes.extend(f'validation:{item}' for item in style_findings)
            return validation

    # Language mismatch: user wrote in Russian, persona responded in English (or vice versa)
    if route.selected_route in persona_voice_routes and persona_used and request_text:
        req_lowered = normalize_name(request_text)
        req_cyrillic = sum(1 for c in req_lowered if 'а' <= c <= 'я' or 'ё' == c)
        resp_cyrillic = sum(1 for c in lowered if 'а' <= c <= 'я' or 'ё' == c)
        req_latin = sum(1 for c in req_lowered if 'a' <= c <= 'z')
        resp_latin = sum(1 for c in lowered if 'a' <= c <= 'z')
        req_is_russian = req_cyrillic > req_latin * 2
        resp_is_english = resp_latin > resp_cyrillic * 2 and resp_latin > 20
        if req_is_russian and resp_is_english:
            validation.ok = False
            validation.route_match = False
            validation.mismatch_reason = 'persona_responded_in_wrong_language'
            validation.repair_strategy = 'regenerate_style_guard'
            validation.reason_codes.append('validation:persona_language_mismatch')
            return validation

    if route.selected_route == 'clarification_request':
        if '?' not in text and not lowered.startswith('what') and not lowered.startswith('which') and not lowered.startswith('кто') and not lowered.startswith('что'):
            validation.ok = False
            validation.route_match = False
            validation.mismatch_reason = 'clarification_route_did_not_ask_question'
            validation.repair_strategy = 'deterministic_clarification'
            validation.reason_codes.append('validation:clarification_missing')
            return validation

    return validation
