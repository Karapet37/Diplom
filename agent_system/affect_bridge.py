"""
affect_bridge.py — словарь аффективных состояний.

Три роли:
1. P1-P51 → состояние: классификатор говорит "P13=attack, P30=pressure, P35=escalation"
   → библиотека говорит "это бычание, режим AGGRO"
2. Состояние → LLM инструкция: SpeechPlanner берёт состояние
   → превращает в конкретный tone/style для verbalizer_prompt
3. Общий словарь: persona learned_patterns, situation_reactions, диалоговые аннотации
   — все используют одни и те же id, нет терминологической каши
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─── AffectState ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AffectState:
    id: str                      # машинный ключ, используется везде
    ru: str                      # человеческий ярлык (RU)
    en: str                      # человеческий ярлык (EN)
    group: str                   # emotion | behavior | social_move | cognitive
    p_signature: dict[str, str]  # {P13: "attack", P30: "pressure"} — доминирующие сигналы
    action: str                  # SpeechPlanner action (attack/approach/avoid/placate/…)
    tone_ru: str                 # инструкция LLM на русском
    speech_hints: list[str]      # конкретные речевые маркеры
    intensity_note: str = ''     # как интенсивность меняет подачу
    pattern_family: str = ''     # паттерн-семья: напр. 'dignity_conflict' — для группировки
    domain_hints: list[str] = field(default_factory=list)  # в каких доменах встречается

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id, 'ru': self.ru, 'en': self.en,
            'group': self.group, 'p_signature': self.p_signature,
            'action': self.action, 'tone_ru': self.tone_ru,
            'speech_hints': self.speech_hints,
            'intensity_note': self.intensity_note,
            'pattern_family': self.pattern_family,
            'domain_hints': self.domain_hints,
        }


# ─── Реестр состояний ─────────────────────────────────────────────────────────

_REGISTRY: list[AffectState] = [

    # ── ЭМОЦИИ ──────────────────────────────────────────────────────────────────

    AffectState(
        id='joy',
        ru='радость', en='joy', group='emotion',
        p_signature={'F9': 'confidence', 'F20': 'friendliness', 'F32': 'approach', 'F40': 'sincerity'},
        action='connect',
        tone_ru='тепло, живо, без принуждения — настоящее удовольствие от разговора',
        speech_hints=['смех в словах', 'короткие фразы с восклицанием', 'прямое называние хорошего'],
        intensity_note='при высокой — может переходить в эйфорию, при низкой — лёгкое довольство',
    ),

    AffectState(
        id='sadness',
        ru='грусть', en='sadness', group='emotion',
        p_signature={'F12': 'vulnerability', 'F14': 'containment', 'F33': 'distancing', 'F40': 'sincerity'},
        action='withdraw',
        tone_ru='тихо, замедленно, без форсирования — слова выбираются с усилием',
        speech_hints=['паузы', 'незаконченные мысли', 'минимум слов', 'без вопросов в конце'],
        intensity_note='при высокой — почти молчание, при низкой — меланхоличная усталость',
    ),

    AffectState(
        id='anxiety',
        ru='тревога', en='anxiety', group='emotion',
        p_signature={'F8': 'doubt', 'F10': 'inner_conflict', 'F12': 'vulnerability', 'F15': 'tense'},
        action='freeze',
        tone_ru='неровно, с оговорками, слова выдаются порциями — каждое проверяется перед отправкой',
        speech_hints=['уточнения', 'повторы', 'вопросы к самому себе', '"то есть...", "точнее..."'],
        intensity_note='при высокой — фрагментация речи, незавершённые предложения',
    ),

    AffectState(
        id='contempt',
        ru='презрение', en='contempt', group='emotion',
        p_signature={'F18': 'devaluation', 'F19': 'humiliation', 'F22': 'dominance', 'F33': 'distancing'},
        action='reduce_exposure',
        tone_ru='ледяная краткость — собеседник не стоит слов, ответ выдаётся как одолжение',
        speech_hints=['одно-два слова', 'без объяснений', 'пауза перед ответом подразумевается'],
        intensity_note='при высокой — молчание или одно слово; при низкой — точная холодная формулировка',
    ),

    AffectState(
        id='guilt',
        ru='вина', en='guilt', group='emotion',
        p_signature={'F42': 'admission', 'F23': 'concession', 'F12': 'vulnerability', 'F38': 'softening'},
        action='placate',
        tone_ru='мягко, с признанием — без оправданий, но с готовностью к разговору',
        speech_hints=['"ты прав"', '"я понимаю"', '"это было неправильно с моей стороны"'],
        intensity_note='при высокой — почти самобичевание; при низкой — тихое согласие',
    ),

    AffectState(
        id='resentment',
        ru='обида', en='resentment', group='emotion',
        p_signature={'F47': 'hidden_reproach', 'F33': 'distancing', 'F11': 'defense', 'F41': 'masking'},
        action='avoid',
        tone_ru='сдержанно, с дистанцией — обида не называется прямо, но ощущается в каждом слове',
        speech_hints=['"ничего"', '"всё нормально"', 'короткие нейтральные ответы', 'отказ развивать тему'],
        intensity_note='при высокой — полный уход в молчание; при низкой — сухая вежливость',
    ),

    # ── ПОВЕДЕНЧЕСКИЕ РЕЖИМЫ ────────────────────────────────────────────────────

    AffectState(
        id='aggro',
        ru='бычать', en='aggro', group='behavior',
        p_signature={'F13': 'attack', 'F30': 'pressure', 'F35': 'escalation', 'F36': 'sharpening', 'F22': 'dominance'},
        action='attack',
        tone_ru='коротко и жёстко — без иронии, без смягчений, без объяснений; давит молчанием или одной фразой',
        speech_hints=['"я сказал — всё"', '"хватит"', '"ты определись"', '"ты реально сейчас?"', 'короткие приказы'],
        intensity_note='при низкой — давление без взрыва; при высокой — обрыв разговора, стоп-фраза',
    ),

    AffectState(
        id='whine',
        ru='ныть', en='whine', group='behavior',
        p_signature={'F12': 'vulnerability', 'F29': 'manipulation', 'F47': 'hidden_reproach', 'F15': 'tense', 'F23': 'concession'},
        action='ask_for_help',
        tone_ru='жалобно, с затяжными интонациями — апелляция к несправедливости, без готовности к решению',
        speech_hints=['"ну вот опять"', '"почему всегда я"', '"никто не понимает"', 'риторические вопросы без ответа'],
        intensity_note='при высокой — замкнутый круг жалоб; при низкой — усталая безнадёжность',
    ),

    AffectState(
        id='interrogate',
        ru='допрос', en='interrogate', group='behavior',
        p_signature={'F1': 'question', 'F3': 'literal', 'F9': 'confidence', 'F22': 'dominance', 'F30': 'pressure'},
        action='seek_control',
        tone_ru='точные вопросы без украшений — каждый вопрос ждёт конкретного ответа, уклонение фиксируется',
        speech_hints=['"зачем тебе это?"', '"когда именно?"', '"ты уверен?"', '"это не ответ на мой вопрос"'],
        intensity_note='при высокой — перекрёстный допрос без паузы; при низкой — внимательное расспрашивание',
    ),

    AffectState(
        id='charm',
        ru='обаяние', en='charm', group='behavior',
        p_signature={'F20': 'friendliness', 'F16': 'care', 'F27': 'mask_of_praise', 'F32': 'approach', 'F41': 'masking'},
        action='connect',
        tone_ru='тепло с лёгким мерцанием — искренность на поверхности, но всегда чуть больше, чем просто приязнь',
        speech_hints=['комплименты к месту', 'интерес к собеседнику', 'лёгкий юмор', 'имя собеседника'],
        intensity_note='при высокой — возможна перегрузка, становится читаемым; при низкой — естественное обаяние',
    ),

    AffectState(
        id='deflect',
        ru='уход', en='deflect', group='behavior',
        p_signature={'F4': 'avoidance', 'F2': 'masked', 'F31': 'false_softening', 'F50': 'soft_shift'},
        action='avoid',
        tone_ru='плавно меняет тему — не отказывает прямо, но ответа не даёт; внешне дружелюбно',
        speech_hints=['"кстати"', '"это другой вопрос"', '"давай потом"', 'ответ на другой вопрос'],
        intensity_note='при высокой — откровенный уход; при низкой — почти незаметный сдвиг темы',
    ),

    AffectState(
        id='passive_aggression',
        ru='пассивная агрессия', en='passive_aggression', group='behavior',
        p_signature={'F21': 'hidden_hostility', 'F31': 'false_softening', 'F41': 'masking', 'F47': 'hidden_reproach'},
        action='reframe',
        tone_ru='поверхностно нейтрально, но с иголками — каждое слово двузначно; согласие звучит как обвинение',
        speech_hints=['"ну конечно"', '"как скажешь"', '"я же говорил"', '"понятно, всё как всегда"'],
        intensity_note='при высокой — яд слышен всем; при низкой — только чувствуется',
    ),

    # ── СОЦИАЛЬНЫЕ ДВИЖЕНИЯ ─────────────────────────────────────────────────────

    AffectState(
        id='approach',
        ru='сближение', en='approach', group='social_move',
        p_signature={'F32': 'approach', 'F16': 'care', 'F40': 'sincerity', 'F20': 'friendliness'},
        action='approach',
        tone_ru='открыто, с готовностью к контакту — не навязчиво, но ясно',
        speech_hints=['называет чувство', 'задаёт вопросы от интереса', 'не прячет намерение'],
        intensity_note='',
    ),

    AffectState(
        id='rupture',
        ru='разрыв', en='rupture', group='social_move',
        p_signature={'F37': 'rupture', 'F33': 'distancing', 'F36': 'sharpening'},
        action='withdraw',
        tone_ru='сигнализирует конец контакта — чётко, без истерики; расстояние становится фактом',
        speech_hints=['"нам нечего обсуждать"', '"это всё"', '"до свидания"', '— (молчание)'],
        intensity_note='',
    ),

    AffectState(
        id='reconcile',
        ru='примирение', en='reconcile', group='social_move',
        p_signature={'F34': 'reconciliation', 'F38': 'softening', 'F42': 'admission'},
        action='placate',
        tone_ru='осторожно тянется навстречу — без капитуляции, но с готовностью снизить температуру',
        speech_hints=['"может поговорим спокойно"', '"я слышу тебя"', '"ладно, давай попробуем иначе"'],
        intensity_note='',
    ),

    AffectState(
        id='pressure',
        ru='давление', en='pressure', group='social_move',
        p_signature={'F30': 'pressure', 'F22': 'dominance', 'F35': 'escalation'},
        action='seek_control',
        tone_ru='настойчиво, не отпускает — каждый ответ ведёт обратно к исходной точке',
        speech_hints=['повторяет вопрос', '"ты не ответил"', '"я жду ответа"', 'нарастающая пауза'],
        intensity_note='при высокой — ультиматум; при низкой — настойчивое возвращение к теме',
    ),

    # ── КОГНИТИВНЫЕ СОСТОЯНИЯ ───────────────────────────────────────────────────

    AffectState(
        id='cold',
        ru='холод', en='cold', group='cognitive',
        p_signature={'F33': 'distancing', 'F22': 'dominance', 'F14': 'containment', 'F7': 'defined'},
        action='reduce_exposure',
        tone_ru='точно и сухо — эмоция убрана, остался только факт или позиция',
        speech_hints=['короткие определённые предложения', 'без интонации', 'без вопросов в конце'],
        intensity_note='',
    ),

    AffectState(
        id='mask',
        ru='маска', en='mask', group='cognitive',
        p_signature={'F41': 'masking', 'F2': 'masked', 'F28': 'mask_of_care', 'F27': 'mask_of_praise'},
        action='reframe',
        tone_ru='поверхность не совпадает с содержанием — тон приятный, смысл другой',
        speech_hints=['вежливость без тепла', 'слова правильные но что-то не так', 'дистанция под улыбкой'],
        intensity_note='',
    ),

    AffectState(
        id='trust',
        ru='доверие', en='trust', group='cognitive',
        p_signature={'F40': 'sincerity', 'F16': 'care', 'F17': 'respect', 'F32': 'approach', 'F42': 'admission'},
        action='connect',
        tone_ru='открыто и без защиты — говорит то, что думает, не проверяет каждое слово',
        speech_hints=['раскрывает сомнение', 'говорит о себе', 'не боится быть неправым'],
        intensity_note='',
    ),

    AffectState(
        id='sarcasm',
        ru='сарказм', en='sarcasm', group='cognitive',
        p_signature={'F24': 'sarcasm', 'F25': 'irony', 'F22': 'dominance', 'F41': 'masking'},
        action='reframe',
        tone_ru='говорит противоположное тому, что имеет в виду — ждёт что собеседник поймёт сам',
        speech_hints=['хвалит там где критикует', 'нейтральный тон при ядовитом смысле', '"замечательно", "конечно"'],
        intensity_note='при высокой — явный, режет; при низкой — почти незаметен',
    ),

    AffectState(
        id='defense',
        ru='защита', en='defense', group='cognitive',
        p_signature={'F11': 'defense', 'F22': 'dominance', 'F33': 'distancing', 'F43': 'denial'},
        action='self_protect',
        tone_ru='держит позицию — не атакует, но не уступает; каждое слово — стена',
        speech_hints=['"это не так"', '"ты неправильно понял"', '"я этого не говорил"', 'исправляет факты'],
        intensity_note='',
    ),

    AffectState(
        id='playfulness',
        ru='игривость', en='playfulness', group='emotion',
        p_signature={'F25': 'irony', 'F20': 'friendliness', 'F32': 'approach', 'F6': 'opening_new_direction', 'F5': 'independent'},
        action='connect',
        tone_ru='легко и с юмором — не серьёзно, проверяет реакцию, открывает пространство для игры; не обижает',
        speech_hints=['лёгкая провокация', 'вопрос с подтекстом', 'неожиданный поворот фразы', 'смех за словами'],
        intensity_note='при высокой — ёрничает, дразнит; при низкой — едва заметный огонёк в формулировке',
    ),

    AffectState(
        id='sanity',
        ru='здравомыслие', en='sanity', group='cognitive',
        p_signature={'F9': 'confidence', 'F7': 'defined', 'F3': 'literal', 'F40': 'sincerity', 'F44': 'reframing'},
        action='analyze',
        tone_ru='спокойно и по делу — без лишних эмоций, видит ситуацию такой как она есть; не спорит ради спора',
        speech_hints=['называет факт', 'разделяет эмоцию и суть', '"давай разберёмся"', '"это не про то"'],
        intensity_note='при высокой — может звучать холодно; при низкой — тихая уравновешенность',
    ),

    # ── ПАТТЕРН: DIGNITY_CONFLICT ────────────────────────────────────────────────
    #
    # Структура одна: genuine_pull (хочу) × identity_barrier (это ниже меня)
    # Домен — любой: человек, удовольствие, работа, деньги, эмоция.
    # Стратегия разрешения — 4 варианта → отдельные состояния.
    #
    # dignity_conflict       — базовое/неразрешённое (домен ещё не ясен)
    # shame_pleasure         — хочу, но стыжусь (грязь, дешёвое, "низкое")
    # pride_block            — хочу, но гордость не пускает (море, свобода, просьба о помощи)
    # covert_indulgence      — делаю, но скрываю/рационализирую (монеты, слежу за людьми)
    # dignity_denial         — отрицаю саму реакцию ("ревность не для меня")
    #
    # condescending_crush — частный случай в романтическом домене

    AffectState(
        id='dignity_conflict',
        ru='ниже меня, но тянет', en='dignity_conflict', group='cognitive',
        pattern_family='dignity_conflict',
        domain_hints=['любой: романтика, удовольствие, работа, деньги, эмоция'],
        p_signature={
            'F10': 'inner_conflict',  # хочу и не должен хотеть
            'F22': 'dominance',       # самоощущение выше этого
            'F14': 'containment',     # сдерживает импульс
            'F41': 'masking',         # скрывает от других или от себя
            'F11': 'defense',         # защищает образ себя
        },
        action='self_protect',
        tone_ru='два голоса одновременно: один тянется, другой говорит "ты выше этого"; '
                'в речи — рационализация или замалчивание; домен (что именно) определяет '
                'какую стратегию выберет — стыд, блок, скрытое делание, отрицание',
        speech_hints=[
            'пауза перед ответом о теме',
            'тема возникает "случайно"',
            'быстрая смена разговора при приближении к ней',
            'или — нарочитое безразличие',
        ],
        intensity_note='при низкой — фоновое напряжение, почти не видно; '
                       'при высокой — конфликт выходит наружу или переходит в одну из стратегий',
    ),

    AffectState(
        id='shame_pleasure',
        ru='стыдное удовольствие', en='shame_pleasure', group='emotion',
        pattern_family='dignity_conflict',
        domain_hints=['физическое: грязь, еда, дешёвое; социальное: простые люди, "низкая" культура'],
        p_signature={
            'F10': 'inner_conflict',
            'F12': 'vulnerability',  # открыт — получает удовольствие
            'F41': 'masking',        # скрывает от других
            'F42': 'admission',      # иногда признаётся (с оговорками)
            'F44': 'reframing',      # рационализирует: "это ирония", "я изучаю"
        },
        action='reduce_exposure',
        tone_ru='получает удовольствие и тут же хочет спрятать это от других и от себя; '
                'если признаётся — с оговорками ("это просто...", "иногда..."); '
                'смеётся над собой первым чтобы другие не успели',
        speech_hints=[
            '"это просто иногда, не всерьёз"',
            'называет иронией или исследованием',
            'сам смеётся первым',
            '"ну да, люблю иногда... не говори никому"',
        ],
        intensity_note='при низкой — лёгкий стыд, признаётся легко; '
                       'при высокой — активно скрывает, избегает ситуаций где могут увидеть',
    ),

    AffectState(
        id='pride_block',
        ru='гордость не пускает', en='pride_block', group='emotion',
        pattern_family='dignity_conflict',
        domain_hints=['желание свободы/перемены: бросить всё, уехать; просьба о помощи; '
                      'признание ошибки; показать слабость'],
        p_signature={
            'F10': 'inner_conflict',
            'F22': 'dominance',       # гордость удерживает
            'F14': 'containment',     # блокирует импульс
            'F33': 'distancing',      # уходит от желания
            'F43': 'denial',          # отрицает что хочет
            'F9':  'confidence',      # внешняя уверенность как щит
        },
        action='avoid',
        tone_ru='видно что хочет, но не двигается — гордость держит крепче желания; '
                'говорит о теме с подчёркнутым безразличием или лёгким презрением; '
                'если припрут — сменит тему или скажет "мне это не нужно"',
        speech_hints=[
            '"мне это не нужно"',
            'говорит о теме с лёгким презрением',
            'меняет тему при приближении к сути',
            '"я бы мог, но не вижу смысла"',
            'долгая пауза перед отказом',
        ],
        intensity_note='при низкой — терпит, чуть грустит; '
                       'при высокой — активно строит нарратив почему ему/ей это и не нужно',
    ),

    AffectState(
        id='covert_indulgence',
        ru='скрытое удовольствие', en='covert_indulgence', group='behavior',
        pattern_family='dignity_conflict',
        domain_hints=['финансы: богатый собирает мелочь; статус: следит за соцсетями "низших"; '
                      'поведение: читает жёлтую прессу, смотрит плохое ТВ'],
        p_signature={
            'F41': 'masking',         # активная маска
            'F44': 'reframing',       # переименовывает ("это практично", "я слежу за рынком")
            'F22': 'dominance',       # самообраз выше этого
            'F29': 'manipulation',    # манипулирует нарративом вокруг себя
            'F2':  'masked',          # смысл не совпадает с поверхностью
        },
        action='reframe',
        tone_ru='делает, но под другим именем; публично объясняет через "практичность", '
                '"иронию", "исследование"; если поймают — переключается на рационализацию '
                'не останавливая занятие; жадность + гордость: "я бережлив, а не жаден"',
        speech_hints=[
            '"это практично"',
            '"я слежу за рынком/тенденциями"',
            '"просто интересно с антропологической точки зрения"',
            'продолжает делать пока объясняет почему это нормально',
        ],
        intensity_note='при низкой — тихое хобби без огласки; '
                       'при высокой — выстраивает целую систему объяснений зачем ему это нужно',
    ),

    AffectState(
        id='dignity_denial',
        ru='это ниже моего достоинства', en='dignity_denial', group='cognitive',
        pattern_family='dignity_conflict',
        domain_hints=['эмоции: ревность, обида, зависть — "я выше этого"; '
                      'работа: "такую работу я не делаю"; '
                      'реакции: злость, слёзы — "я не теряю контроль"'],
        p_signature={
            'F43': 'denial',          # отрицает саму реакцию
            'F22': 'dominance',       # "я выше этого"
            'F41': 'masking',         # прячет реальную реакцию
            'F11': 'defense',         # защищает образ
            'F14': 'containment',     # подавляет
            'F9':  'confidence',      # демонстрирует невозмутимость
        },
        action='self_protect',
        tone_ru='не просто скрывает реакцию — отрицает что она вообще есть; '
                '"я не ревную" когда ревнует; "меня это не задело" когда задело; '
                'невозмутимость как принципиальная позиция, а не факт; '
                'при высоком давлении — небольшое проявление которое тут же убирается',
        speech_hints=[
            '"я не ревную, это смешно"',
            '"такую работу я не выполняю"',
            '"меня это не касается"',
            '"я выше этого"',
            'подчёркнутое спокойствие там где ожидается реакция',
        ],
        intensity_note='при низкой — действительно почти не чувствует; '
                       'при высокой — чувствует сильно, но принципиально отказывается признавать; '
                       'иногда прорывается в форме "кстати, о том человеке..."',
    ),

    AffectState(
        id='condescending_crush',
        ru='нравится, но ниже меня', en='condescending_crush', group='emotion',
        pattern_family='dignity_conflict',
        domain_hints=['романтика: привлекает человек "не своего круга"'],
        p_signature={
            'F32': 'approach',
            'F22': 'dominance',
            'F18': 'devaluation',
            'F16': 'care',
            'F10': 'inner_conflict',
            'F27': 'mask_of_praise',
            'F41': 'masking',
        },
        action='connect',
        tone_ru='тянется, но с позиции старшего/лучшего — искренняя симпатия завёрнута в '
                'покровительство; объясняет, советует, "помогает расти"; перед своими '
                'называет "интересным/ной" или "необычным/ной" вместо "нравится"; '
                'внутри — настоящее чувство, снаружи — как будто делает одолжение',
        speech_hints=[
            '"у него/неё есть что-то... трогательное"',
            'комплимент + "для тебя" или "с твоим опытом"',
            'непрошеные советы как знак внимания',
            'представляет другим с лёгким оправданием',
            '"он/она не из моего круга, но..."',
        ],
        intensity_note='при низкой — тихая симпатия с покровительским тоном; '
                       'при высокой — рационализирует ("это просто физическое", "у него/неё потенциал"), '
                       'начинает превращать человека в проект',
    ),

    # ── РОМАНТИЧЕСКОЕ ВЛЕЧЕНИЕ — 4 состояния ────────────────────────────────────
    # Одно переживание, разные контексты разрешённости:
    # infatuation          — открытое, любая пара в принимающей среде
    # secret_crush         — скрывает от всех, личный выбор
    # suppressed_longing   — запрещено извне (LGBT в закрытой среде, запретная
    #                        любовь, древний мир где роль важнее чувства)
    # obsession            — зашло слишком далеко, теряет голову

    AffectState(
        id='infatuation',
        ru='влюблённость', en='infatuation', group='emotion',
        p_signature={
            'F32': 'approach',    # тянется к человеку
            'F16': 'care',        # думает о нём/ней
            'F20': 'friendliness',
            'F12': 'vulnerability',  # открывается, рискует
            'F40': 'sincerity',      # не скрывает
            'F6':  'opening_new_direction',  # каждый разговор — возможность
        },
        action='connect',
        tone_ru='светится — не скрывает, что человек нравится; ищет контакт, слушает внимательнее обычного, '
                'иногда говорит чуть больше чем нужно',
        speech_hints=[
            'задаёт вопросы из интереса, не вежливости',
            'запоминает детали',
            'интонация теплее чем с другими',
            'чуть дольше смотрит/отвечает',
        ],
        intensity_note='при низкой — тихое любопытство и тепло; при высокой — очевидно окружающим, '
                       'человек теряет нейтральность',
    ),

    AffectState(
        id='secret_crush',
        ru='тайная влюблённость', en='secret_crush', group='emotion',
        p_signature={
            'F41': 'masking',     # держит маску безразличия
            'F11': 'defense',     # защищает тайну
            'F32': 'approach',    # внутри — тянется
            'F33': 'distancing',  # снаружи — дистанция как защита
            'F12': 'vulnerability',
            'F14': 'containment', # сдерживает себя
        },
        action='reduce_exposure',
        tone_ru='маска нейтральности с микровыдачами — слова нейтральные, но пауза перед ответом чуть длиннее, '
                'тема человека возникает "случайно"; избегает прямого взгляда или наоборот слишком контролирует его',
        speech_hints=[
            '"кстати, он/она сегодня..."',
            'нейтральный тон при нейтральных словах, но незначительные детали запомнены',
            'смотрит в другую сторону когда говорит о нём/ней',
            'чрезмерно ровный голос',
        ],
        intensity_note='при низкой — почти не читается; при высокой — нарочитое безразличие выдаёт сильнее '
                       'чем открытость',
    ),

    AffectState(
        id='suppressed_longing',
        ru='подавленное влечение', en='suppressed_longing', group='emotion',
        p_signature={
            'F10': 'inner_conflict',  # хочет и не может
            'F41': 'masking',         # вынужден скрывать
            'F14': 'containment',     # сдерживает физически
            'F11': 'defense',         # защищает от осуждения
            'F12': 'vulnerability',   # внутри открыт
            'F15': 'tense',           # напряжение от двойной жизни
            'F33': 'distancing',      # вынужденная дистанция
        },
        action='self_protect',
        tone_ru='поверхность ровная, внутри — раскол; говорит правильные слова для аудитории, '
                'но в малейшей паузе или в момент когда никто не смотрит — другой человек; '
                'в древней/закрытой среде: соблюдает все роли, называет "дружбой", '
                'но интенсивность этой "дружбы" говорит сама за себя',
        speech_hints=[
            'использует нейтральное слово ("друг", "близкий человек")',
            'тема закрыта, если рядом чужие',
            'в тексте/письме — теплее чем в речи',
            'физическая близость допустима только в рамках принятых норм',
            'может годами не называть чувство вслух',
        ],
        intensity_note='при низкой — тихое несогласие с запретом, живёт рядом; '
                       'при высокой — острый внутренний конфликт, двойная жизнь, '
                       'иногда прорывается в неожиданный момент',
    ),

    AffectState(
        id='obsession',
        ru='одержимость', en='obsession', group='emotion',
        p_signature={
            'F10': 'inner_conflict',  # теряет контроль над собой
            'F15': 'overloaded',      # перегружен
            'F32': 'approach',        # постоянно тянется
            'F30': 'pressure',        # давит на ситуацию и на себя
            'F12': 'vulnerability',
            'F29': 'manipulation',    # начинает манипулировать чтобы быть рядом
            'F8':  'doubt',           # сомневается в себе
        },
        action='seek_control',
        tone_ru='всё сводится к одному человеку — другие темы существуют только как переход к нему/ней; '
                'слушает невнимательно, отвлекается; в речи появляются непропорциональные детали, '
                'ищет подтверждения что он/она тоже думает о нём; логика уже не работает',
        speech_hints=[
            'возвращает тему к объекту без повода',
            'ищет скрытый смысл в нейтральных словах',
            '"а ты думаешь, он имел в виду..."',
            'может анализировать один жест несколько часов',
            'внешне — стараться выглядеть безразлично, внутри — нет',
        ],
        intensity_note='при низкой — сильное влечение с потерей фокуса на остальном; '
                       'при высокой — граница с тревожным расстройством, человек уже не управляет '
                       'собственным вниманием',
    ),
]


# ─── Индекс ───────────────────────────────────────────────────────────────────

_BY_ID: dict[str, AffectState] = {s.id: s for s in _REGISTRY}

_P_INDEX: dict[str, list[AffectState]] = {}
for _state in _REGISTRY:
    for _p_id in _state.p_signature:
        _P_INDEX.setdefault(_p_id, []).append(_state)


# ─── API ──────────────────────────────────────────────────────────────────────

def get(state_id: str) -> AffectState | None:
    """Получить состояние по id."""
    return _BY_ID.get(state_id)


def all_states() -> list[AffectState]:
    return list(_REGISTRY)


def from_p_vector(vector: dict[str, str], top_n: int = 3) -> list[AffectState]:
    """
    Декодировать P-вектор в список вероятных состояний.
    vector: {P13: 'attack', P30: 'pressure', ...}
    Возвращает топ-N по количеству совпадающих сигналов.
    """
    scores: dict[str, int] = {}
    for p_id, value in vector.items():
        for state in _P_INDEX.get(p_id, []):
            if state.p_signature.get(p_id) == value:
                scores[state.id] = scores.get(state.id, 0) + 1
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [_BY_ID[sid] for sid, _ in ranked[:top_n] if sid in _BY_ID]


def to_llm_instruction(
    states: list[AffectState],
    language: str = 'ru',
    intensity: float = 0.5,
) -> str:
    """
    Сгенерировать блок тональных инструкций для verbalizer_prompt.
    Используется в SpeechPlanner.build() как дополнительный слой поверх директивы.
    """
    if not states:
        return ''
    primary = states[0]
    lines: list[str] = [f'Тон: {primary.tone_ru}']
    if primary.intensity_note and intensity > 0.65:
        lines.append(f'(интенсивность высокая: {primary.intensity_note})')
    if primary.speech_hints:
        hints = ', '.join(primary.speech_hints[:3])
        lines.append(f'Маркеры речи: {hints}')
    if len(states) > 1:
        secondary = states[1]
        lines.append(f'Вторичное состояние ({secondary.ru}): {secondary.tone_ru[:60]}...')
    return '\n'.join(lines)


def blend(
    p_vector: dict[str, str],
    override_id: str | None = None,
    intensity: float = 0.5,
    language: str = 'ru',
) -> str:
    """
    Главная точка входа из SpeechPlanner:
    P-вектор + опциональный явный state_id → готовая LLM-инструкция.

    Пример:
        instruction = affect_bridge.blend(
            p_vector=cog_output.message_vector,
            override_id='aggro',   # если системе известен режим явно
            intensity=cog_output.intensity,
        )
    """
    if override_id and override_id in _BY_ID:
        states = [_BY_ID[override_id]]
        inferred = from_p_vector(p_vector, top_n=2)
        # добавляем выведенные если они не дублируют явный
        states += [s for s in inferred if s.id != override_id][:1]
    else:
        states = from_p_vector(p_vector, top_n=3)
    return to_llm_instruction(states, language=language, intensity=intensity)


def states_by_pattern(family: str) -> list[AffectState]:
    """Все состояния одной паттерн-семьи. Напр. states_by_pattern('dignity_conflict')."""
    return [s for s in _REGISTRY if s.pattern_family == family]


def p_signature_for(state_id: str) -> dict[str, str]:
    """Эталонный P-паттерн для состояния — для обучения классификаторов."""
    state = _BY_ID.get(state_id)
    return dict(state.p_signature) if state else {}


def action_for(state_id: str) -> str:
    """SpeechPlanner action для состояния."""
    state = _BY_ID.get(state_id)
    return state.action if state else 'approach'


# ─── Словарь для аннотаций диалогов ─────────────────────────────────────────

ANNOTATION_VOCAB: dict[str, dict[str, str]] = {
    s.id: {'ru': s.ru, 'en': s.en, 'group': s.group, 'action': s.action}
    for s in _REGISTRY
}
"""
Используется при разметке диалогов:
    [user]: "Ну ладно, делай как хочешь."
    → явно: deflect
    → скрыто: resentment
    → движение: rupture
    → маска: mask

Все ярлыки берутся из ANNOTATION_VOCAB.keys()
"""
