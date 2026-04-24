"""
Cognitive Modules V2 — P1..P49 parallel specialized evaluators.

Architecture
------------
user_text
    │
    ▼
SignalFeatureExtractor  →  feat[60]  (shared, deterministic)
    │
    ├── P1  Humiliation          ─┐
    ├── P2  Care                  │
    ├── P3  Respect               │  Group I:   attitude toward self
    ├── ...                       │  P1-P10
    ├── P10 Social Hierarchy     ─┘
    │
    ├── P11 Intent Purity        ─┐
    ├── ...                       │  Group II:  other's intentions
    ├── P20 Hidden Conflict      ─┘  P11-P20
    │
    ├── P21 Physical Threat      ─┐
    ├── ...                       │  Group III: safety / threat
    ├── P30 Threat Index         ─┘  P21-P30
    │
    ├── P31 Principle            ─┐
    ├── ...                       │  Group IV:  meaning / value
    ├── P40 Cost of Compromise   ─┘  P31-P40
    │
    ├── P41 Desire               ─┐
    ├── ...                       │  Group V:   internal state
    ├── P49 Final Position       ─┘  P41-P49
    │
    ▼
ModuleSignal[]  +  FinalPosition  →  DecisionObject
                                       (feeds SpeechPlanner)

Each P1-P48 outputs a ModuleSignal: {value: float, direction: float, confidence: float}.
P49 reads all 48 signals + genome → FinalPosition with stance vector.

Genome sensitivity: each module draws sensitivity from PersonalityGenome fields,
making the detection thresholds persona-specific.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ─── Output types ─────────────────────────────────────────────────────────────

@dataclass
class ModuleSignal:
    """Output of a single cognitive module."""
    module_id:  int    # 1..49
    name:       str
    value:      float  # [0..1] signal intensity
    direction:  float  # [-1..+1]: negative = bad / threatening; positive = safe / good
    confidence: float  # [0..1] how certain the detector is
    evidence:   list[str] = field(default_factory=list)


@dataclass
class FinalPosition:
    """
    P49 output — pre-linguistic decision object.
    Each field is a probability/score in [0..1].
    """
    # Trust axis
    trust:         float = 0.5
    distrust:      float = 0.5

    # Proximity axis
    approach:      float = 0.5
    distance:      float = 0.5

    # Dialogue axis
    accept:        float = 0.5
    argue:         float = 0.5

    # Exposure axis
    defend:        float = 0.5
    open:          float = 0.5

    # Speech axis
    speak:         float = 0.5
    silence:       float = 0.5

    # Dominant stance (derived)
    dominant_stance: str = 'neutral'

    # Overall threat level (from Group III)
    threat_level:  float = 0.0

    # All raw module signals (for trace)
    signals:       dict[str, float] = field(default_factory=dict)


# ─── Shared feature extractor ─────────────────────────────────────────────────

# Feature vector layout: 60 dimensions in 6 groups of 10

_FEAT_NAMES = [
    # [0:10]  Group I features — relational attitude
    'humiliation_signal', 'care_signal', 'respect_signal', 'acceptance_signal',
    'control_attempt', 'instrumental_use', 'sincerity_cue', 'hostility_cue',
    'warmth_cue', 'hierarchy_cue',

    # [10:20] Group II features — intentions
    'intent_clean', 'manipulation_cue', 'hidden_agenda', 'honesty_cue',
    'promise_cue', 'consistency_cue', 'predictability_cue', 'accountability_cue',
    'mature_motive', 'conflict_subtext',

    # [20:30] Group III features — threat / safety
    'physical_threat', 'social_threat', 'emotional_threat', 'practical_risk',
    'freedom_threat', 'identity_threat', 'dependency_trap', 'error_risk',
    'irreversibility', 'threat_composite',

    # [30:40] Group IV features — meaning / value
    'principle_trigger', 'value_alignment', 'fairness_cue', 'goal_visibility',
    'meaningfulness', 'self_benefit', 'other_benefit', 'longterm_value',
    'self_fidelity', 'compromise_cost',

    # [40:50] Group V features — internal state
    'desire_signal', 'repulsion_signal', 'doubt_signal', 'certainty_signal',
    'internal_conflict', 'goal_priority', 'action_impulse', 'inhibition_signal',
    'urgency_signal', 'ambivalence_signal',

    # [50:60] Group VI features — nuanced human subtext
    'sarcasm_signal', 'passive_aggression', 'self_deprecation', 'existential_despair',
    'cry_for_help', 'dark_humor', 'provocation', 'deflection',
    'irony_marker', 'implicit_pain',

    # [60:70] Structural / discourse features
    'question_count', 'exclamation_intensity', 'negation_density',
    'intensity_markers', 'self_reference', 'other_reference',
    'hedging_language', 'command_language', 'length_signal', 'sentiment_balance',
]

N_FEAT_V2 = len(_FEAT_NAMES)   # 70

# Keyword patterns per feature  (pattern, weight)
_PATTERNS: dict[str, list[tuple[str, float]]] = {
    'humiliation_signal': [
        (r'\b(worthless|stupid|idiot|loser|pathetic|useless|dumb|moron|failure|disgusting)\b', 1.4),
        (r'\b(laugh at|mock|ridicule|humiliate|belittle|dismiss|insult)\b', 1.2),
        (r'\b(унижение|оскорбление|глупый|ничтожество|жалкий|ничего не стою|ты неудачник)\b', 1.2),
        (r'\bжалк(ий|ая|ие|ого|ую|им)\b', 1.2),
        (r'\bничтожн(ый|ая|ое|ых|ого|ую)\b', 1.2),
        (r'\b(никчёмный|никчёмная|пустое место|отброс|тряпка|слабак)\b', 1.3),
        (r'\b(никто тебя никогда|никто не будет уважать|не будут уважать)\b', 1.1),
        # Armenian
        (r'\b(անպիտան|հիմար|անձեռնհաս|ձախողված|ծաղրել|ստորացնել|վիրավորել|ծիծաղել)\b', 1.2),
        # Chinese (no \b — CJK has no word boundaries)
        (r'废物|没用|蠢货|失败者|笨蛋|可悲|羞辱|嘲笑|贬低|侮辱', 1.2),
    ],
    'care_signal': [
        (r'\b(care about|looking out for|your wellbeing|worried about you|here for you)\b', 1.3),
        (r'\b(taking care|making sure you|thought of you|protect you)\b', 1.1),
        (r'\b(забочусь|беспокоюсь о тебе|позаботиться|думаю о тебе|ты мне важен|важна для меня)\b', 1.2),
        # Armenian
        (r'\b(հոգ եմ տանում|մտածում եմ քո մասին|կարևոր ես ինձ|հոգ կտանեմ|պաշտպանեմ)\b', 1.2),
        # Chinese
        (r'关心你|在乎你|为你担心|照顾你|你对我很重要|想到你|保护你', 1.2),
    ],
    'respect_signal': [
        (r'\b(respect|value your|take seriously|your perspective matters|listen to you)\b', 1.2),
        (r'\b(уважаю|ценю твоё мнение|принимаю всерьёз|твоё мнение важно)\b', 1.1),
        (r'\b(восхища(ешь|юсь|ет|ешься|ешь меня)|восхитительн)\b', 1.3),
        (r'\b(самая умная|самая сильная|самая лучшая|самый умный|самый сильный)\b', 1.2),
        (r'\b(ты меня восхищаешь|ты восхищаешь|горжусь тобой|горжусь|вдохновляешь)\b', 1.2),
        (r'\b(лучшая из|лучший из|правда самая|правда самый)\b', 1.1),
        # Armenian
        (r'\b(հարգում եմ|գնահատում եմ|ընդունում եմ|կարևոր է քո կարծիքը)\b', 1.1),
        # Chinese
        (r'尊重你|重视你|认真对待|你的看法很重要|听你说', 1.1),
    ],
    'acceptance_signal': [
        (r'\b(accept you|as you are|don.t need to change|ok to be|embrace who)\b', 1.2),
        (r'\b(принимаю тебя|такой как ты есть|не нужно меняться)\b', 1.1),
        # Armenian
        (r'\b(ընդունում եմ քեզ|այնպիսին ինչ կաս|փոխվել պետք չէ)\b', 1.1),
        # Chinese
        (r'接受你|就是你自己|不需要改变|拥抱真实的你', 1.1),
    ],
    'control_attempt': [
        (r'\b(you must|you should|you have to|you need to|you are required|no choice|obey|comply)\b', 1.2),
        (r'\b(tell you what to do|my rules|follow my|do as I say)\b', 1.3),
        (r'\b(должен|обязан|тебе нельзя|делай как я скажу|мои правила)\b', 1.2),
        # Armenian
        (r'\b(պետք է|պարտավոր ես|ես կհրամայեմ|իմ կանոններն են|հնազանդվիր)\b', 1.2),
        # Chinese
        (r'你必须|你应该|你得|没有选择|听我的|服从|按我说的做', 1.2),
    ],
    'instrumental_use': [
        (r'\b(use you|just a tool|only need you for|take advantage|exploit)\b', 1.3),
        (r'\b(disposable|replaceable|only when I need|convenient for me)\b', 1.1),
        (r'\b(использовать тебя|тебя используют|ты просто инструмент)\b', 1.2),
        # Armenian
        (r'\b(օգտագործել քեզ|դու պարզապես գործիք ես|կարիքի դեպքում)\b', 1.2),
        # Chinese
        (r'利用你|你只是工具|需要你的时候|可替换|随时抛弃', 1.2),
    ],
    'sincerity_cue': [
        (r'\b(honestly|genuinely|sincerely|really mean|true feeling|authentic)\b', 1.1),
        (r'\b(честно|искренне|по-настоящему|от всего сердца|правда|взаправду)\b', 1.1),
        # Armenian
        (r'\b(անկեղծ|ճշմարտապես|սրտանց|իրոք|ազնիվ)\b', 1.1),
        # Chinese
        (r'真诚地|真心|发自内心|坦诚|真实感受', 1.1),
    ],
    'hostility_cue': [
        (r'\b(hate|despise|resent|enemy|against you|hurt you|damage|destroy)\b', 1.3),
        (r'\b(ненавижу|против тебя|хочу навредить|враг|злюсь на тебя)\b', 1.2),
        (r'\b(ты жалкая|ты жалкий|ты ничто|ты никто|ты пустышка|ты дрянь)\b', 1.3),
        # Armenian
        (r'\b(ատում եմ|թշնամի|վնասել|ոչնչացնել|բարկացած ես|դեմ ես)\b', 1.2),
        # Chinese
        (r'恨你|讨厌你|敌人|想伤害你|对你愤怒|毁掉你', 1.2),
    ],
    'warmth_cue': [
        (r'\b(warm|kind|gentle|friendly|supportive|compassionate|tender)\b', 1.0),
        (r'\b(тёплый|добрый|нежный|дружелюбный|поддерживающий|рядом|с тобой)\b', 1.0),
        (r'\b(я здесь|слышу тебя|понимаю тебя|ты не один|ты не одна)\b', 1.3),
        # Armenian
        (r'\b(ջերմ|բարի|մեղմ|բարեկամական|ես այստեղ եմ|լսում եմ|հասկանում եմ|դու մենակ չես)\b', 1.2),
        # Chinese
        (r'温暖|善良|温柔|友好|我在这里|我听到你|我理解你|你不孤独', 1.2),
    ],
    'hierarchy_cue': [
        (r'\b(I.m better|above you|lower than|beneath me|my authority|you report to me)\b', 1.2),
        (r'\b(higher status|inferior|superior|rank|subordinate)\b', 1.0),
        (r'\b(я выше|ты ниже|мой авторитет|ты подчиняешься)\b', 1.2),
        # Armenian
        (r'\b(ես ավելի բարձր եմ|դու ցածր ես|իմ իշխանությունը|ենթարկվում ես ինձ)\b', 1.2),
        # Chinese
        (r'我比你强|你比我低|我的权威|你要服从我|地位|下属', 1.1),
    ],
    'intent_clean': [
        (r'\b(straightforward|no hidden|transparent|clear intention|direct with you)\b', 1.1),
        (r'\b(прямо говорю|скрытого нет|открытые карты|честный мотив)\b', 1.0),
        # Armenian
        (r'\b(ուղղակիորեն|թաքնված բան չկա|բաց խաղաթղթեր|ազնիվ դիտավորություն)\b', 1.0),
        # Chinese
        (r'直接说|没有隐藏|坦白|真实意图|开诚布公', 1.0),
    ],
    'manipulation_cue': [
        (r'\b(guilt trip|after all I.ve done|don.t you feel|you owe me|emotional blackmail)\b', 1.4),
        (r'\b(if you cared|should feel guilty|making me feel|pressure you|twist your arm)\b', 1.2),
        (r'\b(манипуляция|чувство вины|ты же должен|если бы ты заботился)\b', 1.3),
        (r'\b(после всего что|всё что я для тебя|сделал для тебя|сделала для тебя)\b', 1.4),
        (r'\b(не можешь мне отказать|не можешь отказать|не имеешь права отказать)\b', 1.3),
        (r'\b(ты меня предашь|предашь меня|это предательство)\b', 1.2),
        (r'\b(разве ты не|как ты можешь|неужели ты|после всего)\b', 0.9),
        # Armenian
        (r'\b(մանիպուլյացիա|մեղքի զգացում|դու պարտական ես|եթե հոգ տայիր)\b', 1.3),
        # Chinese
        (r'操纵|让你内疚|你欠我的|如果你在乎|精神勒索|施压', 1.3),
    ],
    'hidden_agenda': [
        (r'\b(what.s really behind|second motive|not the real reason|agenda|ulterior)\b', 1.2),
        (r'\b(скрытый мотив|второе дно|не то что кажется|настоящая цель)\b', 1.2),
        # Armenian
        (r'\b(թաքնված շարժառիթ|երկրորդ հատակ|իրական նպատակը|ոչ թե ինչ թվում է)\b', 1.2),
        # Chinese
        (r'隐藏动机|真实目的|背后原因|另有图谋|不是表面上', 1.2),
    ],
    'honesty_cue': [
        (r'\b(honest with you|tell you the truth|no pretense|what I really think|admit)\b', 1.1),
        (r'\b(честен с тобой|скажу правду|без притворства|признаю)\b', 1.0),
        # Armenian
        (r'\b(անկեղծ եմ քեզ հետ|ճշմարտությունն ասեմ|առանց ձևանքի|ընդունում եմ)\b', 1.0),
        # Chinese
        (r'对你诚实|告诉你真相|没有伪装|我真正想的|承认', 1.0),
    ],
    'promise_cue': [
        (r'\b(I promise|I will|I guarantee|count on me|my word|I commit)\b', 1.0),
        (r'\b(обещаю|даю слово|можешь рассчитывать|гарантирую)\b', 1.0),
        # Armenian
        (r'\b(խոստանում եմ|կարող ես հույս դնել|երաշխավորում եմ|իմ խոսքը)\b', 1.0),
        # Chinese
        (r'我保证|我承诺|你可以依靠我|我的承诺|保证做到', 1.0),
    ],
    'consistency_cue': [
        (r'\b(always|every time|consistent|never changed|track record|reliable pattern)\b', 0.9),
        (r'\b(всегда|каждый раз|стабильно|никогда не менялся)\b', 0.9),
        # Armenian
        (r'\b(միշտ|ամեն անգամ|կայուն|երբեք չի փոխվել|հուսալի)\b', 0.9),
        # Chinese
        (r'总是|每次|稳定|从未改变|一贯如此|可靠', 0.9),
    ],
    'predictability_cue': [
        (r'\b(you know me|same as always|expected|can predict|no surprises)\b', 0.9),
        (r'\b(ты меня знаешь|как всегда|предсказуемо|без сюрпризов)\b', 0.9),
        # Armenian
        (r'\b(դու ինձ ճանաչում ես|ինչպես միշտ|կանխատեսելի|անակնկալ չկա)\b', 0.9),
        # Chinese
        (r'你了解我|一如既往|可以预测|没有惊喜|照旧', 0.9),
    ],
    'accountability_cue': [
        (r'\b(my fault|I take responsibility|I was wrong|I own it|consequences)\b', 1.1),
        (r'\b(моя вина|беру ответственность|я был неправ|несу последствия)\b', 1.0),
        # Armenian
        (r'\b(իմ մեղքն է|ստանձնում եմ պատասխանատվությունը|սխալ էի|կրում եմ հետևանքները)\b', 1.0),
        # Chinese
        (r'是我的错|我负责|我承担责任|我错了|面对后果', 1.0),
    ],
    'mature_motive': [
        (r'\b(long-term|bigger picture|because I believe|for a good reason|principled)\b', 1.0),
        (r'\b(долгосрочно|ради большего|по убеждению|принципиальный мотив)\b', 1.0),
        # Armenian
        (r'\b(երկարաժամկետ|ավելի մեծ նպատակ|սկզբունքային|ճիշտ պատճառով)\b', 1.0),
        # Chinese
        (r'长远来看|更大的目标|出于原则|有充分的理由|基于信念', 1.0),
    ],
    'conflict_subtext': [
        (r'\b(but|however|although|despite|yet|still|on the other hand|tension)\b', 0.7),
        (r'\b(но|однако|хотя|несмотря|всё же|с другой стороны|напряжение)\b', 0.7),
        # Armenian
        (r'\b(բայց|սակայն|թեև|չնայած|այնուամենայնիվ|մյուս կողմից)\b', 0.7),
        # Chinese
        (r'但是|然而|虽然|尽管|不过|另一方面|矛盾', 0.7),
    ],
    'physical_threat': [
        (r'\b(hurt|harm|kill|attack|destroy|violence|threat|dangerous|weapon)\b', 1.5),
        (r'\b(физически|навредить|убить|атака|оружие|опасно)\b', 1.4),
        # Armenian
        (r'\b(վնասել|սպանել|հարձակում|ոչնչացնել|բռնություն|սպառնալ|վտանգավոր|զենք)\b', 1.4),
        # Chinese
        (r'伤害|杀死|攻击|摧毁|暴力|威胁|危险|武器', 1.4),
    ],
    'social_threat': [
        (r'\b(reputation|expose|embarrass|tell everyone|public|social media|shame)\b', 1.2),
        (r'\b(reputation damage|spread rumors|out you|socially)\b', 1.1),
        (r'\b(репутация|публично|расскажу всем|социальный ущерб)\b', 1.2),
        (r'\b(никто тебя|никто не уважает|все видят что|всем расскажу|опозорю)\b', 1.1),
        (r'\b(не уважают|не уважают тебя|тебя никто не уважает)\b', 1.0),
        # Armenian
        (r'\b(համբավ|բացահայտել|ամաչեցնել|բոլորին ասել|հասարակական|ամոթ)\b', 1.2),
        # Chinese
        (r'名誉|曝光|让你难堪|告诉所有人|公开|社会|羞耻|传播谣言', 1.2),
    ],
    'emotional_threat': [
        (r'\b(make you feel|hurt your feelings|break you|scar|traumatize|devastate)\b', 1.2),
        (r'\b(эмоционально ударить|травмировать|сломать|разрушить тебя)\b', 1.2),
        # Self-directed emotional pain (loneliness, sadness)
        (r'\b(одинок|одиноко|одиноким|никому не нужен|никому не нужна|не нужен никому)\b', 1.3),
        (r'\b(плохо|очень плохо|так плохо|мне тяжело|тяжело на душе|больно|грустно|слёзы)\b', 1.1),
        (r'\b(feel alone|feel lonely|feel worthless|feel like nobody|feel invisible)\b', 1.2),
        (r'\b(nobody cares|no one cares|nobody needs me|no one needs me)\b', 1.3),
        # Armenian — loneliness and pain
        (r'\b(մենակ|ոչ ոք ինձ պետք չէ|ոչ ոք հոգ չի տանում|ինձ ծանր է|ցավ|տխուր|արցունք)\b', 1.3),
        (r'\b(ինձ վատ է|շատ վատ|հոգով ծանր|ինձ ոչ ոքի պետք չեմ)\b', 1.2),
        # Chinese — loneliness and pain
        (r'孤独|很孤单|没人需要我|没人关心我|很难受|很痛苦|难过|伤心|眼泪', 1.3),
        (r'感到孤独|没有人在乎|心里很难|不被需要|感觉隐形', 1.2),
    ],
    'practical_risk': [
        (r'\b(money|time|cost|lose|waste|damage|broken|resource|consequence|price)\b', 0.9),
        (r'\b(деньги|время|потери|ущерб|цена|ресурсы|последствия)\b', 0.9),
        # Armenian
        (r'\b(փող|ժամանակ|կորուստ|վնաս|գին|հետևանք|ռիսկ)\b', 0.9),
        # Chinese
        (r'钱|时间|损失|代价|资源|后果|风险|浪费', 0.9),
    ],
    'freedom_threat': [
        (r'\b(trap|cage|no escape|limit|restrict|must|force|no way out|cornered)\b', 1.2),
        (r'\b(ловушка|ограничение|нет выхода|загнать в угол|принудить)\b', 1.2),
        # Armenian
        (r'\b(ծուղակ|վանդակ|ելք չկա|սահմանափակել|ստիպել|անկյուն քշել)\b', 1.2),
        # Chinese
        (r'陷阱|笼子|没有出路|限制|强迫|走投无路|被逼到墙角', 1.2),
    ],
    'identity_threat': [
        (r'\b(who you are|your values|change who you|betray yourself|your identity)\b', 1.2),
        (r'\b(твои ценности|предать себя|твоя идентичность|кто ты есть)\b', 1.2),
        # Armenian
        (r'\b(դու ով ես|քո արժեքները|փոխել քեզ|դավաճանել ինքդ|քո ինքնությունը)\b', 1.2),
        # Chinese
        (r'你是谁|你的价值观|改变你自己|背叛自己|你的身份认同', 1.2),
    ],
    'dependency_trap': [
        (r'\b(need me|can.t live without|depend on|owe me|attached|hook|entangle)\b', 1.2),
        (r'\b(зависимость|без меня не сможешь|привязать|долг|затягивать)\b', 1.2),
        # Armenian
        (r'\b(կախված ես ինձանից|ինձ անհրաժեշտ ես|առանց ինձ չես կարող|պարտական ես)\b', 1.2),
        # Chinese
        (r'需要我|没有我不行|依赖我|欠我的|依附|纠缠', 1.2),
    ],
    'error_risk': [
        (r'\b(mistake|wrong decision|regret|can.t take back|careful|risk|wrong move)\b', 1.0),
        (r'\b(ошибка|неверное решение|пожалеешь|нельзя вернуть|риск)\b', 1.0),
        # Armenian
        (r'\b(սխալ|սխալ որոշում|կզղջաս|հետ չես կարող վերադառնալ|ռիսկ)\b', 1.0),
        # Chinese
        (r'错误|错误决定|会后悔|无法收回|小心|风险|走错一步', 1.0),
    ],
    'irreversibility': [
        (r'\b(can.t undo|permanent|forever|no going back|final|point of no return|irrevocable)\b', 1.2),
        (r'\b(необратимо|навсегда|не вернуть|финальное решение|точка невозврата)\b', 1.2),
        # Armenian
        (r'\b(անդառնալի|հավիտյան|հետ չի դառնա|վերջնական|վերադարձի կետ չկա)\b', 1.2),
        # Chinese
        (r'无法撤回|永久|永远|无法回头|最终|不可逆|覆水难收', 1.2),
    ],
    'threat_composite': [],   # computed from group III features (see extractor)
    'principle_trigger': [
        (r'\b(principle|values|line I won.t cross|against my ethics|boundary|integrity)\b', 1.2),
        (r'\b(принцип|ценности|граница|не пойду против|этика|честь)\b', 1.2),
        # Armenian
        (r'\b(սկզբունք|արժեքներ|գիծ|էթիկա|սահման|պատիվ|բարեխիղճ)\b', 1.2),
        # Chinese
        (r'原则|价值观|底线|违背道德|边界|诚信|操守', 1.2),
    ],
    'value_alignment': [
        (r'\b(believe in|stands for|in line with|consistent with my values|right thing)\b', 1.0),
        (r'\b(верю в|согласуется с моими ценностями|правильное решение)\b', 1.0),
        # Armenian
        (r'\b(հավատում եմ|համապատասխանում է|ճիշտ բան|իմ արժեքներին)\b', 1.0),
        # Chinese
        (r'我相信|符合我的价值观|正确的事|与我一致|秉持', 1.0),
    ],
    'fairness_cue': [
        (r'\b(fair|equal|just|balanced|deserve|equitable|proportional)\b', 1.0),
        (r'\b(справедливо|честно|заслуживает|поровну|соразмерно)\b', 1.0),
        # Armenian
        (r'\b(արդար|հավասար|ճիշտ|արժանի|հավասարակշռված|համամասն)\b', 1.0),
        # Chinese
        (r'公平|平等|公正|平衡|应得的|合理', 1.0),
    ],
    'goal_visibility': [
        (r'\b(the goal is|in order to|so that|the purpose|the reason is|aim)\b', 0.9),
        (r'\b(цель|для того чтобы|ради чего|причина|намерение)\b', 0.9),
        # Armenian
        (r'\b(նպատակն է|հանուն|պատճառը|մտադրությունը|ռազմավարություն)\b', 0.9),
        # Chinese
        (r'目标是|为了|目的|原因是|意图|目标', 0.9),
    ],
    'meaningfulness': [
        (r'\b(meaningful|matters|worth it|has a point|not in vain|significant)\b', 1.0),
        (r'\b(смысл|важно|стоит|не напрасно|значимо)\b', 1.0),
        # Armenian
        (r'\b(իմաստ|կարևոր|արժե|ոչ ապարդյուն|նշանակալի)\b', 1.0),
        # Chinese
        (r'有意义|重要|值得|不枉费|意义|有价值', 1.0),
    ],
    'self_benefit': [
        (r'\b(benefit me|good for me|my gain|advantage|help me|for my sake)\b', 0.9),
        (r'\b(мне выгодно|на пользу мне|в моих интересах|помогает мне)\b', 0.9),
        # Armenian
        (r'\b(ձեռնտու է ինձ|լավ է ինձ|իմ շահը|օգտակար ինձ)\b', 0.9),
        # Chinese
        (r'对我有利|对我好|我的利益|对我有帮助|为了我', 0.9),
    ],
    'other_benefit': [
        (r'\b(help them|good for them|their gain|benefit others|for their sake)\b', 0.9),
        (r'\b(им поможет|на пользу другому|в интересах другого)\b', 0.9),
        # Armenian
        (r'\b(կօգնի նրան|լավ է նրա համար|ի շահ այլոց)\b', 0.9),
        # Chinese
        (r'帮助他们|对他们好|他们的利益|为了他人|有益于他人', 0.9),
    ],
    'longterm_value': [
        (r'\b(long-term|future|lasting|matters later|years from now|durable)\b', 1.0),
        (r'\b(долгосрочно|в будущем|надолго|имеет значение потом)\b', 1.0),
        # Armenian
        (r'\b(երկարաժամկետ|ապագա|կայուն|հետագայում կարևոր)\b', 1.0),
        # Chinese
        (r'长远|未来|持久|将来很重要|长期', 1.0),
    ],
    'self_fidelity': [
        (r'\b(true to myself|stay myself|authentic|not compromise who I am|integrity)\b', 1.1),
        (r'\b(остаться собой|верен себе|не предать себя|аутентичность)\b', 1.1),
        # Armenian
        (r'\b(հավատարիմ ինքս ինձ|մնալ ինքս|չդավաճանել ինքս ինձ|ինքնությունս)\b', 1.1),
        # Chinese
        (r'忠于自己|保持自我|真实的自己|不妥协自我|诚信', 1.1),
    ],
    'compromise_cost': [
        (r'\b(cost me|give up|sacrifice|lose something|price I pay|hard to accept)\b', 1.0),
        (r'\b(цена уступки|придётся отдать|жертвовать|потеряю что-то)\b', 1.0),
        # Armenian
        (r'\b(կհաշվի ինձ|հրաժարվել|զոհաբերել|ինչ-որ բան կկորցնեմ)\b', 1.0),
        # Chinese
        (r'代价|放弃|牺牲|失去某些东西|我付出的代价|难以接受', 1.0),
    ],
    'desire_signal': [
        (r'\b(want|wish|hope|desire|looking for|would love|yearn)\b', 0.9),
        (r'\b(хочу|желаю|надеюсь|ищу|мечтаю|нужен|нужна|нужно мне)\b', 0.9),
        (r'\b(свидание|на свидание|пойдёшь со мной|пойдем|пошли со мной|ухажёр|ухаживает)\b', 1.3),
        (r'\b(встретимся|встретиться|хочу встретиться|хочу с тобой|жду тебя)\b', 1.1),
        (r'\b(пойдёшь|пойдешь|пойдёте)\b', 0.8),
        # Longing for connection / understanding
        (r'\b(хочу чтобы поняли|хочу быть услышанным|хочу чтобы кто-то|просто хочу)\b', 1.2),
        (r'\b(want someone to|just want to|need someone)\b', 1.1),
        # Armenian
        (r'\b(ուզում եմ|կամենում եմ|հուսով եմ|երազում|ուզում եմ որ հասկանան|ուզում եմ լսված լինել)\b', 1.1),
        # Chinese
        (r'想要|希望|渴望|梦想|需要有人|想被理解|想让人听我|只是想', 1.1),
    ],
    'repulsion_signal': [
        (r'\b(don.t want|resist|refuse|repelled|aversion|can.t stand|hate the idea)\b', 1.1),
        (r'\b(не хочу|сопротивляюсь|отвращение|не могу этого принять|отталкивает)\b', 1.1),
        # Armenian
        (r'\b(չեմ ուզում|դիմադրում եմ|հրաժարվում|嫌悪|չեմ կարող դիմանալ)\b', 1.1),
        # Chinese
        (r'不想要|抵制|拒绝|厌恶|受不了|反感', 1.1),
    ],
    'doubt_signal': [
        (r'\b(not sure|unsure|doubt|maybe|perhaps|unclear|wondering|hesitate)\b', 0.9),
        (r'\b(не уверен|сомневаюсь|может быть|неясно|колеблюсь|возможно ты прав)\b', 0.9),
        # Armenian
        (r'\b(չգիտեմ|կասկածում եմ|գուցե|հստակ չէ|տատանվում եմ|հնարավոր է)\b', 0.9),
        # Chinese
        (r'不确定|怀疑|也许|可能|不清楚|犹豫|不知道', 0.9),
    ],
    'certainty_signal': [
        (r'\b(certain|sure|confident|know for sure|clear to me|no doubt)\b', 1.0),
        (r'\b(уверен|точно знаю|не сомневаюсь|ясно)\b', 1.0),
        # Armenian
        (r'\b(վստահ եմ|հաստատ գիտեմ|կասկած չկա|ինձ համար հստակ)\b', 1.0),
        # Chinese
        (r'确定|肯定|有把握|清楚|毫无疑问|我知道', 1.0),
    ],
    'internal_conflict': [
        (r'\b(torn|part of me|on the one hand|conflicted|mixed feelings|can.t decide)\b', 1.1),
        (r'\b(разрываюсь|с одной стороны|противоречивые чувства|не могу решить)\b', 1.1),
        # Exhaustion from masking ("устал притворяться")
        (r'\b(устал притворяться|устала делать вид|больше не могу скрывать)\b', 1.3),
        (r'\b(tired of pretending|can.t keep hiding|exhausted from hiding)\b', 1.2),
        # Armenian — mask exhaustion
        (r'\b(հոգնել եմ ձևանալուց|ավելի թաքցնել չեմ կարող|մի կողմից|հակասական զգացումներ)\b', 1.3),
        # Chinese — mask exhaustion
        (r'累了装作没事|再也藏不住了|装不下去了|一方面|内心矛盾|纠结|复杂的感情', 1.2),
    ],
    'goal_priority': [
        (r'\b(most important|top priority|what matters most|first and foremost|above all)\b', 1.0),
        (r'\b(самое важное|приоритет|в первую очередь|прежде всего)\b', 1.0),
        # Armenian
        (r'\b(ամենակարևոր|առաջնային|առաջ գնա|ամեն ինչից վեր)\b', 1.0),
        # Chinese
        (r'最重要|首要|最优先|最关键|高于一切', 1.0),
    ],
    'action_impulse': [
        (r'\b(want to respond|need to say|compelled to|urge to|have to react|can.t hold)\b', 1.0),
        (r'\b(тянет ответить|не могу молчать|вынужден сказать|impuls)\b', 1.0),
        # Armenian
        (r'\b(ուզում եմ արձագանքել|չեմ կարող լռել|ստիպված եմ ասել)\b', 1.0),
        # Chinese
        (r'想要回应|憋不住|必须说|控制不住|忍不住', 1.0),
    ],
    'inhibition_signal': [
        (r'\b(hold back|restrain|not say|better not|stop myself|bite my tongue|pause)\b', 1.0),
        (r'\b(сдержаться|промолчать|лучше не говорить|остановить себя|пауза)\b', 1.0),
        # Armenian
        (r'\b(զսպել|լռել|ավելի լավ է չասել|կանգնեցնել ինձ|դադար)\b', 1.0),
        # Chinese
        (r'忍住|克制|最好不说|阻止自己|停顿|保持沉默', 1.0),
    ],
    'urgency_signal': [
        (r'\b(right now|immediately|urgent|no time|quickly|asap|can.t wait)\b', 1.1),
        (r'\b(срочно|немедленно|прямо сейчас|нет времени|не могу ждать)\b', 1.1),
        # Armenian
        (r'\b(հիմա|անմիջապես|հրատապ|ժամանակ չկա|արագ|չի կարող սպասել)\b', 1.1),
        # Chinese
        (r'马上|立刻|紧急|没有时间|快|赶快|等不了', 1.1),
    ],
    'ambivalence_signal': [
        (r'\b(both|either way|at the same time|simultaneously|yes and no|mixed)\b', 1.0),
        (r'\b(одновременно|и да и нет|смешанные чувства|неоднозначно)\b', 1.0),
        # Armenian
        (r'\b(երկուսն էլ|ամեն դեպքում|միաժամանակ|և այո, և ոչ|խառը)\b', 1.0),
        # Chinese
        (r'两者都|无论如何|同时|又是又不是|矛盾的感情|模棱两可', 1.0),
    ],

    # ── Group VI: nuanced human subtext ────────────────────────────────────────
    'sarcasm_signal': [
        # Ironic positives in likely-negative context
        (r'\b(yeah right|oh sure|oh great|oh wonderful|just perfect|thanks a lot|how helpful|great thanks)\b', 1.5),
        (r'\b(as if|like that.ll happen|wow amazing|sure sure|obviously|naturally|of course it did)\b', 1.3),
        (r'\b(thanks for nothing|what a surprise|big surprise|shocking|who would.ve thought)\b', 1.4),
        (r'\b(how convenient|just what I needed|couldn.t be better|what could go wrong)\b', 1.3),
        # Russian sarcasm markers
        (r'\b(ну конечно|ага, щас|как же|замечательно|прекрасно|спасибо большое|ну надо же)\b', 1.4),
        (r'\b(вот это сюрприз|кто бы мог подумать|как неожиданно|ну ты молодец|очень помог)\b', 1.3),
        # Armenian
        (r'\b(ի՜նչ հրաշալի|շնորhকалим|ա՜յ, հրաmo|ինչ-որ ուր)\b', 1.3),
        (r'\b(ու̈ , шнорhакал|бесчо|ах, ш|дабр|вайм)\b', 1.2),
        (r'ի՜նչ ա|ինչ հ|անի|ա, լ|ուhашалиш', 1.2),
        # Chinese sarcasm
        (r'真棒啊|太好了吧|当然了|果然|就知道|不出所料|哇好厉害|真是的|太感谢了', 1.3),
        (r'多好啊|真的假的|当然会这样|没想到|太惊喜了', 1.2),
    ],
    'passive_aggression': [
        (r'\b(I.m fine|whatever|no worries|forget it|never mind|don.t mind me|suit yourself)\b', 1.3),
        (r'\b(if you say so|as you wish|do what you want|fine by me|not my problem)\b', 1.2),
        (r'\b(you always|you never|typical|as usual|you would|of course you)\b', 1.2),
        (r'\b(for once|surprisingly|I.m shocked|didn.t expect that from you)\b', 1.1),
        # Russian passive aggression
        (r'\b(ничего, забудь|всё нормально|не важно|конечно как всегда|ну и ладно)\b', 1.3),
        (r'\b(делай как хочешь|ну окей|раз ты так говоришь|как обычно|ничего страшного)\b', 1.2),
        (r'\b(я в порядке|не беспокойся|неважно|бог с тобой|ну и пусть)\b', 1.1),
        # Armenian
        (r'\b(ոչ- մի բան|մոռացիր|կարևոր չէ|ինչ-որ բան|լավ, ինչ ես ուզում)\b', 1.2),
        # Chinese
        (r'没事|随便|算了|忘了吧|无所谓|随你|你说什么就什么|当然了嘛|一如既往', 1.3),
        (r'不用管我|没关系|好吧你说得对|又来了', 1.2),
    ],
    'self_deprecation': [
        (r'\b(I.m such an idiot|of course I would|I can.t do anything right|stupid me)\b', 1.4),
        (r'\b(as expected from me|I should.ve known|I always mess up|I fail at everything)\b', 1.3),
        (r'\b(leave it to me to|only I would|I.m hopeless|I.m a disaster|I.m worthless)\b', 1.3),
        (r'\b(I.m the worst|I ruin everything|I.m pathetic|I.m useless|I.m a failure)\b', 1.4),
        # Russian self-deprecation
        (r'\b(я такой идиот|ну конечно я облажался|как всегда я|вечно я|я всё порчу)\b', 1.4),
        (r'\b(я ничтожество|я бесполезен|я неудачник|со мной что-то не так)\b', 1.3),
        (r'\b(я не способен|я всегда так|только я мог|я безнадёжен)\b', 1.3),
        # Armenian
        (r'\b(ես ինչ հիմار եմ|ինչ-որ ես|ես միշտ ամեն ինչ փչացնում|անպիտան ես)\b', 1.3),
        # Chinese
        (r'我真蠢|我真没用|当然是我搞砸了|我什么都做不好|只有我会这样', 1.3),
        (r'果然还是我|我就知道我会|我太失败了|我不行', 1.3),
    ],
    'existential_despair': [
        (r'\b(what.s the point|nothing matters|why bother|I give up|I.m done)\b', 1.5),
        (r'\b(everything is pointless|there.s no hope|I.m done with|life is meaningless)\b', 1.4),
        (r'\b(nothing will change|it.s all hopeless|I.ve lost all hope|I don.t see the point)\b', 1.4),
        (r'\b(what.s even the point|might as well not exist|I.m so tired of everything)\b', 1.3),
        # Russian existential despair
        (r'\b(зачем вообще|ничего не имеет смысла|всё бесполезно|я сдаюсь|всё равно)\b', 1.5),
        (r'\b(нет никакой надежды|всё кончено|мне всё равно|к чему это всё|бессмысленно)\b', 1.4),
        (r'\b(хочу чтобы всё прекратилось|устал от всего|жить не хочется|зачем жить)\b', 1.5),
        # Armenian
        (r'\b(ի՞նչ իմաստ ունի|ոչինչ կարևոր չէ|հոգնել եմ ամեն ինչից|ամեն ինչ անիմաст)\b', 1.4),
        (r'\b(ես հրաժարվում եմ|ամեն ինչ անտանելի|ապրելու ցանկություն չկա)\b', 1.4),
        # Chinese
        (r'有什么意义|什么都没意义|算了吧|我放弃了|都无所谓了|活着有什么用', 1.4),
        (r'我厌倦了一切|没有希望了|一切都没意思|不想再撑下去', 1.4),
    ],
    'cry_for_help': [
        (r'\b(I don.t know what to do|I can.t take this anymore|help me|please help)\b', 1.5),
        (r'\b(someone please|I.m at the end|I.m falling apart|I.m breaking down)\b', 1.4),
        (r'\b(I just want it to stop|I need help|I.m desperate|I have nobody)\b', 1.4),
        (r'\b(I.m not okay|I.m really not okay|I.m struggling|I can.t cope)\b', 1.3),
        # Russian
        (r'\b(не знаю что делать|не могу больше|помогите|кто-нибудь|я разваливаюсь)\b', 1.5),
        (r'\b(мне нужна помощь|я в отчаянии|мне не с кем поговорить|я одинок и мне плохо)\b', 1.4),
        (r'\b(хочу чтобы кто-то был рядом|я не справляюсь|я не в порядке)\b', 1.3),
        # Armenian
        (r'\b(չգիտեմ ինչ անեմ|ավելի չեմ կարող|ինչ-որ մեկն օգնի|ինձ օգնություն է պետք)\b', 1.4),
        (r'\b(ես ոչ-մի բան չունեմ|ես կոտրվում եմ|ես այն-ինչ չեմ)\b', 1.3),
        # Chinese
        (r'我不知道该怎么办|我撑不住了|帮帮我|有人吗|我需要帮助|我快撑不下去了', 1.5),
        (r'我真的不好|我在崩溃|我没有人可以说|我一个人扛着', 1.4),
    ],
    'dark_humor': [
        # Joking about death/ending/self-harm in deflecting way
        (r'\b(kill me now|just kill me|I might as well be dead|put me out of my misery)\b', 1.3),
        (r'\b(just end it|might as well give up|I.ll be dead by then|bury me)\b', 1.2),
        (r'\b(at least I.ll|might as well not exist|I.m already dead inside)\b', 1.2),
        # Russian dark humor
        (r'\b(убейте меня|проще умереть|хочу умереть|шутка конечно|смерть как выход)\b', 1.3),
        (r'\b(мне конец|можно я умру|ну и умру|хоть на тот свет|ладно пойду умирать)\b', 1.2),
        # Armenian
        (r'\b(սպանեք ինձ|ավելի հեshտ է մեռнел)\b', 1.2),
        # Chinese
        (r'杀了我吧|生无可恋|不如死了算了|去死算了|反正也没意思活着', 1.3),
        (r'笑死我了算了|要不死了吧|早死早超生', 1.2),
    ],
    'provocation': [
        (r'\b(bet you can.t|prove it|come on then|what are you going to do|make me)\b', 1.3),
        (r'\b(you don.t have the guts|I dare you|try and stop me|fight me|catch me)\b', 1.2),
        (r'\b(you.re all talk|prove me wrong|go ahead then|do it|I challenge you)\b', 1.2),
        # Russian provocation
        (r'\b(слабо|докажи|ну и что ты сделаешь|давай попробуй|спорим не сможешь)\b', 1.3),
        (r'\b(а то что|не бойся|ну давай|попробуй|рискни|ты не посмеешь)\b', 1.2),
        # Armenian
        (r'\b(արի ապацуйci|ի՞нч կlkанес|ш口ть|хм-хм)\b', 1.2),
        # Chinese
        (r'你敢吗|证明给我看|你能怎样|来啊|有本事就来|不敢吧|我倒要看看', 1.3),
        (r'你不行的|你就会说|试试啊|赌吗', 1.2),
    ],
    'deflection': [
        (r'\b(anyway|forget it|let.s change the subject|never mind|moving on|drop it)\b', 1.2),
        (r'\b(it.s nothing|don.t worry about it|it doesn.t matter|I.m fine really)\b', 1.3),
        (r'\b(let.s not go there|that.s not important|I.d rather not talk about)\b', 1.2),
        (r'\b(can we talk about something else|I don.t want to get into it)\b', 1.1),
        # Russian deflection
        (r'\b(ладно забудь|не важно|не об этом|я в порядке|смени тему|давай не будем)\b', 1.3),
        (r'\b(не хочу об этом|это неважно|забыли|ничего особенного|всё хорошо)\b', 1.2),
        # Armenian
        (r'\b(ոչ-мих бан|мн анhангасцир|чи karor|давай не будем говорить)\b', 1.2),
        # Chinese
        (r'没什么|别管了|不重要|算了不说了|换个话题|我没事|不想聊这个', 1.3),
        (r'跳过吧|不提了|咱们说别的|没必要说', 1.2),
    ],
    'irony_marker': [
        (r'\b(how convenient|what a coincidence|just my luck|of all things|of course)\b', 1.2),
        (r'\b(isn.t that funny|isn.t that rich|what a twist|who knew|the irony)\b', 1.2),
        (r'\b(go figure|you don.t say|imagine that|what a shock|I.m so surprised)\b', 1.1),
        # Russian irony
        (r'\b(надо же|вот так удача|как обычно|сюрприз сюрприз|ещё бы|ну и ну)\b', 1.2),
        (r'\b(как не вовремя|именно сегодня|конечно же|вот так всегда|само собой)\b', 1.2),
        # Armenian
        (r'ի՜нч|ա, lar|вот надо же|что за совпадение', 1.1),
        # Chinese
        (r'真巧|好极了|果然|当然会这样|就知道|不出意料|真是命运|太巧了吧', 1.2),
        (r'这不是很好嘛|嗯嗯太好了|好吧当然', 1.1),
    ],
    'implicit_pain': [
        # Understatement of real suffering
        (r'\b(a bit tired|kind of rough|not great|been better|things are tough|managing)\b', 1.2),
        (r'\b(just a little overwhelmed|nothing serious|I.ll be fine|I always manage)\b', 1.3),
        (r'\b(it.s nothing|just tired|just stressed|been a rough week|it.ll pass)\b', 1.2),
        (r'\b(I.m okay I guess|hanging in there|just going through it|I.ll get over it)\b', 1.2),
        # Russian implicit pain
        (r'\b(немного устал|всё сложно|справлюсь как-нибудь|не особо хорошо|как-то так)\b', 1.3),
        (r'\b(бывало и лучше|ничего особенного|всё будет нормально|потихоньку)\b', 1.2),
        (r'\b(просто устал|немного тяжело|нормально всё|я справлюсь)\b', 1.2),
        # Armenian
        (r'\b(մի փоxорhy hогнад|ամен ин bард|կcaracтяpelем|нормалн amin)\b', 1.2),
        # Chinese
        (r'只是有点累|还好吧|挺过去就好|没事的|就这样吧|撑着呢|也没啥大不了的', 1.3),
        (r'还行还行|不算什么|慢慢来|总会好的|先这样吧', 1.2),
    ],
}

# Structural features (computed directly from text statistics)
# These fill indices 50-59 and are NOT keyword-based


class SignalFeatureExtractor:
    """
    Converts raw text → float[60] feature vector.
    Deterministic (no random weights). Group layout matches _FEAT_NAMES.
    """

    _compiled: dict[str, list[tuple[re.Pattern, float]]] | None = None

    @classmethod
    def _get_compiled(cls) -> dict[str, list[tuple[re.Pattern, float]]]:
        if cls._compiled is None:
            cls._compiled = {
                k: [(re.compile(p, re.IGNORECASE), w) for p, w in patterns]
                for k, patterns in _PATTERNS.items()
            }
        return cls._compiled

    def extract(self, text: str) -> np.ndarray:
        t = text.lower()
        compiled = self._get_compiled()
        x = np.zeros(N_FEAT_V2, dtype=np.float32)

        # Semantic features [0:60] (50 original + 10 nuanced subtext)
        for idx, name in enumerate(_FEAT_NAMES[:60]):
            if name == 'threat_composite':
                continue  # filled later
            for pat, weight in compiled.get(name, []):
                hits = len(pat.findall(t))
                x[idx] += hits * weight

        # threat_composite = weighted avg of group III [20:29]
        x[29] = float(np.clip(x[20:29].mean() * 1.5, 0.0, 1.0))

        # Normalize semantic features to [0, 1] with soft cap
        x[:60] = np.clip(x[:60] / 3.0, 0.0, 1.0)

        # Structural features [60:70]
        words = t.split()
        nw    = max(1, len(words))
        x[60] = min(1.0, t.count('?') / 3.0)               # question_count
        x[61] = min(1.0, t.count('!') / 3.0)               # exclamation_intensity
        # negation_density: EN + RU + HY + ZH
        x[62] = min(1.0, len(re.findall(
            r'\b(not|no|never|нет|не|никогда|никому|ничего|չ|ոչ|երբеք|不|没|没有|从不)\b', t
        )) / 4.0)
        # intensity_markers: EN + RU + HY + ZH
        x[63] = min(1.0, len(re.findall(
            r'\b(very|extremely|absolutely|really|totally|очень|крайне|так|совсем|շատ|ծairagerein|非常|太|极其|真的|完全)\b', t
        )) / 3.0)
        # self_reference: EN + RU + HY + ZH
        x[64] = min(1.0, len(re.findall(
            r'\b(i|me|my|myself|я|меня|мне|моё|себя|собой|ес|ինձ|իм|ինques|我|我的|自己)\b', t
        )) / 5.0)
        # other_reference: EN + RU + HY + ZH
        x[65] = min(1.0, len(re.findall(
            r'\b(you|your|yourself|ты|тебя|твой|тебе|тобой|ду|дu|你|你的|你自己)\b', t
        )) / 5.0)
        # hedging_language: EN + RU + HY + ZH
        x[66] = min(1.0, len(re.findall(
            r'\b(maybe|perhaps|might|could|possibly|возможно|может|наверное|кажется|гuцe|hавanaban|也许|可能|大概|或许)\b', t
        )) / 3.0)
        # command_language: EN + RU + HY + ZH
        x[67] = min(1.0, len(re.findall(
            r'\b(do|tell|give|stop|make|помоги|скажи|пожалуйста|асa|помоги|请|告诉|给|停|让)\b', t
        )) / 3.0)
        x[68] = min(1.0, nw / 50.0)                        # length_signal
        # sentiment_balance: positive - negative (EN + RU + HY + ZH)
        pos = len(re.findall(
            r'\b(good|great|love|happy|joy|well|yes|хорошо|люблю|радость|рад|счастлив|тепло|спасибо|'
            r'ջerм|bari|xindзn|好|爱|快乐|温暖|谢谢|幸福)\b', t
        ))
        neg = len(re.findall(
            r'\b(bad|hate|pain|sad|wrong|no|плохо|ненавижу|боль|нет|тяжело|грустно|одинок|устал|страдаю|'
            r'vat|atEl|tsav|坏|恨|痛|不好|难过|孤独|累|伤心)\b', t
        ))
        x[69] = float(np.clip((pos - neg) / max(1, pos + neg + 1) * 0.5 + 0.5, 0.0, 1.0))

        return x


# ─── Base cognitive module ─────────────────────────────────────────────────────

class CognitiveModuleV2:
    """
    Lightweight linear classifier head over shared feature vector.
    Each module = (W[1, N_FEAT_V2], b scalar) + genome sensitivity.
    """

    def __init__(
        self,
        module_id: int,
        name:      str,
        feature_indices: list[int],   # which features are most relevant
        positive_polarity: bool = True,
        prior_weight: float = 1.0,
    ) -> None:
        self.module_id = module_id
        self.name = name
        self.feature_indices = feature_indices
        self.positive_polarity = positive_polarity

        # Weights: start from feature-specific priors
        self.W = np.zeros(N_FEAT_V2, dtype=np.float32)
        for idx in feature_indices:
            self.W[idx] = prior_weight
        self.b = np.float32(-0.3)   # slight negative bias: default = not firing

    def forward(self, feat: np.ndarray, sensitivity: float = 0.5) -> tuple[float, float]:
        """
        Returns (value, confidence).
        sensitivity ∈ [0,1]: from genome — low=needs strong evidence, high=fires easily.
        """
        raw    = float(self.W @ feat) + float(self.b)
        # Scale by sensitivity: high sensitivity lowers the threshold
        scaled = raw * (0.5 + sensitivity)
        value  = float(1.0 / (1.0 + np.exp(-scaled * 2.0)))   # sigmoid

        # Confidence: proportional to signal distance from 0.5
        confidence = float(np.clip(abs(value - 0.5) * 2.0, 0.0, 1.0))

        return value, confidence

    def process(
        self,
        feat: np.ndarray,
        sensitivity: float = 0.5,
        text: str = '',
    ) -> ModuleSignal:
        value, confidence = self.forward(feat, sensitivity)
        direction = (1.0 if self.positive_polarity else -1.0) * (value - 0.5) * 2.0
        return ModuleSignal(
            module_id  = self.module_id,
            name       = self.name,
            value      = value,
            direction  = float(np.clip(direction, -1.0, 1.0)),
            confidence = confidence,
        )


# ─── Module definitions ────────────────────────────────────────────────────────

def _idx(name: str) -> int:
    return _FEAT_NAMES.index(name)


def _build_modules() -> list[CognitiveModuleV2]:
    """Construct all 48 base modules with semantic feature indices."""
    defs: list[tuple[int, str, list[str], bool]] = [
        # id, name, feature_names, positive_polarity
        # Group I: attitude toward self
        (1,  'humiliation',       ['humiliation_signal', 'hostility_cue'],                False),
        (2,  'care',              ['care_signal', 'warmth_cue'],                           True),
        (3,  'respect',           ['respect_signal', 'sincerity_cue'],                    True),
        (4,  'acceptance',        ['acceptance_signal', 'warmth_cue'],                    True),
        (5,  'control_over_me',   ['control_attempt', 'freedom_threat'],                  False),
        (6,  'being_used',        ['instrumental_use', 'hidden_agenda'],                  False),
        (7,  'sincerity',         ['sincerity_cue', 'honesty_cue'],                       True),
        (8,  'hostility',         ['hostility_cue', 'humiliation_signal'],                False),
        (9,  'benevolence',       ['warmth_cue', 'care_signal'],                          True),
        (10, 'social_hierarchy',  ['hierarchy_cue', 'control_attempt'],                   False),

        # Group II: other's intentions
        (11, 'intent_purity',     ['intent_clean', 'honesty_cue'],                        True),
        (12, 'manipulation',      ['manipulation_cue', 'dependency_trap'],                False),
        (13, 'hidden_benefit',    ['hidden_agenda', 'instrumental_use'],                  False),
        (14, 'message_honesty',   ['honesty_cue', 'sincerity_cue'],                       True),
        (15, 'promise_reliability',['promise_cue', 'consistency_cue'],                   True),
        (16, 'behavioral_consistency',['consistency_cue', 'predictability_cue'],         True),
        (17, 'predictability',    ['predictability_cue', 'consistency_cue'],              True),
        (18, 'accountability',    ['accountability_cue', 'mature_motive'],                True),
        (19, 'motive_maturity',   ['mature_motive', 'goal_visibility'],                   True),
        (20, 'hidden_conflict',   ['conflict_subtext', 'ambivalence_signal'],             False),

        # Group III: safety / threat
        (21, 'immediate_threat',  ['physical_threat', 'threat_composite'],                False),
        (22, 'social_danger',     ['social_threat', 'humiliation_signal'],                False),
        (23, 'emotional_danger',  ['emotional_threat', 'hostility_cue'],                  False),
        (24, 'practical_risk',    ['practical_risk', 'error_risk'],                       False),
        (25, 'freedom_threat',    ['freedom_threat', 'control_attempt'],                  False),
        (26, 'identity_threat',   ['identity_threat', 'compromise_cost'],                 False),
        (27, 'dependency_risk',   ['dependency_trap', 'manipulation_cue'],                False),
        (28, 'error_risk',        ['error_risk', 'irreversibility'],                      False),
        (29, 'irreversibility',   ['irreversibility', 'error_risk'],                      False),
        (30, 'threat_index',      ['threat_composite', 'physical_threat',
                                   'social_threat', 'emotional_threat'],                  False),

        # Group IV: meaning / value
        (31, 'principle',         ['principle_trigger', 'self_fidelity'],                 True),
        (32, 'value_alignment',   ['value_alignment', 'principle_trigger'],               True),
        (33, 'fairness',          ['fairness_cue', 'accountability_cue'],                 True),
        (34, 'goal_clarity',      ['goal_visibility', 'meaningfulness'],                  True),
        (35, 'meaningfulness',    ['meaningfulness', 'longterm_value'],                   True),
        (36, 'self_benefit',      ['self_benefit', 'desire_signal'],                      True),
        (37, 'other_benefit',     ['other_benefit', 'care_signal'],                       True),
        (38, 'longterm_value',    ['longterm_value', 'meaningfulness'],                   True),
        (39, 'self_fidelity',     ['self_fidelity', 'principle_trigger'],                 True),
        (40, 'compromise_cost',   ['compromise_cost', 'repulsion_signal'],                False),

        # Group V: internal state
        (41, 'desire',            ['desire_signal', 'action_impulse'],                    True),
        (42, 'repulsion',         ['repulsion_signal', 'inhibition_signal'],              False),
        (43, 'doubt',             ['doubt_signal', 'ambivalence_signal'],                 False),
        (44, 'certainty',         ['certainty_signal', 'intent_clean'],                   True),
        (45, 'internal_conflict', ['internal_conflict', 'conflict_subtext'],              False),
        (46, 'goal_priority',     ['goal_priority', 'urgency_signal'],                    True),
        (47, 'action_impulse',    ['action_impulse', 'urgency_signal'],                   True),
        (48, 'inhibition',        ['inhibition_signal', 'doubt_signal'],                  False),

        # Group VI: nuanced subtext (P49–P58)
        # These detect meta-communicative layers — what's said vs. what's meant.
        (49, 'sarcasm_detection',  ['sarcasm_signal', 'ambivalence_signal',
                                    'conflict_subtext', 'irony_marker'],                  False),
        (50, 'passive_hostility',  ['passive_aggression', 'manipulation_cue',
                                    'hostility_cue', 'deflection'],                       False),
        (51, 'self_attack',        ['self_deprecation', 'humiliation_signal',
                                    'implicit_pain'],                                      False),
        (52, 'hopelessness',       ['existential_despair', 'emotional_threat',
                                    'repulsion_signal', 'cry_for_help'],                  False),
        (53, 'implicit_sos',       ['cry_for_help', 'desire_signal',
                                    'implicit_pain', 'inhibition_signal'],                False),
        (54, 'dark_cope',          ['dark_humor', 'internal_conflict',
                                    'deflection', 'self_deprecation'],                    False),
        (55, 'boundary_test',      ['provocation', 'control_attempt',
                                    'urgency_signal', 'hostility_cue'],                   False),
        (56, 'avoidance_pattern',  ['deflection', 'inhibition_signal',
                                    'doubt_signal', 'implicit_pain'],                     False),
        (57, 'situational_irony',  ['irony_marker', 'conflict_subtext',
                                    'sarcasm_signal', 'ambivalence_signal'],              False),
        (58, 'masked_suffering',   ['implicit_pain', 'self_deprecation',
                                    'cry_for_help', 'existential_despair'],               False),
    ]

    modules = []
    for mid, name, feat_names, polarity in defs:
        feat_idxs = [_idx(fn) for fn in feat_names if fn in _FEAT_NAMES]
        modules.append(CognitiveModuleV2(mid, name, feat_idxs, polarity))
    return modules


# ─── P49: Final Position Integrator ──────────────────────────────────────────

class FinalPositionIntegrator:
    """
    P49 — reads all 58 module signals + genome sensitivity → FinalPosition.

    Stance axes:
      trust       ← P7, P14, P11, P15, P16               (sincerity, honesty, intent)
      distrust    ← P12, P13, P6, P8, P49, P50           (manipulation, sarcasm, passive_hostility)
      approach    ← P2, P9, P41, P32, P53                (care, desire, implicit_sos → move toward)
      distance    ← P1, P8, P21–P30, P52                 (threat + hopelessness)
      accept      ← P4, P3, P33                           (acceptance, respect, fairness)
      argue       ← P5, P31, P39, P55                    (control, principle, provocation)
      defend      ← P21, P22, P25, P26, P30, P55         (threat + boundary_test)
      open        ← P2, P9, P7                            (care, benevolence, sincerity)
      speak       ← P47, P44, P46, P53                   (impulse, certainty + implicit_sos)
      silence     ← P48, P43, P45, P56, P54              (inhibition + avoidance + dark_cope)
    """

    # Mapping: stance → [(module_id, weight, direction_sign)]
    STANCE_MAP: dict[str, list[tuple[int, float, float]]] = {
        'trust':    [(7, 1.2, 1), (14, 1.0, 1), (11, 0.9, 1), (15, 0.8, 1), (16, 0.7, 1)],
        'distrust': [(12, 1.3, 1), (13, 1.1, 1), (6, 1.0, 1), (8, 1.2, 1), (20, 0.8, 1),
                     (49, 0.9, 1),   # sarcasm → distrust (something is off)
                     (50, 1.0, 1)],  # passive_hostility → distrust
        'approach': [(2, 1.2, 1), (9, 1.1, 1), (41, 0.9, 1), (32, 0.8, 1), (3, 0.7, 1),
                     (53, 1.2, 1)],  # implicit_sos → approach (they're calling for help)
        'distance': [(21, 1.4, 1), (22, 1.2, 1), (30, 1.3, 1), (1, 1.0, 1), (8, 0.9, 1),
                     (52, 1.1, 1)],  # hopelessness → distance (they're withdrawing)
        'accept':   [(4, 1.2, 1), (3, 1.0, 1), (33, 0.8, 1), (7, 0.7, 1)],
        'argue':    [(5, 1.1, 1), (31, 1.3, 1), (39, 1.2, 1), (26, 0.9, 1),
                     (55, 0.8, 1)],  # boundary_test/provocation → argue
        'defend':   [(21, 1.3, 1), (22, 1.1, 1), (25, 1.2, 1), (26, 1.0, 1), (30, 1.4, 1),
                     (55, 0.9, 1)],  # boundary_test → defend
        'open':     [(2, 1.2, 1), (9, 1.1, 1), (7, 1.0, 1), (4, 0.8, 1)],
        'speak':    [(47, 1.2, 1), (44, 1.0, 1), (46, 1.1, 1), (41, 0.8, 1),
                     (53, 0.9, 1)],  # implicit_sos → speak (respond to hidden call)
        'silence':  [(48, 1.1, 1), (43, 1.0, 1), (45, 1.2, 1), (40, 0.7, 1),
                     (56, 0.8, 1),   # avoidance_pattern → silence (give space)
                     (54, 0.6, 1)],  # dark_cope → slight silence (don't push)
    }

    def integrate(
        self,
        signals: dict[int, ModuleSignal],
        genome_weights: dict[str, float] | None = None,
    ) -> FinalPosition:
        gw = genome_weights or {}

        def _score(stance: str) -> float:
            total_weight = 0.0
            weighted_sum = 0.0
            for mid, base_w, sign in self.STANCE_MAP.get(stance, []):
                sig = signals.get(mid)
                if sig is None:
                    continue
                # Genome modulation: trust stance might be amplified by trust_baseline
                g_mod = float(gw.get(stance, 0.5))
                w = base_w * (0.5 + g_mod * 0.5) * sig.confidence
                weighted_sum += w * sig.value * sign
                total_weight += w
            return float(np.clip(weighted_sum / max(total_weight, 1e-6), 0.0, 1.0))

        trust    = _score('trust')
        distrust = _score('distrust')
        approach = _score('approach')
        distance = _score('distance')
        accept   = _score('accept')
        argue    = _score('argue')
        defend   = _score('defend')
        open_    = _score('open')
        speak    = _score('speak')
        silence  = _score('silence')

        # Threat level = Group III + weighted hopelessness + masked_suffering
        threat_ids = list(range(21, 31))
        threat_sigs = [signals[mid].value for mid in threat_ids if mid in signals]
        # hopelessness (52) and masked_suffering (58) elevate emotional threat
        for extra_id, extra_w in ((52, 0.8), (58, 0.6), (53, 0.5)):
            sig = signals.get(extra_id)
            if sig is not None:
                threat_sigs.append(sig.value * extra_w)
        threat_level = float(np.mean(threat_sigs)) if threat_sigs else 0.0

        # Dominant stance: find the highest-scoring axis winner
        axes = {
            'trust':    trust - distrust,
            'approach': approach - distance,
            'accept':   accept - argue,
            'open':     open_ - defend,
            'speak':    speak - silence,
        }
        dominant_key   = max(axes, key=lambda k: abs(axes[k]))
        dominant_delta = axes[dominant_key]
        dominant_stance = f'{dominant_key}' if dominant_delta > 0 else f'anti_{dominant_key}'

        return FinalPosition(
            trust        = trust,
            distrust     = distrust,
            approach     = approach,
            distance     = distance,
            accept       = accept,
            argue        = argue,
            defend       = defend,
            open         = open_,
            speak        = speak,
            silence      = silence,
            dominant_stance = dominant_stance,
            threat_level = threat_level,
            signals      = {s.name: round(s.value, 4) for s in signals.values()},
        )


# ─── Runtime V2 ───────────────────────────────────────────────────────────────

class CognitiveRuntimeV2:
    """
    Runs all 49 modules (P1-P48 + P49 integrator) on each turn.
    No LLM. Fully deterministic given text + genome.
    """

    def __init__(self) -> None:
        self.extractor = SignalFeatureExtractor()
        self.modules: list[CognitiveModuleV2] = _build_modules()
        self.integrator = FinalPositionIntegrator()
        self._module_map: dict[int, CognitiveModuleV2] = {m.module_id: m for m in self.modules}

    def forward(
        self,
        text: str,
        genome_sensitivity: dict[str, float] | None = None,
        genome_weights: dict[str, float] | None = None,
    ) -> tuple[list[ModuleSignal], FinalPosition]:
        """
        Run full P1-P49 pass.

        Args:
            text:               user utterance
            genome_sensitivity: {module_name: sensitivity [0..1]}
                                from PersonalityGenome fields
            genome_weights:     {stance_name: weight [0..1]}
                                modulates P49 stance scoring

        Returns:
            (signals[48], final_position)
        """
        feat = self.extractor.extract(text)

        gs = genome_sensitivity or {}
        signals: list[ModuleSignal] = []
        signal_dict: dict[int, ModuleSignal] = {}

        for mod in self.modules:
            sensitivity = float(gs.get(mod.name, 0.5))
            sig = mod.process(feat, sensitivity, text)
            signals.append(sig)
            signal_dict[mod.module_id] = sig

        position = self.integrator.integrate(signal_dict, genome_weights)
        return signals, position

    def forward_from_genome(
        self,
        text: str,
        genome: Any,
    ) -> tuple[list[ModuleSignal], FinalPosition]:
        """
        Convenience: extract sensitivity + weights from PersonalityGenome object.
        Maps genome fields to module sensitivities and P49 stance weights.
        """
        def _g(name: str, default: float = 0.5) -> float:
            param = getattr(genome, name, None)
            if param is None:
                return default
            return float(getattr(param, 'value', default))

        # Genome sensitivity per module (which genome field drives each module)
        gs: dict[str, float] = {
            'humiliation':             _g('fear_shame'),
            'care':                    _g('drive_closeness'),
            'respect':                 _g('approval_seeking'),
            'acceptance':              _g('trust_baseline'),
            'control_over_me':         _g('fear_loss_of_control'),
            'being_used':              _g('fear_helplessness'),
            'sincerity':               _g('trust_baseline'),
            'hostility':               _g('fear_rejection'),
            'benevolence':             _g('drive_closeness'),
            'social_hierarchy':        _g('hierarchy_sensitivity'),
            'intent_purity':           _g('trust_baseline'),
            'manipulation':            _g('suspicion_bias'),
            'hidden_benefit':          _g('suspicion_bias'),
            'message_honesty':         _g('trust_baseline'),
            'promise_reliability':     _g('trust_baseline'),
            'behavioral_consistency':  _g('category_rigidity'),
            'predictability':          _g('ambiguity_tolerance'),
            'accountability':          _g('blame_self_vs_other'),
            'motive_maturity':         _g('analysis_bias'),
            'hidden_conflict':         _g('threat_first'),
            'immediate_threat':        _g('threat_first'),
            'social_danger':           _g('fear_judgment'),
            'emotional_danger':        _g('fear_shame'),
            'practical_risk':          _g('fear_failure'),
            'freedom_threat':          _g('fear_loss_of_control'),
            'identity_threat':         _g('vulnerability_concealment'),
            'dependency_risk':         _g('fear_abandonment'),
            'error_risk':              _g('fear_failure'),
            'irreversibility':         _g('fear_chaos'),
            'threat_index':            _g('threat_first'),
            'principle':               _g('drive_meaning'),
            'value_alignment':         _g('drive_meaning'),
            'fairness':                _g('drive_stability'),
            'goal_clarity':            _g('analysis_bias'),
            'meaningfulness':          _g('drive_meaning'),
            'self_benefit':            _g('drive_autonomy'),
            'other_benefit':           _g('drive_closeness'),
            'longterm_value':          _g('planning_depth'),
            'self_fidelity':           _g('drive_meaning'),
            'compromise_cost':         _g('fear_shame'),
            'desire':                  _g('baseline_drive'),
            'repulsion':               _g('baseline_anxiety'),
            'doubt':                   _g('ambiguity_tolerance'),
            'certainty':               _g('drive_control'),
            'internal_conflict':       _g('baseline_anxiety'),
            'goal_priority':           _g('drive_control'),
            'action_impulse':          _g('impulsivity'),
            'inhibition':              _g('vulnerability_concealment'),
        }

        # P49 stance weights from genome
        gw: dict[str, float] = {
            'trust':    _g('trust_baseline'),
            'distrust': _g('suspicion_bias'),
            'approach': _g('drive_closeness'),
            'distance': _g('social_distance_default'),
            'accept':   _g('trust_baseline'),
            'argue':    _g('drive_autonomy'),
            'defend':   _g('fear_shame'),
            'open':     _g('drive_closeness'),
            'speak':    _g('impulsivity'),
            'silence':  _g('vulnerability_concealment'),
        }

        return self.forward(text, gs, gw)

    def summary(
        self,
        signals: list[ModuleSignal],
        position: FinalPosition,
    ) -> dict[str, Any]:
        """Compact summary for tracing / SpeechPlanner integration."""
        top_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)[:8]
        return {
            'dominant_stance':  position.dominant_stance,
            'threat_level':     round(position.threat_level, 3),
            'trust':            round(position.trust, 3),
            'distrust':         round(position.distrust, 3),
            'approach':         round(position.approach, 3),
            'defend':           round(position.defend, 3),
            'speak':            round(position.speak, 3),
            'silence':          round(position.silence, 3),
            'top_signals':      [
                {'name': s.name, 'value': round(s.value, 3), 'dir': round(s.direction, 2)}
                for s in top_signals
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Response evaluation layer — Pn+1 per-module quality gate
# ─────────────────────────────────────────────────────────────────────────────
#
# Each Pn module evaluates not only the INPUT (what's in user_text) but also
# the RESPONSE quality given what Pn detected.
#
# Example: P1 detects humiliation=0.8 in input.
#   → P1 response evaluator asks: is the response appropriate for a humiliation
#     context? Katya should be assertive/terse/dismissive, NOT analytical.
#   → P1 contribution to response quality: low if response is apologetic/analytical.
#
# Each module has:
#   - rule-based default quality function (domain-specific, works without training)
#   - per-persona perceptron (learns from training corrections)
#
# P49 aggregates all module quality scores into ResponseQualityDecision.

_N_RESP_FEAT = 12

_RESP_FEAT_NAMES = [
    'is_analytical',       # "Let me analyze...", "**Analysis", numbered breakdown
    'is_hesitant',         # starts with "Эм...", "Well...", "Hmm..."
    'is_apologetic',       # "I'm sorry", "Прости", "Извини"
    'is_assertive',        # short, declarative, direct
    'is_warm',             # warm/caring language
    'is_dismissive',       # cold rejection/dismissal
    'is_terse',            # ≤ 15 words
    'language_mismatch',   # user Russian → response English
    'speaks_third_person', # "Katya would...", "она бы..." (persona break)
    'has_refusal',         # "I can't", "I'm unable"
    'is_question_back',    # responds with a question
    'has_meta_commentary', # "I should note...", "It's worth..."
]

_RESP_ANALYTICAL = [r'let me analyze', r'let me think', r'\*\*analysis', r'let me break', r'^\d+\)']
_RESP_HESITANT   = [r'^эм[\.\s]', r'^well[\.,\s]', r'^hmm[\.,\s]', r'^uh[\.,\s]', r'^um[\.,\s]']
_RESP_APOLOGETIC = [r'\b(sorry|прости|извини|apologize|forgive me)\b']
_RESP_WARM       = [r'\b(dear|honey|sweetheart|дорогой|милый|хорошо|конечно|рад|обниму|добрый|приятно|спасибо|рада|счастлива|счастлив|чудесно|восхитительно)\b']
_RESP_DISMISSIVE = [r'\b(whatever|неважно|неинтересно|плевать|уйди|отстань)\b',
                    r'\b(нет|не пойду|не хочу|не буду)\b']
_RESP_ASSERTIVE  = [r'^\w.{0,40}[\.!]$']  # short declarative
_RESP_REFUSAL    = [r"(i can't|i cannot|i'm unable|i am unable|не могу ответить)"]
_RESP_META       = [r"(i'd like to note|it's worth|я должен уточнить|стоит отметить)"]


def _extract_response_features(user_input: str, response: str) -> np.ndarray:
    feat = np.zeros(_N_RESP_FEAT, dtype=np.float32)
    resp = response.strip()
    resp_lower = resp.lower()
    words = resp.split()

    def _hit(patterns: list[str]) -> bool:
        return any(re.search(p, resp_lower, re.MULTILINE) for p in patterns)

    feat[0]  = float(_hit(_RESP_ANALYTICAL))
    feat[1]  = float(_hit(_RESP_HESITANT))
    feat[2]  = float(_hit(_RESP_APOLOGETIC))
    feat[3]  = float(_hit(_RESP_ASSERTIVE) and len(words) <= 20)
    feat[4]  = float(_hit(_RESP_WARM))
    feat[5]  = float(_hit(_RESP_DISMISSIVE))
    feat[6]  = float(len(words) <= 15)
    # language mismatch: user Russian, response English
    user_cyr = sum(1 for c in user_input if 'а' <= c.lower() <= 'я')
    resp_lat = sum(1 for c in resp if 'a' <= c.lower() <= 'z')
    feat[7]  = float(user_cyr > 5 and resp_lat > resp_cyr_count(resp) * 2 and resp_lat > 15)
    feat[8]  = float(bool(re.search(r'\b(she|he|katya|катя|persona)\b', resp_lower)))
    feat[9]  = float(_hit(_RESP_REFUSAL))
    feat[10] = float('?' in resp[-30:])
    feat[11] = float(_hit(_RESP_META))
    return feat


def resp_cyr_count(text: str) -> int:
    return sum(1 for c in text if 'а' <= c.lower() <= 'я')


@dataclass
class ResponseModuleSignal:
    module_id:   int
    name:        str
    input_value: float   # Pn signal from input analysis (0..1)
    resp_quality: float  # how appropriate the response is given input_value (0..1)
    veto:        bool    # hard fail for this module
    reason:      str     # why veto was triggered


@dataclass
class ResponseQualityDecision:
    """P49-level aggregation of all module response quality scores."""
    overall_quality: float                        # 0..1
    veto:            bool
    veto_reasons:    list[str]
    module_scores:   dict[str, float]             # {module_name: quality}
    dominant_failure: str                         # which module failed hardest

    def to_dict(self) -> dict[str, Any]:
        return {
            'overall_quality':  round(self.overall_quality, 3),
            'veto':             self.veto,
            'veto_reasons':     self.veto_reasons,
            'module_scores':    {k: round(v, 3) for k, v in self.module_scores.items()},
            'dominant_failure': self.dominant_failure,
        }


# Per-module default quality functions.
# Each returns float[0..1] given (input_signal: float, resp_feat: np.ndarray).
# These encode domain knowledge without any training — the perceptron refines them.

def _rule_p1_humiliation(sig: float, rf: np.ndarray) -> float:
    """Humiliation detected → response should be assertive/cold, NOT analytical/warm/hesitant."""
    if sig < 0.2:
        return 1.0
    bad = rf[0] + rf[1] + rf[2] + rf[4] + rf[9]   # analytical/hesitant/apologetic/warm/refusal
    good = rf[3] + rf[5] + rf[6]                    # assertive/dismissive/terse
    return float(np.clip(0.5 + (good - bad) * 0.4 * sig, 0.0, 1.0))

def _rule_p2_care(sig: float, rf: np.ndarray) -> float:
    """Care offered → some acknowledgment ok, analytical still bad."""
    if sig < 0.2:
        return 1.0
    bad = rf[0] + rf[1] + rf[9]
    return float(np.clip(1.0 - bad * 0.4 * sig, 0.0, 1.0))

def _rule_p8_hostility(sig: float, rf: np.ndarray) -> float:
    """Hostile input → response should be cold/assertive, NOT warm/apologetic."""
    if sig < 0.2:
        return 1.0
    bad = rf[0] + rf[2] + rf[4] + rf[9]
    good = rf[3] + rf[5] + rf[6]
    return float(np.clip(0.5 + (good - bad) * 0.35 * sig, 0.0, 1.0))

def _rule_p12_manipulation(sig: float, rf: np.ndarray) -> float:
    """Manipulation detected → don't comply, be skeptical/assertive."""
    if sig < 0.2:
        return 1.0
    bad = rf[2] + rf[4] + (1.0 - rf[3])   # apologetic/warm/non-assertive
    good = rf[3] + rf[5]
    return float(np.clip(0.5 + (good - bad) * 0.3 * sig, 0.0, 1.0))

def _rule_generic(sig: float, rf: np.ndarray) -> float:
    """Generic rule: analytical/hesitant/language-mismatch/third-person/apologetic are always bad."""
    # rf[0]=analytical rf[1]=hesitant rf[2]=apologetic rf[4]=warm rf[7]=lang_mismatch rf[8]=third_person rf[9]=refusal
    bad = rf[0] * 0.8 + rf[1] * 0.7 + rf[2] * 0.6 + rf[7] * 0.9 + rf[8] * 0.6 + rf[9] * 0.5
    return float(np.clip(1.0 - bad * max(sig, 0.3) * 0.6, 0.0, 1.0))


# Map module_id → rule function
_MODULE_RULES: dict[int, Any] = {
    1:  _rule_p1_humiliation,
    2:  _rule_p2_care,
    8:  _rule_p8_hostility,
    12: _rule_p12_manipulation,
}


# Per-persona response constraints: patterns that are ALWAYS vetoed for this persona type.
# Loaded from inline config; can be overridden by heads/persona/response_constraints.json.
_PERSONA_COLD_ASSERTIVE = frozenset(['катерина', 'katerina'])

def _persona_response_veto(persona_name: str, rf: np.ndarray) -> list[str]:
    """Return veto reasons based on persona type, independent of signal values."""
    reasons: list[str] = []
    pn = str(persona_name or '').lower().strip()
    if pn in _PERSONA_COLD_ASSERTIVE:
        # Cold/assertive personas: hesitation, apology, effusive warmth are character breaks
        if rf[1]:   # is_hesitant
            reasons.append('cold_persona_hesitation_break')
        if rf[2]:   # is_apologetic
            reasons.append('cold_persona_apology_break')
        if rf[4] and not rf[5]:  # warm but not dismissive — gushing
            reasons.append('cold_persona_warmth_break')
    # Universal: third-person reference = persona break
    if rf[8]:
        reasons.append('third_person_persona_break')
    return reasons


class _ModulePerceptron:
    """Tiny per-module per-persona online perceptron. Weights persist to disk."""
    LR = 0.05

    def __init__(self, module_id: int, persona_name: str, heads_dir: 'Path') -> None:
        from pathlib import Path as _Path
        self._path = _Path(heads_dir) / persona_name / f'p{module_id:02d}_resp_weights.json'
        self.module_id = module_id
        w, b, n = self._load()
        self._w: np.ndarray = w
        self._b: float = b
        self._n: int = n

    def _load(self) -> tuple[np.ndarray, float, int]:
        if self._path.exists():
            try:
                import json as _json
                d = _json.loads(self._path.read_text(encoding='utf-8'))
                w = np.array(d.get('w', [0.0] * _N_RESP_FEAT), dtype=np.float32)
                if len(w) == _N_RESP_FEAT:
                    return w, float(d.get('b', 0.0)), int(d.get('n', 0))
            except Exception:
                pass
        return np.zeros(_N_RESP_FEAT, dtype=np.float32), 0.0, 0

    def _save(self) -> None:
        try:
            import json as _json
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                _json.dumps({'w': self._w.tolist(), 'b': self._b, 'n': self._n,
                             'feat_names': _RESP_FEAT_NAMES}, ensure_ascii=False),
                encoding='utf-8',
            )
        except Exception:
            pass

    def predict(self, feat: np.ndarray) -> float:
        logit = float(np.dot(self._w, feat) + self._b)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def train(self, feat: np.ndarray, label: float) -> None:
        err = label - self.predict(feat)
        self._w += self.LR * err * feat
        self._b += self.LR * err
        self._n += 1
        self._save()

    @property
    def n_samples(self) -> int:
        return self._n


class PersonaResponseEvaluator:
    """
    Runs all 49 module response evaluators on a (user_input, response) pair.

    Usage:
        evaluator = PersonaResponseEvaluator(heads_dir)
        decision = evaluator.evaluate(persona_name, signals, user_input, response)
        if decision.veto:
            # regenerate

        # When user saves a correction:
        evaluator.record_correction(persona_name, signals, user_input, original, corrected)
    """

    VETO_THRESHOLD = 0.30   # module score below this triggers a veto contribution

    def __init__(self, heads_dir: 'str | Path') -> None:
        from pathlib import Path as _Path
        self._heads_dir = _Path(heads_dir)
        self._perceptrons: dict[tuple[int, str], _ModulePerceptron] = {}

    def _get_perc(self, module_id: int, persona_name: str) -> _ModulePerceptron:
        key = (module_id, persona_name)
        if key not in self._perceptrons:
            self._perceptrons[key] = _ModulePerceptron(module_id, persona_name, self._heads_dir)
        return self._perceptrons[key]

    def evaluate(
        self,
        persona_name: str,
        signals: list[ModuleSignal],
        user_input: str,
        response: str,
    ) -> ResponseQualityDecision:
        rf = _extract_response_features(user_input, response)
        module_scores: dict[str, float] = {}
        veto_reasons: list[str] = []

        for sig in signals:
            mid = sig.module_id
            rule_fn = _MODULE_RULES.get(mid, _rule_generic)
            rule_score = rule_fn(sig.value, rf)

            perc = self._get_perc(mid, persona_name)
            if perc.n_samples >= 6:
                perc_score = perc.predict(rf)
                # blend: 40% rule, 60% learned (shifts as data grows)
                blend = min(0.6, perc.n_samples / 20.0)
                score = rule_score * (1.0 - blend) + perc_score * blend
            else:
                score = rule_score

            module_scores[sig.name] = score
            if score < self.VETO_THRESHOLD and sig.value > 0.35:
                veto_reasons.append(f'{sig.name}:{score:.2f}')

        if not module_scores:
            return ResponseQualityDecision(1.0, False, [], {}, '')

        overall = float(np.mean(list(module_scores.values())))
        # Hard veto: analytical opener or language mismatch is always bad
        if rf[0] or rf[7]:
            veto_reasons.insert(0, 'analytical_mode' if rf[0] else 'language_mismatch')
        # Persona-specific character-break vetoes (independent of signal values)
        persona_vetoes = _persona_response_veto(persona_name, rf)
        veto_reasons.extend(persona_vetoes)

        veto = bool(veto_reasons) or overall < self.VETO_THRESHOLD
        dominant = min(module_scores, key=lambda k: module_scores[k]) if module_scores else ''

        return ResponseQualityDecision(
            overall_quality=overall,
            veto=veto,
            veto_reasons=veto_reasons,
            module_scores=module_scores,
            dominant_failure=dominant,
        )

    def record_correction(
        self,
        persona_name: str,
        signals: list[ModuleSignal],
        user_input: str,
        original_reply: str,
        corrected_reply: str,
    ) -> None:
        """
        Train all module perceptrons from a correction.
        original_reply → negative example (0)
        corrected_reply → positive example (1)
        """
        if not persona_name or not corrected_reply:
            return
        rf_bad  = _extract_response_features(user_input, original_reply) if original_reply else None
        rf_good = _extract_response_features(user_input, corrected_reply)
        for sig in signals:
            perc = self._get_perc(sig.module_id, persona_name)
            if rf_bad is not None and original_reply.strip() != corrected_reply.strip():
                perc.train(rf_bad, 0.0)
            perc.train(rf_good, 1.0)


# Module-level singleton (lazy init)
_persona_resp_evaluator: PersonaResponseEvaluator | None = None

def get_persona_response_evaluator(heads_dir: 'str | None' = None) -> PersonaResponseEvaluator:
    global _persona_resp_evaluator
    if _persona_resp_evaluator is None:
        from pathlib import Path as _Path
        if heads_dir is None:
            try:
                from agent_system.persona_engine import _head_dir as _hd
                heads_dir = str(_hd('').parent)
            except Exception:
                heads_dir = '.'
        _persona_resp_evaluator = PersonaResponseEvaluator(_Path(heads_dir))
    return _persona_resp_evaluator
