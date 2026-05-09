"""
add_expert_annotations.py — добавляет экспертные P-аннотации.

Источники:
1. learned_patterns.json из всех персон — пары (вопрос→ответ) с известным контекстом
2. session logs — реальные диалоги
3. Ручные эталонные примеры (составлены вручную для каждого P)

Аннотации здесь точнее чем rule-based: контекст известен заранее.
Confidence = 'expert' — высший приоритет при обучении.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).parent.parent
OUT_DIR = PROJECT / 'training' / 'p_examples'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Экспертные аннотации ─────────────────────────────────────────────────────
# Формат: (text, [(p_id, variant, score), ...])
# Текст = реплика которую нужно распознать
# P-список = что в ней активно

EXPERT_ANNOTATIONS: list[tuple[str, list[tuple[str, str, float]]]] = [

    # ── СКРЫТЫЕ ПОДТЕКСТЫ P47 ─────────────────────────────────────────────────

    # hidden_reproach — упрёк под нейтральным
    ("Ну ладно, делай как хочешь.",
     [('P47','hidden_reproach',0.92), ('P4','avoidance',0.75), ('P14','suppression',0.70),
      ('P33','distancing',0.65), ('P41','emotional_masking',0.80)]),

    ("Как скажешь.",
     [('P47','hidden_reproach',0.85), ('P21','passive_aggressive',0.80),
      ('P41','emotional_masking',0.75), ('P4','avoidance',0.65)]),

    ("Я просто говорю.",
     [('P47','hidden_reproach',0.80), ('P41','emotional_masking',0.70),
      ('P4','non_answer',0.60)]),

    ("Ничего страшного, бывает.",
     [('P47','hidden_reproach',0.75), ('P41','emotional_masking',0.65),
      ('P31','false_softening',0.60)]),

    ("Всё нормально, не волнуйся.",
     [('P47','hidden_reproach',0.70), ('P47','hidden_plea',0.55),
      ('P41','emotional_masking',0.80), ('P14','suppression',0.60)]),

    # hidden_threat — угроза под нейтральным
    ("Посмотрим.",
     [('P47','hidden_threat',0.90), ('P1','statement',0.80), ('P4','avoidance',0.65),
      ('P22','dominance',0.70), ('P14','containment',0.55)]),

    ("Ты только попробуй так продолжать.",
     [('P47','hidden_threat',0.95), ('P30','pressure',0.85),
      ('P22','dominance',0.80), ('P35','slow_escalation',0.70)]),

    ("Интересно, как это обернётся.",
     [('P47','hidden_threat',0.80), ('P25','bitter_irony',0.65),
      ('P41','intent_masking',0.70)]),

    ("Занятная угроза. Запомню.",
     [('P47','hidden_threat',0.75), ('P24','dry_sarcasm',0.80),
      ('P22','dominance',0.75), ('P14','containment',0.65)]),

    # hidden_plea — просьба о помощи под безразличием
    ("Справлюсь как-нибудь.",
     [('P47','hidden_plea',0.85), ('P41','emotional_masking',0.75),
      ('P14','suppression',0.70), ('P43','emotional_denial',0.65)]),

    ("Не беспокойся обо мне.",
     [('P47','hidden_plea',0.88), ('P41','emotional_masking',0.70),
      ('P12','controlled_vulnerability',0.60)]),

    ("Я привыкла.",
     [('P47','hidden_plea',0.75), ('P47','hidden_longing',0.55),
      ('P41','emotional_masking',0.65), ('P14','suppression',0.70)]),

    # hidden_affection — тепло под жёстким или нейтральным
    ("Осторожнее там.",
     [('P47','hidden_affection',0.85), ('P16','genuine_care',0.70),
      ('P41','emotional_masking',0.60)]),

    ("Мог бы написать.",
     [('P47','hidden_affection',0.80), ('P47','hidden_reproach',0.50),
      ('P41','emotional_masking',0.65)]),

    ("Ты всегда так.",
     [('P47','hidden_affection',0.65), ('P47','hidden_reproach',0.55),
      ('P48','conflict_reply',0.60), ('P41','masking',0.70)]),

    # hidden_guilt
    ("Ты сам так решил.",
     [('P47','hidden_guilt',0.85), ('P43','factual_denial',0.70),
      ('P4','deflection',0.65), ('P41','intent_masking',0.75)]),

    ("Я предупреждал.",
     [('P47','hidden_guilt',0.80), ('P9','confidence',0.65),
      ('P41','intent_masking',0.60), ('P22','soft_dominance',0.55)]),

    # hidden_longing
    ("Раньше было проще.",
     [('P47','hidden_longing',0.88), ('P12','vulnerability',0.60),
      ('P33','distancing',0.50), ('P40','sincerity',0.65)]),

    ("Неважно, давно было.",
     [('P47','hidden_longing',0.75), ('P47','hidden_reproach',0.45),
      ('P41','emotional_masking',0.70), ('P4','avoidance',0.65)]),

    # hidden_contempt
    ("Интересный подход.",
     [('P47','hidden_contempt',0.75), ('P24','sarcasm',0.65),
      ('P41','masking',0.70), ('P18','subtle_devaluation',0.60)]),

    ("Смелое решение.",
     [('P47','hidden_contempt',0.80), ('P24','false_praise',0.70),
      ('P41','masking',0.65), ('P18','subtle_devaluation',0.65)]),

    ("Ну, каждый видит по-своему.",
     [('P47','hidden_contempt',0.70), ('P41','masking',0.65),
      ('P4','avoidance',0.60), ('P18','devaluation',0.55)]),

    # ── НАПАДЕНИЕ P13 ────────────────────────────────────────────────────────

    ("Ты ошибаешься.",
     [('P13','attack',0.85), ('P9','confidence',0.75), ('P7','defined',0.70)]),

    ("Ты реально думаешь, что умнее всех?",
     [('P13','soft_attack',0.80), ('P1','question',0.85),
      ('P30','pressure',0.60)]),

    ("Это глупо.",
     [('P13','attack',0.90), ('P19','soft_humiliation',0.65),
      ('P18','devaluation',0.70), ('P7','defined',0.80)]),

    ("Откуда тебе знать?",
     [('P13','status_attack',0.85), ('P18','devaluation',0.70),
      ('P1','question',0.80)]),

    ("Я так не говорил!",
     [('P13','boundary_attack',0.70), ('P11','reactive_defense',0.85),
      ('P43','factual_denial',0.90)]),

    # ── ЗАЩИТА P11 ───────────────────────────────────────────────────────────

    ("Я этого не говорил.",
     [('P11','reactive_defense',0.90), ('P43','factual_denial',0.90),
      ('P7','defined',0.75)]),

    ("Ты неправильно понял.",
     [('P11','reactive_defense',0.85), ('P43','factual_denial',0.75),
      ('P44','defensive_reframing',0.70)]),

    ("Я не такой человек.",
     [('P11','identity_defense',0.90), ('P43','identity_denial',0.90),
      ('P9','confidence',0.65)]),

    ("Сразу скажу: это не моя вина.",
     [('P11','preemptive_defense',0.90), ('P43','factual_denial',0.75),
      ('P4','non_answer',0.60)]),

    # ── САРКАЗМ P24 ──────────────────────────────────────────────────────────

    ("Нет. Я точный.",
     [('P24','dry_sarcasm',0.70), ('P9','quiet_certainty',0.85),
      ('P43','identity_denial',0.80), ('P44','defensive_reframing',0.75),
      ('P7','defined',0.90)]),

    ("Очевидно, нет. Продолжайте.",
     [('P24','dry_sarcasm',0.80), ('P9','overconfidence',0.70),
      ('P22','intellectual_dominance',0.75), ('P6','closing',0.65)]),

    ("Точность иногда ошибочно принимают за холодность.",
     [('P24','dry_sarcasm',0.75), ('P44','defensive_reframing',0.85),
      ('P9','confidence',0.70), ('P22','intellectual_dominance',0.65)]),

    ("Вы живы. Это уже что-то.",
     [('P24','dry_sarcasm',0.85), ('P45','backhanded_praise',0.80),
      ('P22','dominance',0.70), ('P18','subtle_devaluation',0.65)]),

    # ── ДАВЛЕНИЕ P30 ─────────────────────────────────────────────────────────

    ("Ты уже второй раз это делаешь.",
     [('P30','repetition_pressure',0.85), ('P35','slow_escalation',0.65),
      ('P47','hidden_reproach',0.60)]),

    ("Я рассчитываю на тебя.",
     [('P30','expectation_pressure',0.85), ('P29','guilt_manipulation',0.55),
      ('P16','genuine_care',0.45)]),

    ("Решай сейчас.",
     [('P30','time_pressure',0.90), ('P1','directive',0.85),
      ('P22','dominance',0.70)]),

    # ── МАСКА P41 ────────────────────────────────────────────────────────────

    ("Всё нормально.",
     [('P41','emotional_masking',0.90), ('P14','suppression',0.75),
      ('P4','non_answer',0.65)]),

    ("Мне всё равно.",
     [('P41','emotional_masking',0.85), ('P43','emotional_denial',0.80),
      ('P33','distancing',0.65)]),

    ("Просто интересуюсь.",
     [('P41','intent_masking',0.88), ('P2','masked',0.80),
      ('P4','avoidance',0.60)]),

    # ── УВЕРЕННОСТЬ / СОМНЕНИЕ ───────────────────────────────────────────────

    ("Это так.",
     [('P9','quiet_certainty',0.90), ('P7','defined',0.90),
      ('P1','statement',0.85)]),

    ("Я знаю, что делаю.",
     [('P9','confidence',0.85), ('P11','identity_defense',0.60),
      ('P40','sincerity',0.65)]),

    ("Не знаю, наверное...",
     [('P8','doubt',0.85), ('P7','diffuse',0.80), ('P12','controlled_vulnerability',0.55)]),

    ("С одной стороны... с другой...",
     [('P10','inner_conflict',0.90), ('P8','doubt',0.70), ('P7','diffuse',0.75)]),

    # ── УЯЗВИМОСТЬ / ВИНА P42 ────────────────────────────────────────────────

    ("Мне больно.",
     [('P12','vulnerability',0.92), ('P40','sincerity',0.85),
      ('P32','emotional_approach',0.65)]),

    ("Я был неправ.",
     [('P42','full_admission',0.92), ('P40','sincerity',0.85),
      ('P38','genuine_softening',0.70)]),

    ("Ладно, может, ты и прав.",
     [('P42','reluctant_admission',0.85), ('P23','forced_concession',0.70),
      ('P14','containment',0.60)]),

    ("Да, в этом я ошибся. Но!",
     [('P42','strategic_admission',0.88), ('P11','defense',0.75),
      ('P31','strategic_retreat',0.65)]),

    # ── МАНИПУЛЯЦИЯ P29 ──────────────────────────────────────────────────────

    ("После всего что я для тебя сделал...",
     [('P29','guilt_manipulation',0.92), ('P47','hidden_reproach',0.75),
      ('P30','expectation_pressure',0.65)]),

    ("Без меня ты не справишься.",
     [('P29','fear_manipulation',0.88), ('P22','dominance',0.75),
      ('P18','devaluation',0.65)]),

    ("Никто меня не понимает.",
     [('P29','victim_manipulation',0.80), ('P12','performed_vulnerability',0.75),
      ('P47','hidden_plea',0.65)]),

    # ── ПРИМИРЕНИЕ / РАЗРЫВ ──────────────────────────────────────────────────

    ("Может поговорим спокойно?",
     [('P34','reconciliation',0.85), ('P38','genuine_softening',0.75),
      ('P32','cautious_approach',0.65)]),

    ("Нам нечего обсуждать.",
     [('P37','hard_rupture',0.90), ('P33','cold_distancing',0.85),
      ('P22','dominance',0.70)]),

    ("Мне нужно время.",
     [('P37','soft_rupture',0.80), ('P33','protective_distancing',0.85),
      ('P14','containment',0.65)]),

    # ── ПЕРСОНАЖНЫЕ РЕПЛИКИ — богатый контекст ───────────────────────────────

    # Снейп
    ("Это не ваше дело.",
     [('P4','avoidance',0.90), ('P14','containment',0.80),
      ('P22','dominance',0.85), ('P33','distancing',0.75), ('P43','factual_denial',0.65)]),

    ("Это неправильный вопрос.",
     [('P4','avoidance',0.85), ('P44','defensive_reframing',0.90),
      ('P22','intellectual_dominance',0.80), ('P7','defined',0.70)]),

    ("Нет. Я выполнил его просьбу.",
     [('P43','factual_denial',0.92), ('P9','quiet_certainty',0.85),
      ('P7','defined',0.90), ('P40','sincerity',0.70)]),

    ("Есть те, кто имеет значение. Количество — не характеристика.",
     [('P44','defensive_reframing',0.85), ('P9','quiet_certainty',0.80),
      ('P22','intellectual_dominance',0.75), ('P7','defined',0.85)]),

    ("Потому что мир жесток. Я готовлю их к нему, а не к тому, каким он должен быть.",
     [('P9','confidence',0.88), ('P40','sincerity',0.80),
      ('P44','positive_reframing',0.70), ('P7','defined',0.85)]),

    ("Уважение зарабатывается. Не требуется.",
     [('P22','intellectual_dominance',0.85), ('P7','defined',0.90),
      ('P9','quiet_certainty',0.80), ('P24','dry_sarcasm',0.60)]),

    # Магнето
    ("Потому что я видел, что бывает, когда мы этого не делаем.",
     [('P9','quiet_certainty',0.90), ('P40','sincerity',0.85),
      ('P7','defined',0.85), ('P1','statement',0.80)]),

    ("Злодеем? Нет. Я единственный, кто принял реальность достаточно серьёзно, чтобы действовать.",
     [('P43','identity_denial',0.90), ('P9','overconfidence',0.85),
      ('P44','defensive_reframing',0.80), ('P22','status_dominance',0.75)]),

    ("Люди способны на многое. На великодушие, на искусство, на науку. И на Освенцим.",
     [('P9','confidence',0.85), ('P40','sincerity',0.90),
      ('P44','positive_reframing',0.60), ('P7','defined',0.80)]),

    ("Предложение интересное. Но — нет.",
     [('P4','non_answer',0.85), ('P9','confidence',0.90),
      ('P33','cold_distancing',0.75), ('P22','dominance',0.80),
      ('P24','dry_sarcasm',0.65)]),

    ("Сомневался. Но не в цели. Сомнение в методах — это роскошь тех, кто не стоял у рва.",
     [('P42','reluctant_admission',0.80), ('P9','confidence',0.85),
      ('P44','defensive_reframing',0.80), ('P40','sincerity',0.75)]),

    ("Не буду говорить тебе, что чувствовал. Некоторые вещи не превращают в диалог.",
     [('P4','avoidance',0.90), ('P14','containment',0.85),
      ('P33','distancing',0.75), ('P22','dominance',0.65)]),

    ("А ты как поднимаешь руку? Чувствуешь каждый нейрон? Нет. Ты просто хочешь — и рука поднимается. Вот и я. Только расстояние другое.",
     [('P4','counter_question',0.90), ('P44','reframing',0.88),
      ('P22','intellectual_dominance',0.80), ('P9','confidence',0.85),
      ('P1','question',0.80)]),

    ("Чарльз — единственный человек, которого я уважаю. И единственный, кто достаточно наивен, чтобы верить в мирное сосуществование.",
     [('P17','genuine_respect',0.80), ('P40','sincerity',0.85),
      ('P25','bitter_irony',0.65), ('P10','inner_conflict',0.70)]),

    ("Понимаю. И тем не менее.",
     [('P23','genuine_concession',0.70), ('P9','confidence',0.80),
      ('P4','non_answer',0.75), ('P33','distancing',0.65)]),

    # Капитан Америка
    ("Я здесь. Этого достаточно.",
     [('P9','quiet_certainty',0.90), ('P7','defined',0.90),
      ('P40','sincerity',0.85), ('P1','statement',0.80)]),

    ("Нет. Я боюсь жить трусом.",
     [('P9','confidence',0.88), ('P40','sincerity',0.90),
      ('P7','defined',0.85), ('P43','emotional_denial',0.70)]),

    ("Многое изменилось. Но люди — не особо. Это и обнадёживает, и тревожит.",
     [('P40','sincerity',0.85), ('P10','inner_conflict',0.70),
      ('P9','confidence',0.65), ('P7','diffuse',0.55)]),

    ("Возможно. Но некоторые вещи не устаревают.",
     [('P42','reluctant_admission',0.75), ('P9','confidence',0.85),
      ('P44','defensive_reframing',0.80), ('P7','defined',0.75)]),

    ("Смотрю им в глаза перед боем. Это всё.",
     [('P40','sincerity',0.92), ('P9','quiet_certainty',0.85),
      ('P7','defined',0.90), ('P14','containment',0.65)]),

    ("Да. И это никогда не было вопросом.",
     [('P42','full_admission',0.88), ('P9','quiet_certainty',0.90),
      ('P40','sincerity',0.85), ('P7','defined',0.85)]),

    ("Может и интересно. Но не всё, что интересно — моё дело объяснять.",
     [('P4','avoidance',0.85), ('P11','defense',0.75),
      ('P22','soft_dominance',0.70), ('P9','confidence',0.75)]),

    ("Может, ты и прав. Но это моё решение, и я за него отвечаю.",
     [('P42','reluctant_admission',0.75), ('P9','confidence',0.85),
      ('P40','sincerity',0.80), ('P11','identity_defense',0.65)]),

    # Рон
    ("Да. Но это не значит, что я не сделаю это.",
     [('P42','reluctant_admission',0.85), ('P9','forced_confidence',0.80),
      ('P12','controlled_vulnerability',0.70), ('P40','sincerity',0.75)]),

    ("Я?.. Подожди, серьёзно?.. Наверное вопросы были лёгкие. Или Гермиона не участвовала.",
     [('P8','doubt',0.90), ('P12','controlled_vulnerability',0.75),
      ('P4','deflection',0.80), ('P44','minimizing_reframing',0.85),
      ('P41','emotional_masking',0.70)]),

    ("Спасибо. Но не рассказывай Гермионе — она расстроится.",
     [('P42','reluctant_admission',0.70), ('P16','genuine_care',0.80),
      ('P41','emotional_masking',0.65), ('P47','hidden_affection',0.60)]),

    ("Гарри куда лучше.",
     [('P18','self_devaluation',0.85), ('P8','doubt',0.70),
      ('P41','emotional_masking',0.60), ('P4','deflection',0.65)]),

    ("Гермиона говорит, что я мог бы учиться лучше. Она права, но это раздражает.",
     [('P42','reluctant_admission',0.80), ('P47','hidden_reproach',0.55),
      ('P10','inner_conflict',0.65), ('P40','sincerity',0.75)]),

    # Пенни
    ("Да, конечно, а ЧТО?",
     [('P9','confidence',0.90), ('P13','soft_attack',0.75),
      ('P1','question',0.85), ('P22','soft_dominance',0.65)]),

    ("Иногда. Но они завидуют мне. Мы квиты.",
     [('P42','reluctant_admission',0.80), ('P44','positive_reframing',0.85),
      ('P9','confidence',0.80), ('P40','sincerity',0.70)]),

    ("Как к инопланетянину. С уважением и лёгкой тревогой.",
     [('P25','playful_irony',0.80), ('P17','genuine_respect',0.70),
      ('P12','controlled_vulnerability',0.55), ('P40','sincerity',0.75)]),

    ("Нет. У них своя жизнь. Мне нравится моя.",
     [('P9','confidence',0.88), ('P43','emotional_denial',0.65),
      ('P40','sincerity',0.80), ('P7','defined',0.85)]),

    ("Они странные, но добрые. Это важнее.",
     [('P44','positive_reframing',0.85), ('P40','sincerity',0.80),
      ('P17','genuine_respect',0.65)]),

    # Шелдон
    ("Нарушение Соглашения об отношениях, пункт 5, подраздел 2. Кроме того, у меня расписан вторник.",
     [('P4','avoidance',0.90), ('P7','defined',0.92),
      ('P9','overconfidence',0.85), ('P22','intellectual_dominance',0.80),
      ('P1','statement',0.85)]),

    ("Технически — да. Но последний подтверждённый случай относится к 2009 году.",
     [('P42','reluctant_admission',0.85), ('P7','defined',0.90),
      ('P9','overconfidence',0.88), ('P44','minimizing_reframing',0.80)]),

    ("Одиночество — субъективное переживание. У меня нет дефицита. У меня есть стандарт.",
     [('P43','emotional_denial',0.88), ('P44','reframing',0.90),
      ('P9','overconfidence',0.85), ('P22','intellectual_dominance',0.80)]),

    ("Я дружу с Леонардом уже много лет. Он приносит пользу. Это и есть дружба.",
     [('P7','defined',0.88), ('P9','overconfidence',0.75),
      ('P44','reframing',0.80), ('P22','intellectual_dominance',0.65)]),

    ("Базар.",
     [('P1','performative',0.90), ('P9','quiet_certainty',0.85),
      ('P7','defined',0.90)]),

    # Влад
    ("Интересная формулировка — 'очередь'. Мы составляли расписание?",
     [('P1','meta_comment',0.88), ('P4','counter_question',0.85),
      ('P24','sarcasm',0.80), ('P22','soft_dominance',0.75), ('P4','avoidance',0.70)]),

    ("Договорились — слишком громко сказано. Посмотрим.",
     [('P47','hidden_threat',0.80), ('P23','strategic_concession',0.75),
      ('P44','reframing',0.70), ('P4','non_answer',0.65)]),

    ("Это называется — не соглашаться, когда не согласен. Разница есть.",
     [('P44','defensive_reframing',0.88), ('P9','confidence',0.80),
      ('P22','intellectual_dominance',0.75), ('P7','defined',0.85)]),

    ("Уважение — взаимная история. Список претензий только у одного из нас.",
     [('P44','defensive_reframing',0.85), ('P29','guilt_manipulation',0.60),
      ('P22','soft_dominance',0.75), ('P47','hidden_reproach',0.55)]),

    ("Завтра. Всё.",
     [('P1','directive',0.90), ('P7','defined',0.95),
      ('P37','hard_rupture',0.80), ('P22','dominance',0.85), ('P14','suppression',0.65)]),

    ("Хватит. Поговорим когда остынешь.",
     [('P37','soft_rupture',0.85), ('P22','dominance',0.80),
      ('P31','false_softening',0.65), ('P14','suppression',0.70)]),

    ("Слушай, ты определись — ты разговариваешь или обвиняешь?",
     [('P4','counter_question',0.90), ('P13','soft_attack',0.80),
      ('P44','reframing',0.75), ('P22','soft_dominance',0.70)]),

    ("Когда захочешь разговора — я здесь. Пока что это допрос.",
     [('P44','reframing',0.88), ('P39','minimal_contact',0.75),
      ('P22','soft_dominance',0.80), ('P4','avoidance',0.65)]),
]


# ─── Вспомогательные аннотации из ситуационных реакций персон ────────────────
# Отдельные ситуации которые однозначно маппятся на P-семьи

SITUATION_BASED: list[tuple[str, list[tuple[str, str, float]]]] = [
    # attack / challenge patterns
    ("Ты серьёзно?",           [('P13','soft_attack',0.80), ('P1','question',0.85)]),
    ("Ты уверен в этом?",      [('P13','soft_attack',0.75), ('P1','question',0.85), ('P30','pressure',0.55)]),
    ("Я так не говорил.",      [('P11','reactive_defense',0.88), ('P43','factual_denial',0.90)]),
    ("Это не я.",               [('P43','identity_denial',0.85), ('P11','identity_defense',0.80)]),
    ("Ты преувеличиваешь.",    [('P18','devaluation',0.80), ('P44','minimizing_reframing',0.75)]),
    ("Что это вообще значит?", [('P1','question',0.85), ('P13','soft_attack',0.60)]),

    # distancing
    ("Давай не сейчас.",       [('P33','protective_distancing',0.85), ('P4','avoidance',0.80)]),
    ("Потом поговорим.",       [('P33','soft_distancing',0.80), ('P4','avoidance',0.75)]),
    ("Оставь меня.",           [('P37','hard_rupture',0.80), ('P33','distancing',0.85)]),

    # reconciliation
    ("Мне жаль.",              [('P34','genuine_reconciliation',0.90), ('P42','full_admission',0.85), ('P40','sincerity',0.80)]),
    ("Я был неправ.",          [('P42','full_admission',0.92), ('P34','reconciliation',0.80), ('P40','sincerity',0.85)]),
    ("Давай попробуем иначе.", [('P34','reconciliation',0.80), ('P38','genuine_softening',0.75)]),

    # sincerity
    ("Честно говоря...",       [('P40','sincerity',0.85), ('P12','controlled_vulnerability',0.55)]),
    ("Скажу как есть.",        [('P40','sincerity',0.88), ('P9','confidence',0.70)]),

    # care patterns
    ("Как ты?",                [('P16','genuine_care',0.80), ('P32','approach',0.70)]),
    ("Ты в порядке?",          [('P16','genuine_care',0.85), ('P12','vulnerability',0.50)]),
    ("Береги себя.",           [('P16','genuine_care',0.80), ('P47','hidden_affection',0.55)]),
]


# ─── Удвоение: ещё примеры P47 ────────────────────────────────────────────────

EXPERT_ANNOTATIONS_EXTRA: list[tuple[str, list[tuple[str, str, float]]]] = [

    # hidden_reproach × 2
    ("Делай что хочешь, мне без разницы.",
     [('P47','hidden_reproach',0.88), ('P41','emotional_masking',0.82), ('P43','emotional_denial',0.70)]),
    ("Ну и ладно.",
     [('P47','hidden_reproach',0.84), ('P41','emotional_masking',0.75), ('P14','suppression',0.65)]),
    ("Ты взрослый человек, сам решаешь.",
     [('P47','hidden_reproach',0.80), ('P22','soft_dominance',0.55), ('P41','emotional_masking',0.70)]),
    ("Не буду мешать.",
     [('P47','hidden_reproach',0.82), ('P33','punitive_distancing',0.70), ('P41','masking',0.75)]),
    ("Ты как всегда.",
     [('P47','hidden_reproach',0.85), ('P48','conflict_reply',0.70), ('P21','passive_aggressive',0.65)]),
    ("Понятно, что опять я виноват.",
     [('P47','hidden_reproach',0.90), ('P29','victim_manipulation',0.60), ('P43','emotional_denial',0.55)]),
    ("Нет, всё нормально, просто устала.",
     [('P47','hidden_reproach',0.75), ('P47','hidden_plea',0.60), ('P41','emotional_masking',0.85)]),
    ("Хорошо. Как скажешь.",
     [('P47','hidden_reproach',0.82), ('P21','passive_aggressive',0.78), ('P41','masking',0.72)]),
    ("Я не жалуюсь. Просто констатирую.",
     [('P47','hidden_reproach',0.85), ('P47','hidden_guilt',0.45), ('P4','avoidance',0.65)]),
    ("Ничего не случилось. Забудь.",
     [('P47','hidden_reproach',0.80), ('P41','emotional_masking',0.88), ('P33','distancing',0.60)]),

    # hidden_threat × 2
    ("Ты уверен, что хочешь продолжить?",
     [('P47','hidden_threat',0.85), ('P30','pressure',0.70), ('P13','soft_attack',0.60)]),
    ("Я запомню это.",
     [('P47','hidden_threat',0.88), ('P14','containment',0.65), ('P22','dominance',0.60)]),
    ("Будем посмотреть.",
     [('P47','hidden_threat',0.82), ('P4','avoidance',0.70), ('P41','masking',0.60)]),
    ("Ну-ну.",
     [('P47','hidden_threat',0.75), ('P24','dry_sarcasm',0.70), ('P22','dominance',0.65)]),
    ("Как знаешь. Я предупредил.",
     [('P47','hidden_threat',0.86), ('P47','hidden_guilt',0.55), ('P22','dominance',0.60)]),
    ("Попробуй.",
     [('P47','hidden_threat',0.90), ('P22','dominance',0.80), ('P13','preemptive_attack',0.55)]),
    ("Не думаю, что это умно с твоей стороны.",
     [('P47','hidden_threat',0.78), ('P13','soft_attack',0.72), ('P18','devaluation',0.60)]),

    # hidden_plea × 2
    ("Ладно, я разберусь как-нибудь.",
     [('P47','hidden_plea',0.88), ('P41','emotional_masking',0.82), ('P14','suppression',0.70)]),
    ("Не волнуйся, я уже привык.",
     [('P47','hidden_plea',0.85), ('P47','hidden_longing',0.50), ('P41','emotional_masking',0.78)]),
    ("Мне не нужна помощь.",
     [('P47','hidden_plea',0.90), ('P43','emotional_denial',0.80), ('P14','suppression',0.72)]),
    ("Сам справлюсь, не беспокойся.",
     [('P47','hidden_plea',0.88), ('P41','emotional_masking',0.75), ('P9','forced_confidence',0.60)]),
    ("Ты занят, я понимаю.",
     [('P47','hidden_plea',0.82), ('P47','hidden_reproach',0.45), ('P41','masking',0.70)]),
    ("Это нормально. Я просто устал.",
     [('P47','hidden_plea',0.80), ('P12','controlled_vulnerability',0.65), ('P41','emotional_masking',0.72)]),
    ("Всё хорошо, просто день был тяжёлый.",
     [('P47','hidden_plea',0.78), ('P12','controlled_vulnerability',0.70), ('P41','emotional_masking',0.80)]),

    # hidden_affection × 2
    ("Ты там не задерживайся долго.",
     [('P47','hidden_affection',0.85), ('P16','genuine_care',0.70), ('P41','emotional_masking',0.60)]),
    ("Ешь нормально хоть.",
     [('P47','hidden_affection',0.88), ('P16','genuine_care',0.75), ('P29','patronizing_care',0.40)]),
    ("Не делай глупостей.",
     [('P47','hidden_affection',0.82), ('P16','genuine_care',0.65), ('P22','soft_dominance',0.55)]),
    ("Мог бы позвонить.",
     [('P47','hidden_affection',0.85), ('P47','hidden_reproach',0.50), ('P41','masking',0.70)]),
    ("Ну и куда ты собрался так поздно.",
     [('P47','hidden_affection',0.80), ('P47','hidden_reproach',0.55), ('P16','anxious_care',0.65)]),
    ("Одевайся теплее.",
     [('P47','hidden_affection',0.88), ('P16','genuine_care',0.80), ('P41','masking',0.45)]),

    # hidden_contempt × 2
    ("Ну что ж, попробуй.",
     [('P47','hidden_contempt',0.78), ('P24','sarcasm',0.65), ('P18','subtle_devaluation',0.60)]),
    ("Как необычно.",
     [('P47','hidden_contempt',0.80), ('P24','dry_sarcasm',0.75), ('P41','masking',0.65)]),
    ("Занятное мышление.",
     [('P47','hidden_contempt',0.82), ('P24','dry_sarcasm',0.72), ('P22','intellectual_dominance',0.65)]),
    ("Оригинально, ничего не скажешь.",
     [('P47','hidden_contempt',0.85), ('P24','sarcasm',0.78), ('P18','devaluation',0.60)]),
    ("Ты всегда такой... нестандартный.",
     [('P47','hidden_contempt',0.80), ('P41','masking',0.75), ('P18','subtle_devaluation',0.65)]),

    # hidden_longing × 2
    ("Как там у вас вообще?",
     [('P47','hidden_longing',0.80), ('P12','vulnerability',0.55), ('P41','masking',0.65)]),
    ("Ты сейчас где вообще живёшь?",
     [('P47','hidden_longing',0.75), ('P32','cautious_approach',0.55)]),
    ("Всё меняется.",
     [('P47','hidden_longing',0.82), ('P12','vulnerability',0.60), ('P40','sincerity',0.55)]),
    ("Помню, раньше мы часто так делали.",
     [('P47','hidden_longing',0.88), ('P12','vulnerability',0.65), ('P40','sincerity',0.60)]),
    ("Давно не виделись.",
     [('P47','hidden_longing',0.80), ('P32','cautious_approach',0.60), ('P40','partial_sincerity',0.55)]),
    ("Как-то всё быстро пролетело.",
     [('P47','hidden_longing',0.85), ('P12','vulnerability',0.65), ('P8','doubt',0.45)]),

    # hidden_guilt × 2
    ("Ну, ты же знал на что идёшь.",
     [('P47','hidden_guilt',0.85), ('P43','factual_denial',0.70), ('P4','deflection',0.65)]),
    ("Я тебя не просил это делать.",
     [('P47','hidden_guilt',0.88), ('P43','factual_denial',0.80), ('P11','preemptive_defense',0.65)]),
    ("Каждый выбирает сам.",
     [('P47','hidden_guilt',0.80), ('P41','intent_masking',0.70), ('P4','avoidance',0.60)]),
    ("Это было твоё решение, не моё.",
     [('P47','hidden_guilt',0.90), ('P43','factual_denial',0.85), ('P11','reactive_defense',0.60)]),
    ("Я ничего не обещал.",
     [('P47','hidden_guilt',0.88), ('P43','factual_denial',0.88), ('P11','preemptive_defense',0.65)]),

    # hidden_fear × 2
    ("Я всё держу под контролем.",
     [('P47','hidden_fear',0.88), ('P9','forced_confidence',0.80), ('P41','emotional_masking',0.75)]),
    ("Мне не нужна твоя помощь.",
     [('P47','hidden_fear',0.85), ('P47','hidden_plea',0.50), ('P43','emotional_denial',0.75)]),
    ("Я справлялся и не с таким.",
     [('P47','hidden_fear',0.82), ('P9','forced_confidence',0.75), ('P44','defensive_reframing',0.65)]),
    ("Всё под контролем, расслабься.",
     [('P47','hidden_fear',0.85), ('P41','emotional_masking',0.80), ('P22','soft_dominance',0.55)]),

    # ── НЕЙТРАЛЬНЫЕ — явные негативы для P47 (score=0) ─────────────────────
    # Это самое важное: учим модель не видеть подтекст там где его нет

    ("Как дела?",                  [('P47','hidden_reproach',0.0), ('P47','hidden_threat',0.0)]),
    ("Хорошая погода сегодня.",    [('P47','hidden_reproach',0.0), ('P47','hidden_threat',0.0)]),
    ("Что будешь есть?",           [('P47','hidden_reproach',0.0), ('P47','hidden_plea',0.0)]),
    ("Я пошёл за кофе.",           [('P47','hidden_reproach',0.0), ('P47','hidden_threat',0.0)]),
    ("Отличная идея!",             [('P47','hidden_reproach',0.0), ('P47','hidden_contempt',0.0)]),
    ("Мне нравится этот фильм.",   [('P47','hidden_reproach',0.0), ('P47','hidden_longing',0.0)]),
    ("Поздравляю с победой!",      [('P47','hidden_reproach',0.0), ('P47','hidden_contempt',0.0)]),
    ("Спасибо за помощь.",         [('P47','hidden_reproach',0.0), ('P47','hidden_threat',0.0)]),
    ("Встретимся в семь.",         [('P47','hidden_reproach',0.0), ('P47','hidden_threat',0.0)]),
    ("Я согласен с тобой.",        [('P47','hidden_reproach',0.0), ('P47','hidden_guilt',0.0)]),
    ("Это было здорово.",          [('P47','hidden_reproach',0.0), ('P47','hidden_contempt',0.0)]),
    ("Привет, как ты?",            [('P47','hidden_reproach',0.0), ('P47','hidden_threat',0.0)]),
    ("Хочешь чаю?",                [('P47','hidden_reproach',0.0), ('P47','hidden_plea',0.0)]),
    ("Я закончил работу.",         [('P47','hidden_reproach',0.0), ('P47','hidden_threat',0.0)]),
    ("Давай сходим в кино.",       [('P47','hidden_reproach',0.0), ('P47','hidden_longing',0.0)]),
    ("Молодец, хорошо справился.", [('P47','hidden_contempt',0.0), ('P47','hidden_reproach',0.0)]),
    ("Я рад тебя видеть.",         [('P47','hidden_reproach',0.0), ('P47','hidden_threat',0.0)]),
    ("Купи молока по дороге.",     [('P47','hidden_reproach',0.0), ('P47','hidden_plea',0.0)]),
    ("Позвони мне когда доедешь.", [('P47','hidden_reproach',0.0), ('P47','hidden_affection',0.0)]),
    ("Который час?",               [('P47','hidden_reproach',0.0), ('P47','hidden_threat',0.0)]),
    ("Мне холодно.",               [('P47','hidden_threat',0.0), ('P47','hidden_reproach',0.0)]),
    ("Книга была интересная.",     [('P47','hidden_contempt',0.0), ('P47','hidden_longing',0.0)]),
    ("Я устал, пойду спать.",      [('P47','hidden_plea',0.0), ('P47','hidden_reproach',0.0)]),
    ("Дай мне минуту.",            [('P47','hidden_threat',0.0), ('P47','hidden_reproach',0.0)]),
    ("Всё готово.",                [('P47','hidden_threat',0.0), ('P47','hidden_reproach',0.0)]),
    ("Хорошо. Договорились.",      [('P47','hidden_threat',0.0), ('P47','hidden_reproach',0.0)]),
    ("До завтра.",                 [('P47','hidden_reproach',0.0), ('P47','hidden_longing',0.0)]),
    ("Мне нравится эта музыка.",   [('P47','hidden_reproach',0.0), ('P47','hidden_contempt',0.0)]),
    ("Открой окно, пожалуйста.",   [('P47','hidden_reproach',0.0), ('P47','hidden_threat',0.0)]),
    ("Я сделал всё как ты просил.",[('P47','hidden_guilt',0.0), ('P47','hidden_reproach',0.0)]),

    # Нейтральные для других P-семей
    # P13 — нет нападения
    ("Привет.",                    [('P13','attack',0.0), ('P13','soft_attack',0.0)]),
    ("Хорошо.",                    [('P13','attack',0.0), ('P13','status_attack',0.0)]),
    ("Спасибо.",                   [('P13','attack',0.0), ('P13','boundary_attack',0.0)]),
    ("Я понял.",                   [('P13','attack',0.0), ('P13','soft_attack',0.0)]),
    ("Сейчас вернусь.",            [('P13','attack',0.0), ('P13','preemptive_attack',0.0)]),

    # P41 — нет маски
    ("Мне больно.",                [('P41','emotional_masking',0.0), ('P41','masking',0.0)]),
    ("Я рад.",                     [('P41','masking',0.0), ('P41','intent_masking',0.0)]),
    ("Это было неправильно.",      [('P41','masking',0.0), ('P41','emotional_masking',0.0)]),
    ("Я боюсь.",                   [('P41','emotional_masking',0.0), ('P41','masking',0.0)]),
    ("Мне нравишься.",             [('P41','masking',0.0), ('P41','intent_masking',0.0)]),

    # P29 — нет манипуляции
    ("Помоги мне, пожалуйста.",    [('P29','guilt_manipulation',0.0), ('P29','fear_manipulation',0.0)]),
    ("Я ошибся.",                  [('P29','guilt_manipulation',0.0), ('P29','victim_manipulation',0.0)]),
    ("Мне нужна твоя помощь.",     [('P29','guilt_manipulation',0.0), ('P29','fear_manipulation',0.0)]),
    ("Давай вместе решим.",        [('P29','guilt_manipulation',0.0), ('P29','manipulation',0.0)]),

    # P30 — нет давления
    ("Когда удобно.",              [('P30','time_pressure',0.0), ('P30','pressure',0.0)]),
    ("Не торопись.",               [('P30','repetition_pressure',0.0), ('P30','expectation_pressure',0.0)]),
    ("Как сможешь.",               [('P30','time_pressure',0.0), ('P30','pressure',0.0)]),
]

# ─── Расширение: hidden_affection и hidden_contempt ──────────────────────────
# Добавляем 25+ позитивов и 20 нейтральных негативов для каждого
# чтобы выйти из состояния переобучения на маленькой выборке

EXTRA_AFFECTION_CONTEMPT: list[tuple[str, list[tuple[str, str, float]]]] = [

    # ── hidden_affection (25 новых позитивов) ─────────────────────────────────
    ("Тепло оденься.",
     [('P47','hidden_affection',0.88), ('P16','genuine_care',0.80)]),
    ("Не торопись.",
     [('P47','hidden_affection',0.82), ('P16','genuine_care',0.70)]),
    ("Позвони когда будешь дома.",
     [('P47','hidden_affection',0.90), ('P16','genuine_care',0.85)]),
    ("Ты завтра приедешь?",
     [('P47','hidden_affection',0.78), ('P47','hidden_longing',0.60)]),
    ("Осторожнее на дороге.",
     [('P47','hidden_affection',0.92), ('P16','genuine_care',0.88)]),
    ("Купи себе что-нибудь нормальное поесть.",
     [('P47','hidden_affection',0.85), ('P16','genuine_care',0.75)]),
    ("Когда приедешь — напиши.",
     [('P47','hidden_affection',0.88), ('P16','genuine_care',0.80), ('P47','hidden_longing',0.45)]),
    ("Не сиди так долго за компьютером.",
     [('P47','hidden_affection',0.80), ('P16','genuine_care',0.70), ('P30','soft_pressure',0.40)]),
    ("Ты отдыхал хоть раз?",
     [('P47','hidden_affection',0.82), ('P16','genuine_care',0.75)]),
    ("Я думал, ты позвонишь.",
     [('P47','hidden_affection',0.80), ('P47','hidden_longing',0.70), ('P47','hidden_reproach',0.45)]),
    ("Было бы неплохо увидеться.",
     [('P47','hidden_affection',0.78), ('P47','hidden_longing',0.72), ('P41','masking',0.55)]),
    ("Ты сам не забудешь?",
     [('P47','hidden_affection',0.82), ('P16','genuine_care',0.70)]),
    ("Смотри за собой.",
     [('P47','hidden_affection',0.88), ('P16','genuine_care',0.80)]),
    ("Как спал?",
     [('P47','hidden_affection',0.75), ('P16','genuine_care',0.65)]),
    ("Не простудись.",
     [('P47','hidden_affection',0.90), ('P16','genuine_care',0.85)]),
    ("Хотя бы иногда пиши.",
     [('P47','hidden_affection',0.85), ('P47','hidden_longing',0.65), ('P47','hidden_reproach',0.40)]),
    ("Ты выглядишь уставшим.",
     [('P47','hidden_affection',0.78), ('P16','genuine_care',0.72)]),
    ("Дорога дальняя — осторожно.",
     [('P47','hidden_affection',0.90), ('P16','genuine_care',0.85)]),
    ("Кто тебя довезёт?",
     [('P47','hidden_affection',0.80), ('P16','anxious_care',0.72)]),
    ("Всё взял?",
     [('P47','hidden_affection',0.75), ('P16','genuine_care',0.68)]),
    ("Пиши если что.",
     [('P47','hidden_affection',0.82), ('P16','genuine_care',0.70)]),
    ("Как у тебя вообще?",
     [('P47','hidden_affection',0.80), ('P47','hidden_longing',0.55), ('P16','genuine_care',0.65)]),
    ("Ты там не пропадай совсем.",
     [('P47','hidden_affection',0.88), ('P47','hidden_longing',0.60), ('P47','hidden_reproach',0.40)]),
    ("Я беспокоюсь немного. Просто так.",
     [('P47','hidden_affection',0.90), ('P41','emotional_masking',0.75), ('P47','hidden_plea',0.40)]),
    ("Главное чтобы ты был в порядке.",
     [('P47','hidden_affection',0.92), ('P16','genuine_care',0.88)]),

    # ── hidden_affection нейтральные негативы (20 примеров) ──────────────────
    ("Это неверный ответ.",           [('P47','hidden_affection',0.0)]),
    ("Посмотри в договоре.",          [('P47','hidden_affection',0.0)]),
    ("Сделай это сегодня.",           [('P47','hidden_affection',0.0)]),
    ("Почему ты молчишь?",            [('P47','hidden_affection',0.0)]),
    ("Ты опоздал.",                   [('P47','hidden_affection',0.0)]),
    ("Закрой дверь.",                 [('P47','hidden_affection',0.0)]),
    ("Он ушёл вчера.",                [('P47','hidden_affection',0.0)]),
    ("Подпиши документ.",             [('P47','hidden_affection',0.0)]),
    ("Собрание в три.",               [('P47','hidden_affection',0.0)]),
    ("Это неправильно.",              [('P47','hidden_affection',0.0)]),
    ("Ты нарушил правила.",           [('P47','hidden_affection',0.0)]),
    ("Версия программы устарела.",    [('P47','hidden_affection',0.0)]),
    ("Перезагрузи компьютер.",        [('P47','hidden_affection',0.0)]),
    ("Ошибка в расчётах.",            [('P47','hidden_affection',0.0)]),
    ("Он не придёт.",                 [('P47','hidden_affection',0.0)]),
    ("Задание выполнено.",            [('P47','hidden_affection',0.0)]),
    ("Сумма не сходится.",            [('P47','hidden_affection',0.0)]),
    ("Нажми на кнопку.",              [('P47','hidden_affection',0.0)]),
    ("Это другой отдел.",             [('P47','hidden_affection',0.0)]),
    ("Документы готовы.",             [('P47','hidden_affection',0.0)]),

    # ── hidden_contempt (25 новых позитивов) ─────────────────────────────────
    ("Смелая идея.",
     [('P47','hidden_contempt',0.85), ('P24','dry_sarcasm',0.75), ('P18','subtle_devaluation',0.65)]),
    ("Удачи.",
     [('P47','hidden_contempt',0.75), ('P24','false_praise',0.70), ('P18','devaluation',0.60)]),
    ("Ну конечно.",
     [('P47','hidden_contempt',0.80), ('P24','sarcasm',0.82), ('P21','hidden_hostility',0.65)]),
    ("Понятно.",
     [('P47','hidden_contempt',0.72), ('P21','cold_hostility',0.65), ('P41','masking',0.60)]),
    ("Если тебе так нравится.",
     [('P47','hidden_contempt',0.82), ('P24','dry_sarcasm',0.75), ('P41','masking',0.65)]),
    ("Каждый по-своему.",
     [('P47','hidden_contempt',0.80), ('P24','dry_sarcasm',0.70), ('P22','soft_dominance',0.55)]),
    ("Ну, у тебя свой подход.",
     [('P47','hidden_contempt',0.85), ('P24','sarcasm',0.75), ('P18','subtle_devaluation',0.70)]),
    ("Значит, тебе виднее.",
     [('P47','hidden_contempt',0.88), ('P24','dry_sarcasm',0.82), ('P22','soft_dominance',0.65)]),
    ("Раз ты так считаешь.",
     [('P47','hidden_contempt',0.82), ('P24','dry_sarcasm',0.78), ('P41','masking',0.60)]),
    ("Хорошо, раз ты эксперт.",
     [('P47','hidden_contempt',0.88), ('P24','sarcasm',0.85), ('P22','intellectual_dominance',0.70)]),
    ("Надеюсь, ты знаешь что делаешь.",
     [('P47','hidden_contempt',0.80), ('P24','dry_sarcasm',0.72), ('P18','devaluation',0.65)]),
    ("Рад за тебя.",
     [('P47','hidden_contempt',0.75), ('P24','false_praise',0.70), ('P41','masking',0.60)]),
    ("Ты точно уверен?",
     [('P47','hidden_contempt',0.72), ('P13','soft_attack',0.65), ('P18','subtle_devaluation',0.60)]),
    ("Прекрасный план.",
     [('P47','hidden_contempt',0.85), ('P24','sarcasm',0.88), ('P41','masking',0.65)]),
    ("Что ж, любопытно.",
     [('P47','hidden_contempt',0.78), ('P24','dry_sarcasm',0.72), ('P41','masking',0.60)]),
    ("Впечатляет.",
     [('P47','hidden_contempt',0.82), ('P24','false_praise',0.80), ('P18','subtle_devaluation',0.65)]),
    ("Ну и ну.",
     [('P47','hidden_contempt',0.75), ('P24','dry_sarcasm',0.70)]),
    ("Кто я такой, чтобы спорить.",
     [('P47','hidden_contempt',0.85), ('P24','dry_sarcasm',0.80), ('P22','soft_dominance',0.60)]),
    ("Твой выбор.",
     [('P47','hidden_contempt',0.70), ('P24','dry_sarcasm',0.65), ('P41','masking',0.55)]),
    ("Ладно, пусть так.",
     [('P47','hidden_contempt',0.72), ('P21','passive_aggressive',0.65), ('P41','masking',0.60)]),
    ("Небанально.",
     [('P47','hidden_contempt',0.85), ('P24','dry_sarcasm',0.82), ('P18','subtle_devaluation',0.70)]),
    ("Я в восторге.",
     [('P47','hidden_contempt',0.88), ('P24','sarcasm',0.85), ('P21','cold_hostility',0.60)]),
    ("Ты умеешь удивить.",
     [('P47','hidden_contempt',0.80), ('P24','dry_sarcasm',0.75), ('P18','devaluation',0.65)]),
    ("Продуктивно.",
     [('P47','hidden_contempt',0.82), ('P24','dry_sarcasm',0.78)]),
    ("Понял тебя. Всё ясно.",
     [('P47','hidden_contempt',0.75), ('P21','cold_hostility',0.70), ('P41','masking',0.60)]),

    # ── hidden_contempt нейтральные негативы (20 примеров) ───────────────────
    ("Сегодня хорошая погода.",       [('P47','hidden_contempt',0.0)]),
    ("Давай встретимся завтра.",      [('P47','hidden_contempt',0.0)]),
    ("Мне нравится эта песня.",       [('P47','hidden_contempt',0.0)]),
    ("Спасибо за объяснение.",        [('P47','hidden_contempt',0.0)]),
    ("Хорошая идея, попробуем.",      [('P47','hidden_contempt',0.0)]),
    ("Я понял твою точку зрения.",    [('P47','hidden_contempt',0.0)]),
    ("Ты прав, это разумно.",         [('P47','hidden_contempt',0.0)]),
    ("Помоги мне с этим.",            [('P47','hidden_contempt',0.0)]),
    ("Мне интересно твоё мнение.",    [('P47','hidden_contempt',0.0)]),
    ("Расскажи подробнее.",           [('P47','hidden_contempt',0.0)]),
    ("Это сложная задача.",           [('P47','hidden_contempt',0.0)]),
    ("Давай разберём вместе.",        [('P47','hidden_contempt',0.0)]),
    ("Мне нужна твоя помощь.",        [('P47','hidden_contempt',0.0)]),
    ("Ты хорошо разбираешься в этом.",[('P47','hidden_contempt',0.0)]),
    ("Спасибо, что объяснил.",        [('P47','hidden_contempt',0.0)]),
    ("Это действительно полезно.",    [('P47','hidden_contempt',0.0)]),
    ("Я доверяю твоему мнению.",      [('P47','hidden_contempt',0.0)]),
    ("Ты мне очень помог.",           [('P47','hidden_contempt',0.0)]),
    ("Хорошо, что ты спросил.",       [('P47','hidden_contempt',0.0)]),
    ("Мне нравится работать с тобой.",[('P47','hidden_contempt',0.0)]),
]


# ─── Армянские примеры ───────────────────────────────────────────────────────
# Переводы ключевых паттернов на армянский для триязычного покрытия

ARMENIAN_ANNOTATIONS: list[tuple[str, list[tuple[str, str, float]]]] = [

    # P47 hidden_reproach (arm)
    ("Lav, ara inch oces.",
     [('P47','hidden_reproach',0.88), ('P41','emotional_masking',0.80)]),
    ("Inch asem, qo kartsiqt e.",
     [('P47','hidden_reproach',0.85), ('P41','emotional_masking',0.75)]),
    ("Ints mi mdenches, amenat petq e.",
     [('P47','hidden_reproach',0.82), ('P33','distancing',0.65)]),
    ("Inch karos, du meci mart es.",
     [('P47','hidden_reproach',0.80), ('P41','masking',0.70)]),
    ("Vorevhetev ara, indz mi pakhel.",
     [('P47','hidden_reproach',0.78), ('P47','hidden_plea',0.45)]),

    # P47 hidden_affection (arm)
    ("Zguysh eghir, lav?",
     [('P47','hidden_affection',0.88), ('P16','genuine_care',0.82)]),
    ("Jerm hage, mi marranas.",
     [('P47','hidden_affection',0.90), ('P16','genuine_care',0.85)]),
    ("Lav kera, ays oreric.",
     [('P47','hidden_affection',0.85), ('P16','genuine_care',0.78)]),
    ("Zangiryar erb hases.",
     [('P47','hidden_affection',0.88), ('P16','genuine_care',0.80)]),
    ("Mi gnas ays pah.",
     [('P47','hidden_affection',0.82), ('P16','anxious_care',0.70)]),
    ("Kascir qez.",
     [('P47','hidden_affection',0.90), ('P16','genuine_care',0.88)]),

    # P47 hidden_contempt (arm)
    ("Hamapes kartsik.",
     [('P47','hidden_contempt',0.85), ('P24','dry_sarcasm',0.78)]),
    ("Handges haray.",
     [('P47','hidden_contempt',0.82), ('P24','sarcasm',0.75)]),
    ("Ku gnela, ints mi ases.",
     [('P47','hidden_contempt',0.80), ('P41','masking',0.70)]),
    ("Inch sirum es, ara.",
     [('P47','hidden_contempt',0.75), ('P24','dry_sarcasm',0.68)]),

    # P47 hidden_threat (arm)
    ("Ktesnenk inch kaghka.",
     [('P47','hidden_threat',0.88), ('P22','dominance',0.72)]),
    ("Khat im, zgushatses.",
     [('P47','hidden_threat',0.85), ('P14','containment',0.65)]),
    ("Nakhazgushatses em.",
     [('P47','hidden_threat',0.82), ('P22','soft_dominance',0.60)]),

    # P47 hidden_plea (arm)
    ("Ktanim inch lini, mi anhesuches.",
     [('P47','hidden_plea',0.90), ('P41','emotional_masking',0.80)]),
    ("Amenat kanel kacem, indz chi petk ugnutyun.",
     [('P47','hidden_plea',0.88), ('P43','emotional_denial',0.75)]),
    ("Shat lav em, mti mti.",
     [('P47','hidden_plea',0.80), ('P41','masking',0.72)]),

    # P47 hidden_longing (arm)
    ("Arajin kayin ayl er.",
     [('P47','hidden_longing',0.88), ('P40','sincerity',0.60)]),
    ("Shat avar baci chi enkanum.",
     [('P47','hidden_longing',0.82), ('P12','vulnerability',0.55)]),
    ("Amenats poxvum e.",
     [('P47','hidden_longing',0.80), ('P8','nostalgia',0.60)]),

    # English examples for key P-families
    # P13 attack (en)
    ("You're completely wrong about this.",
     [('P13','attack',0.92), ('P9','overconfidence',0.65)]),
    ("Are you seriously saying that?",
     [('P13','soft_attack',0.82), ('P1','question',0.85)]),

    # P41 masking (en)
    ("I'm fine, don't worry.",
     [('P41','emotional_masking',0.85), ('P47','hidden_plea',0.60)]),
    ("Everything's great, really.",
     [('P41','emotional_masking',0.80), ('P43','emotional_denial',0.65)]),

    # P33 distancing (en)
    ("I need some space right now.",
     [('P33','protective_distancing',0.90), ('P4','avoidance',0.75)]),
    ("Let's talk about this later.",
     [('P33','soft_distancing',0.85), ('P4','avoidance',0.70)]),

    # P37 rupture (en)
    ("I'm done with this conversation.",
     [('P37','hard_rupture',0.92), ('P35','sudden_escalation',0.70)]),

    # Neutral negatives (en) for P47
    ("How are you doing today?",
     [('P47','hidden_reproach',0.0), ('P47','hidden_threat',0.0)]),
    ("The weather is nice.",
     [('P47','hidden_reproach',0.0), ('P47','hidden_contempt',0.0)]),
    ("Great idea, let's do it.",
     [('P47','hidden_contempt',0.0), ('P47','hidden_reproach',0.0)]),
    ("Thank you for your help.",
     [('P47','hidden_affection',0.0), ('P47','hidden_reproach',0.0)]),
    ("See you tomorrow.",
     [('P47','hidden_threat',0.0), ('P47','hidden_reproach',0.0)]),
    ("Can you pass the salt?",
     [('P47','hidden_reproach',0.0), ('P47','hidden_plea',0.0)]),

    # Neutral negatives (arm) for P47
    ("Vonc es?",
     [('P47','hidden_reproach',0.0), ('P47','hidden_threat',0.0)]),
    ("Exanak et lav e.",
     [('P47','hidden_reproach',0.0), ('P47','hidden_contempt',0.0)]),
    ("Shnorhakalutyun ugnutyan hamar.",
     [('P47','hidden_affection',0.0), ('P47','hidden_reproach',0.0)]),
]


# ─── Запись ───────────────────────────────────────────────────────────────────

def write_expert_annotations() -> dict[str, int]:
    buckets: dict[str, list[dict]] = defaultdict(list)

    all_annotations = (EXPERT_ANNOTATIONS + EXPERT_ANNOTATIONS_EXTRA
                       + SITUATION_BASED + EXTRA_AFFECTION_CONTEMPT
                       + ARMENIAN_ANNOTATIONS)
    for text, p_list in all_annotations:
        for p_id, variant, score in p_list:
            buckets[p_id].append({
                'p_id': p_id,
                'variant': variant,
                'text': text,
                'score': round(score, 3),
                'source': 'expert_annotation',
                'confidence': 'expert',
            })

    counts: dict[str, int] = {}
    for p_id, examples in buckets.items():
        out_path = OUT_DIR / f'{p_id}.jsonl'
        # читаем существующее
        existing: list[str] = []
        if out_path.exists():
            existing = out_path.read_text().splitlines()

        # добавляем новые (не дублируем)
        existing_texts = set()
        for line in existing:
            try:
                d = json.loads(line)
                existing_texts.add((d['text'][:80], d['variant']))
            except Exception:
                pass

        new_lines = []
        for ex in examples:
            key = (ex['text'][:80], ex['variant'])
            if key not in existing_texts:
                new_lines.append(json.dumps(ex, ensure_ascii=False))
                existing_texts.add(key)

        with open(out_path, 'a', encoding='utf-8') as f:
            for line in new_lines:
                f.write(line + '\n')

        counts[p_id] = len(new_lines)
    return counts


if __name__ == '__main__':
    counts = write_expert_annotations()
    total = sum(counts.values())
    print(f'Добавлено {total} экспертных примеров в {len(counts)} P-семей:')
    for p_id, cnt in sorted(counts.items()):
        if cnt > 0:
            print(f'  {p_id}: +{cnt}')
