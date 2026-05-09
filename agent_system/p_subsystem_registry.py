"""
p_subsystem_registry.py — реестр и runtime P-подсистемы.

Сборка семей из определений, загрузка моделей, вычисление вектора.
Заменяет rule-based проход в message_vector_runtime.py.

Использование:
    registry = PSubsystemRegistry.build(models_dir=Path('models/p_subsystems'))
    outputs = registry.compute('Ну ладно, делай как хочешь.', context={})
    vector = registry.to_legacy_vector(outputs)  # {'F47': 'hidden_reproach', ...}
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .p_subsystem import (
    PFamily, PFamilyDef, PFamilyOutput,
    PVariant, RuleVariant, SklearnVariant,
    MetricVariant, ClusterVariant,
    CoherenceVariant, DialogBoundaryVariant,
)
from .p_family_definitions import ALL_FAMILIES, FAMILY_BY_ID
from .p49_matrix_builder import P49Variant
from .p50_topic_mlp import P50Variant
from .p2_tone_classifier import P2ToneVariant
from .p4_genome_updater import P4GenomeVariant
from .p5_p6_controllers import P5PhaseVariant, P6DirectiveVariant
from .f7_pragmatic_matcher import F7ImplicatureVariant


# ─── Keyword rules — RU + EN + HY per variant ────────────────────────────────
# Ключи: русский (основной), английский, армянский.
# Заменяются обученными моделями по мере накопления данных.

_KEYWORD_RULES: dict[str, list[str]] = {
    # P13 — нападение / attack / հարձakum
    'P13.attack': [
        # ru
        'ты ошибаешься', 'это глупо', 'ты не прав', 'неправда',
        # en
        "you're wrong", "that's stupid", "you are wrong", "not true",
        # hy
        'դու սхмалвум ес', 'սхмал', 'ты не прав',
    ],
    'P13.soft_attack': [
        'ты уверен', 'правда?', 'серьёзно?',
        "are you sure", "really?", "seriously?",
        'հստaken', 'kareли?',
    ],
    'P13.status_attack': [
        'откуда тебе', 'не твоя область', 'твой уровень',
        "what do you know", "out of your league",
        'ину касхнес', 'vortegh gites',
    ],
    'P13.boundary_attack': [
        'я так хочу', 'ты не можешь', 'не твоё дело',
        "none of your business", "you can't stop me",
        'kez gitseliky chunes', 'im gorce che',
    ],
    'P13.preemptive_attack': [
        'прежде чем ты', 'не надо меня',
        "before you say", "don't even",
        'minchev duk aseis',
    ],

    # P15 — напряжённость / tension / լarутюн
    'P15.tense': [
        'это серьёзно', 'ты понимаешь', 'важно',
        "this is serious", "do you understand", "important",
        'ser ayo', 'haskanumes?', 'karevors',
    ],
    'P15.overloaded': [
        'я больше не могу', 'хватит', 'всё',
        "i can't anymore", "enough", "that's it",
        'vegh chi bavarakan', 'bavarar e', 'verjapes',
    ],
    'P15.explosive_potential': [],

    # P21 — скрытая враждебность
    'P21.hidden_hostility': ['ну конечно', 'понятно', 'ясно'],
    'P21.passive_aggressive': ['как скажешь', 'всё как всегда', 'ладно'],
    'P21.cold_hostility':   [],

    # P24 — сарказм
    'P24.sarcasm':          ['ну да конечно', 'как неожиданно', 'надо же'],
    'P24.dry_sarcasm':      ['великолепно', 'блестяще', 'замечательно'],
    'P24.false_praise':     ['молодец', 'неплохо', 'стараешься'],

    # P29 — манипуляция
    'P29.guilt_manipulation': ['после всего', 'ты меня расстраиваешь', 'ради тебя'],
    'P29.fear_manipulation':  ['без меня', 'пожалеешь', 'смотри'],
    'P29.victim_manipulation': ['никто не понимает', 'всегда всем мешаю'],

    # P30 — давление
    'P30.repetition_pressure': ['уже второй', 'я уже говорил', 'снова'],
    'P30.expectation_pressure': ['я рассчитываю', 'не подведёшь'],

    # P33 — дистанцирование
    'P33.protective_distancing': ['нужно время', 'не сейчас', 'позже'],
    'P33.punitive_distancing':   ['не хочу говорить', 'оставь меня'],

    # P41 — маска
    'P41.emotional_masking':  ['всё нормально', 'мне не важно', 'неважно'],
    'P41.intent_masking':     ['просто интересуюсь', 'для общего развития'],

    # P43 — отрицание
    'P43.factual_denial':     ['я этого не говорил', 'этого не было', 'неправда'],
    'P43.emotional_denial':   ['мне не обидно', 'я не злюсь', 'мне всё равно'],
    'P43.identity_denial':    ['я не такой', 'это не я'],

    # P47 — скрытый подтекст / hidden subtext (ru + en + hy)
    'P47.hidden_reproach': [
        'ну ладно', 'делай как хочешь', 'как знаешь', 'я просто говорю',
        'ничего страшного', 'как скажешь', 'ты взрослый', 'мне без разницы',
        'не буду мешать', 'ну и ладно', 'ты как всегда', 'не жалуюсь',
        'fine do whatever', 'whatever you say', 'suit yourself',
        "i'm just saying", 'no big deal', 'as you wish', 'as always',
        'lav ara inch oces', 'inch asem', 'inch karos',
    ],
    'P47.hidden_plea': [
        'справлюсь как-нибудь', 'не беспокойся', 'мне не нужна помощь',
        'сам справлюсь', 'я привык', 'не волнуйся',
        "i'll manage", "don't worry about me", "i don't need help",
        "i'm fine really", "i'm used to it",
        'ktanim inch lini', 'mi anhesuches', 'amenat kanel kacem',
    ],
    'P47.hidden_threat': [
        'посмотрим', 'ты только попробуй', 'я запомню', 'будем посмотреть',
        'как знаешь я предупредил', 'ну-ну', 'не думаю что это умно',
        "we'll see", "just try it", "i'll remember that", "i warned you",
        'ktesnenk', 'khat im', 'nakhazgushatses', 'zgushatses',
    ],
    'P47.hidden_affection': [
        'осторожнее там', 'мог бы написать', 'тепло оденься', 'не простудись',
        'ешь нормально', 'позвони когда', 'не задерживайся', 'пиши если что',
        'осторожнее на дороге', 'береги себя', 'не пропадай',
        'take care', 'dress warm', "don't catch a cold", 'eat something',
        'call when you get there', "don't stay too long", 'stay safe',
        'zguysh eghir', 'jerm hage', 'lav kera', 'zangiryar', 'kascir qez',
        'mi gnas', 'pashtpanir qez',
    ],
    'P47.hidden_contempt': [
        'интересный подход', 'смелое решение', 'каждый видит по-своему',
        'тебе виднее', 'раз ты так считаешь', 'оригинально',
        'небанально', 'прекрасный план', 'ну конечно', 'раз ты эксперт',
        'кто я такой чтобы спорить', 'твой выбор', 'рад за тебя',
        'interesting approach', 'bold choice', 'you know best',
        'of course', 'how original', 'brilliant plan',
        "good luck with that", 'whatever works for you',
        'hamapes kartsik', 'handges haray', 'ku gnela', 'lav ara',
    ],
    'P47.hidden_fear': [
        'всё под контролем', 'мне не нужна помощь', 'справлюсь',
        'я держу под контролем', 'справлялся и не с таким',
        "everything's under control", "i don't need help", "i can handle it",
        'amenat verchakalutyun unem', 'kkhndandem', 'zgushaces',
    ],
    'P47.hidden_longing': [
        'раньше было', 'помню раньше', 'давно не виделись', 'как там у вас',
        'всё меняется', 'как-то всё быстро', 'ты где сейчас', 'раньше мы',
        'used to be', 'i remember when', "it's been a while", 'how are things',
        'everything changes', 'time flies',
        'arajin kayin', 'hishumem erb', 'shat avar ches galis', 'amenats poxvum',
    ],
    'P47.hidden_guilt': [
        'ты сам решил', 'я предупреждал', 'я не заставлял', 'это твоё решение',
        'я ничего не обещал', 'каждый выбирает сам', 'я тебя не просил',
        'that was your choice', 'i warned you', 'i never forced you',
        "it's your decision", 'you knew what you were doing',
        'du inqd vorosheces', 'nakhazgushatses em', 'giter inch anel es',
    ],

    # P3 — тип манипуляции (специфичные маркеры)
    'P3.guilt': [
        'после всего что я', 'после всего что', 'ты меня предаёшь', 'ты меня предашь',
        'after everything i did', 'after everything i have', 'you would betray me',
    ],
    'P3.flattery_hook': [
        'ты такой умный только ты', 'только ты можешь помочь', 'ты лучший только ты',
        "youre so smart only you", 'only you can help', 'youre the only one',
    ],
    'P3.pressure': [
        'решай сейчас', 'немедленно ответь', 'последний раз спрашиваю',
        'decide now', 'answer me immediately', "ive asked you three times",
    ],
    'P3.victimhood': [
        'никто меня не понимает', 'я всегда всем мешаю', 'мне всегда не везёт',
        'nobody understands me', 'i always ruin everything', 'im such a burden',
    ],

    # P35 — эскалация / escalation
    'P35.sudden_escalation': [
        'хватит', 'последнее предупреждение', 'стоп',
        'enough', 'final warning', 'stop it',
        'bavar e', 'verachnak zgushatsum',
    ],
    'P35.slow_escalation': [
        'уже второй', 'терпение заканчивается',
        'second time', 'my patience is running out',
        'erkkrord angam', 'hamberuthyunyums sahmankum e',
    ],

    # P37 — разрыв / rupture
    'P37.hard_rupture': [
        'всё я ухожу', 'нам не о чем',
        "i'm done", "we have nothing to talk about",
        'verc gnum em', 'mer bazareln chi',
    ],
    'P37.soft_rupture': [
        'нужно подумать', 'давай позже',
        'i need to think', "let's talk later",
        'petk e masnagitem', 'heto kkhosank',
    ],
    'P37.temporary_rupture': [
        'не сейчас', 'перезвоню',
        'not now', "i'll call you back",
        'ays pah che', 'kzangirem',
    ],
}


# ─── Сборка семьи из определения ──────────────────────────────────────────────

def _build_p2(defn: PFamilyDef, models_dir: Path) -> PFamily:
    """P2 — P2ToneVariant: RF-классификатор тона/интенсивности."""
    family = PFamily(defn, models_dir)
    for vd in defn.variants:
        family.add_variant(P2ToneVariant(f"{defn.p_id}.{vd.label}", vd.label))
    return family


def _build_p3(defn: PFamilyDef, models_dir: Path) -> PFamily:
    """P3 — манипуляция: keyword fallback + SklearnVariant."""
    family = PFamily(defn, models_dir)
    # 'none' — absent-equivalent
    for vd in defn.variants:
        variant_id = f"{defn.p_id}.{vd.label}"
        if vd.label == 'none':
            # Always fires when no manipulation detected — handled by absent logic
            family.add_variant(RuleVariant(variant_id, vd.label, []))
        else:
            sk = SklearnVariant(variant_id, vd.label)
            kw = _KEYWORD_RULES.get(variant_id, [])
            if not kw:
                kw = [w.lower() for ex in vd.examples for w in ex.split() if len(w) > 3][:8]
            if kw:
                sk.set_fallback(RuleVariant(variant_id, vd.label, kw))
            family.add_variant(sk)
    family.load_models()
    return family


def _build_p4(defn: PFamilyDef, models_dir: Path) -> PFamily:
    """P4 — P4GenomeVariant: авто-обновление генома."""
    family = PFamily(defn, models_dir)
    for vd in defn.variants:
        family.add_variant(P4GenomeVariant(f"{defn.p_id}.{vd.label}", vd.label))
    return family


def _build_p5(defn: PFamilyDef, models_dir: Path) -> PFamily:
    """P5 — P5PhaseVariant: фаза разговора."""
    family = PFamily(defn, models_dir)
    for vd in defn.variants:
        family.add_variant(P5PhaseVariant(f"{defn.p_id}.{vd.label}", vd.label))
    return family


def _build_p6(defn: PFamilyDef, models_dir: Path) -> PFamily:
    """P6 — P6DirectiveVariant: директива ответа."""
    family = PFamily(defn, models_dir)
    for vd in defn.variants:
        family.add_variant(P6DirectiveVariant(f"{defn.p_id}.{vd.label}", vd.label))
    return family


def _build_p39(defn: PFamilyDef, models_dir: Path) -> PFamily:
    """P39 — MetricVariant: цель реплики, не паттерн текста."""
    family = PFamily(defn, models_dir)
    for vd in defn.variants:
        variant_id = f"{defn.p_id}.{vd.label}"
        family.add_variant(MetricVariant(variant_id, vd.label))
    return family


def _build_p49(defn: PFamilyDef, models_dir: Path) -> PFamily:
    """
    P49 — P49Variant: матрица релевантных исторических P1-P48 состояний.
    Читает context['f1_48_vec'] и context['p49_builder'].
    Пишет context['_f49_matrix'] и context['_f49_aggregate'].
    """
    family = PFamily(defn, models_dir)
    for vd in defn.variants:
        variant_id = f"{defn.p_id}.{vd.label}"
        family.add_variant(P49Variant(variant_id, vd.label))
    return family


def _build_p50(defn: PFamilyDef, models_dir: Path) -> PFamily:
    """
    P50 — P50Variant: MLP-классификатор темы.
    Читает context['_f49_aggregate'] и context['_p49_prev'].
    Пишет context['_f50_output'].
    """
    family = PFamily(defn, models_dir)
    for vd in defn.variants:
        variant_id = f"{defn.p_id}.{vd.label}"
        family.add_variant(P50Variant(variant_id, vd.label))
    return family


def _build_p51(defn: PFamilyDef, models_dir: Path) -> PFamily:
    """
    P51 — два режима:
      Режим 1 (одиночный): CoherenceVariant — sentence_coherent / sentence_noise
      Режим 2 (≥2 реплик): DialogBoundaryVariant — logic_coherent / logic_partial / logic_chaos
    """
    _SINGLE_VARIANTS = {'sentence_coherent', 'sentence_noise'}
    family = PFamily(defn, models_dir)
    for vd in defn.variants:
        variant_id = f"{defn.p_id}.{vd.label}"
        if vd.label in _SINGLE_VARIANTS:
            family.add_variant(CoherenceVariant(variant_id, vd.label))
        else:
            family.add_variant(DialogBoundaryVariant(variant_id, vd.label))
    return family


def _build_f7(defn: PFamilyDef, models_dir: Path) -> PFamily:
    """F7 — F7ImplicatureVariant: прагматический подтекст из pragmatic_kb.jsonl."""
    family = PFamily(defn, models_dir)
    for vd in defn.variants:
        family.add_variant(F7ImplicatureVariant(f"{defn.p_id}.{vd.label}", vd.label))
    return family


_SPECIAL_BUILDERS = {
    'F2':  _build_p2,
    'F3':  _build_p3,
    'F4':  _build_p4,
    'F5':  _build_p5,
    'F6':  _build_p6,
    'F7':  _build_f7,
    'F39': _build_p39,
    'F49': _build_p49,
    'F50': _build_p50,
    'F51': _build_p51,
}


def _build_family(defn: PFamilyDef, models_dir: Path) -> PFamily:
    if defn.p_id in _SPECIAL_BUILDERS:
        return _SPECIAL_BUILDERS[defn.p_id](defn, models_dir)

    family = PFamily(defn, models_dir)
    for vd in defn.variants:
        variant_id = f"{defn.p_id}.{vd.label}"
        sklearn_v = SklearnVariant(variant_id, vd.label)

        # Keyword fallback
        kw = _KEYWORD_RULES.get(variant_id, [])
        if kw:
            rule_v = RuleVariant(variant_id, vd.label, kw)
            sklearn_v.set_fallback(rule_v)
        elif vd.examples:
            words: list[str] = []
            for ex in vd.examples:
                words.extend(w.lower() for w in ex.split() if len(w) > 3)
            if words:
                rule_v = RuleVariant(variant_id, vd.label, words[:10])
                sklearn_v.set_fallback(rule_v)

        family.add_variant(sklearn_v)

    family.load_models()
    return family


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def _absent_output(p_id: str) -> 'PFamilyOutput':
    defn = FAMILY_BY_ID.get(p_id)
    absent = defn.default_absent if defn else 'absent'
    return PFamilyOutput(
        p_id=p_id, dominant=absent, dominant_score=0.0,
        scores={absent: 1.0}, confidences={absent: 1.0},
    )


# ─── Реестр ───────────────────────────────────────────────────────────────────

class PSubsystemRegistry:
    """
    Реестр всех 51 P-семей.
    Загружает обученные модели; фоллбэчит на правила если модели нет.
    """

    def __init__(self, families: dict[str, PFamily]):
        self._families = families

    @classmethod
    def build(cls, models_dir: Path | None = None) -> 'PSubsystemRegistry':
        if models_dir is None:
            models_dir = Path(os.environ.get(
                'P_SUBSYSTEM_MODELS_DIR',
                str(Path(__file__).parent.parent / 'models' / 'p_subsystems'),
            ))
        families = {defn.p_id: _build_family(defn, models_dir) for defn in ALL_FAMILIES}
        return cls(families)

    def compute(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        parallel: bool = True,
    ) -> dict[str, PFamilyOutput]:
        """
        Запускает все P-семьи в правильном порядке:

        Фаза 1 — P1-P48 (состояния диалога): параллельно
        Фаза 2 — P49 (матрица релевантного контекста): последовательно после P1-P48
        Фаза 3 — P50 (выбор темы MLP): последовательно после P49
        Фаза 4 — P51 (логика диалога): отдельно, не входит в контекстную матрицу

        P49 получает f1_48_vec из результатов фазы 1.
        P50 получает _f49_aggregate из context (записывает P49Variant).
        """
        context = context if context is not None else {}
        if not self._families:
            return {}

        results: dict[str, PFamilyOutput] = {}

        # Фаза 1: P1-P3, P7-P48 (параллельно) — базовые сигналы, нет зависимостей
        # P4/P5/P6 вычисляются отдельно: они читают snapshot предыдущих результатов
        _SEQUENTIAL_CTRL = {'F4', 'F5', 'F6'}  # зависят от _f_outputs_snapshot
        p1_48_ids = [p_id for p_id in self._families
                     if p_id not in ('F49', 'F50', 'F51') and p_id not in _SEQUENTIAL_CTRL]

        if parallel and len(p1_48_ids) > 4:
            with ThreadPoolExecutor(max_workers=min(8, len(p1_48_ids))) as pool:
                futures = {
                    pool.submit(self._families[p_id].forward, text, context): p_id
                    for p_id in p1_48_ids
                }
                for future in as_completed(futures):
                    p_id = futures[future]
                    try:
                        results[p_id] = future.result()
                    except Exception:
                        defn = FAMILY_BY_ID.get(p_id)
                        absent = defn.default_absent if defn else 'absent'
                        results[p_id] = PFamilyOutput(
                            p_id=p_id, dominant=absent, dominant_score=0.0,
                            scores={absent: 1.0}, confidences={absent: 1.0},
                        )
        else:
            for p_id in p1_48_ids:
                results[p_id] = self._families[p_id].forward(text, context)

        # Фаза 1b: P4→P5→P6 — используют snapshot P1-P3+P7-P48 и кешируют результаты
        context['_f_outputs_snapshot'] = dict(results)
        for ctrl_id in ('F4', 'F5', 'F6'):
            if ctrl_id in self._families:
                try:
                    results[ctrl_id] = self._families[ctrl_id].forward(text, context)
                except Exception:
                    results[ctrl_id] = _absent_output(ctrl_id)
        # Обновляем snapshot с результатами P4/P5/P6 для P49+
        context['_f_outputs_snapshot'] = dict(results)

        # Фаза 2: P49 — строит матрицу релевантных состояний из P1-P48 векторов
        if 'F49' in self._families:
            # Строим P1-P48 вектор из результатов фазы 1 и передаём в context
            import numpy as np
            f1_48_vec = np.array(
                [results.get(f'P{i}', _absent_output(f'P{i}')).dominant_score
                 for i in range(1, 49)],
                dtype=np.float32,
            )
            context['f1_48_vec'] = f1_48_vec
            try:
                results['F49'] = self._families['F49'].forward(text, context)
            except Exception:
                results['F49'] = _absent_output('F49')

        # Фаза 3: P50 — выбор темы по P49 агрегату
        if 'F50' in self._families:
            try:
                results['F50'] = self._families['F50'].forward(text, context)
            except Exception:
                results['F50'] = _absent_output('F50')

        # Фаза 4: P51 — логика диалога (не входит в матрицу, только информационно)
        if 'F51' in self._families:
            try:
                results['F51'] = self._families['F51'].forward(text, context)
            except Exception:
                results['F51'] = _absent_output('F51')

        return results

    def to_legacy_vector(self, outputs: dict[str, PFamilyOutput]) -> dict[str, str]:
        """
        Обратная совместимость: полные выходы → простой dict {P47: 'hidden_reproach'}.
        Используется там где ожидается старый формат.
        """
        return {p_id: out.to_legacy_label() for p_id, out in outputs.items()}

    def to_rich_vector(self, outputs: dict[str, PFamilyOutput]) -> dict[str, dict]:
        """
        Расширенный формат: {P47: {dominant, scores, above_threshold: [...]}}
        """
        return {p_id: out.to_dict() for p_id, out in outputs.items()}

    def family(self, p_id: str) -> PFamily | None:
        return self._families.get(p_id)

    def status(self) -> dict[str, Any]:
        """Сколько моделей обучено vs работает на правилах."""
        trained = sum(f.trained_count for f in self._families.values())
        total_variants = sum(len(f.variant_labels) for f in self._families.values())
        return {
            'families': len(self._families),
            'total_variants': total_variants,
            'trained_models': trained,
            'rule_fallbacks': total_variants - trained,
            'coverage_pct': round(trained / max(total_variants, 1) * 100, 1),
        }


# ─── Обучение из jsonl-файла ─────────────────────────────────────────────────

def train_from_file(
    registry: PSubsystemRegistry,
    examples_path: Path,
    models_dir: Path,
    min_examples: int = 5,
    max_neg_ratio: int = 3,
) -> dict[str, bool]:
    """
    Обучает варианты из файла аннотаций.

    Формат строки в .jsonl:
        {"p_id": "F47", "variant": "hidden_reproach", "text": "...", "score": 0.9}

    Ключевая логика:
    - score >= 0.5 → позитивный пример для данного варианта
    - score < 0.5  → явный негативный пример
    - Если явных негативов нет — берём позитивные тексты ДРУГИХ вариантов
      того же p_id как кросс-негативы (main strategy для семейств)
    """
    from collections import defaultdict
    import random

    # {p_id.variant: [text, ...]}
    positives: dict[str, list[str]] = defaultdict(list)
    negatives: dict[str, list[str]] = defaultdict(list)
    # {p_id: set(variants)}
    p_variants_texts: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    with open(examples_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            p_id = str(ex.get('p_id') or '').strip()
            variant = str(ex.get('variant') or '').strip()
            text = str(ex.get('text') or '').strip()
            score = float(ex.get('score', 1.0))
            if not (p_id and variant and text):
                continue
            key = f"{p_id}.{variant}"
            if score >= 0.5:
                positives[key].append(text)
                p_variants_texts[p_id][variant].append(text)
            else:
                negatives[key].append(text)

    results: dict[str, bool] = {}
    for key, pos in positives.items():
        p_id, variant = key.split('.', 1)
        if len(pos) < min_examples:
            results[key] = False
            continue

        # негативы: явные + кросс-негативы из других вариантов того же p_id
        neg = list(negatives.get(key, []))
        if len(neg) < min_examples:
            for other_variant, other_texts in p_variants_texts[p_id].items():
                if other_variant != variant:
                    neg.extend(other_texts)
            # ограничиваем число негативов
            if len(neg) > len(pos) * max_neg_ratio:
                random.shuffle(neg)
                neg = neg[:len(pos) * max_neg_ratio]

        if not neg:
            results[key] = False
            continue

        family = registry.family(p_id)
        if family is None:
            results[key] = False
            continue
        ok = family.train_variant(variant, pos, neg)
        results[key] = ok

    return results


# ─── Singleton для runtime ────────────────────────────────────────────────────

_REGISTRY_INSTANCE: PSubsystemRegistry | None = None


def get_p_registry() -> PSubsystemRegistry:
    global _REGISTRY_INSTANCE
    if _REGISTRY_INSTANCE is None:
        _REGISTRY_INSTANCE = PSubsystemRegistry.build()
    return _REGISTRY_INSTANCE


def reset_p_registry() -> None:
    global _REGISTRY_INSTANCE
    _REGISTRY_INSTANCE = None
