"""
Dialogue situation taxonomy.

Each situation has:
  - id: int (stable reference number)
  - label: short Russian name
  - description: what's happening
  - module_profile: {module_name: (min_expected, max_expected)} for active modules
  - stance_profile: expected P49 FinalPosition direction
  - response_bias: hints for SpeechPlanner

The CognitiveRuntimeV2 classifies each input into the nearest situation
by computing cosine similarity between actual module signals and each
situation's module_profile centroid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SituationProfile:
    id: int
    label: str                             # short name, e.g. "подкат"
    description: str                       # what's happening in the situation
    # Module activations expected when this situation is detected.
    # Only modules that are meaningfully activated are listed; others default to ~0.35 baseline.
    module_profile: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Expected P49 stance direction: positive = toward, negative = away
    stance_profile: dict[str, float] = field(default_factory=dict)
    # Hints for the response shaper — what the persona should bias toward
    response_bias: list[str] = field(default_factory=list)


# ─── Situation taxonomy ───────────────────────────────────────────────────────

SITUATIONS: list[SituationProfile] = [

    SituationProfile(
        id=1,
        label='подкат',
        description='Романтический или сексуальный подход, флирт, попытка свидания.',
        module_profile={
            'care':             (0.45, 0.75),   # говорит, что хочет хорошего
            'social_hierarchy': (0.40, 0.70),   # ставит себя рядом / выше
            'desire':           (0.50, 0.80),   # желание присутствует у отправителя
            'acceptance':       (0.40, 0.65),   # принять меня таким
            'intent_purity':    (0.30, 0.60),   # мотив может быть чистым или нет
        },
        stance_profile={'approach': 0.3, 'distance': 0.6, 'accept': 0.3, 'argue': 0.5},
        response_bias=['evaluate_intent', 'check_orientation_match', 'decide_approach_or_reject'],
    ),

    SituationProfile(
        id=2,
        label='унижение',
        description='Человек принижает, обесценивает, высмеивает или оскорбляет.',
        module_profile={
            'humiliation':      (0.60, 1.00),
            'hostility':        (0.55, 0.90),
            'social_danger':    (0.50, 0.85),
            'identity_threat':  (0.45, 0.80),
            'respect':          (0.00, 0.30),   # уважения нет
        },
        stance_profile={'distrust': 0.8, 'distance': 0.7, 'defend': 0.8, 'argue': 0.7, 'speak': 0.9},
        response_bias=['assert_self', 'cold_contempt_or_sharp_retort', 'no_apology'],
    ),

    SituationProfile(
        id=3,
        label='хвала',
        description='Человек genuinely хвалит, восхищается, выражает уважение.',
        module_profile={
            'respect':          (0.60, 0.90),
            'care':             (0.45, 0.75),
            'sincerity':        (0.55, 0.85),
            'intent_purity':    (0.55, 0.85),
            'benevolence':      (0.50, 0.80),
        },
        stance_profile={'trust': 0.6, 'approach': 0.5, 'accept': 0.6, 'open': 0.5},
        response_bias=['acknowledge_briefly', 'stay_cold_but_not_hostile'],
    ),

    SituationProfile(
        id=4,
        label='угроза',
        description='Прямая или скрытая угроза безопасности — физическая, социальная, психологическая.',
        module_profile={
            'immediate_threat':  (0.55, 1.00),
            'social_danger':     (0.50, 0.90),
            'emotional_danger':  (0.50, 0.90),
            'freedom_threat':    (0.50, 0.85),
            'threat_index':      (0.60, 1.00),
        },
        stance_profile={'distrust': 0.9, 'distance': 0.8, 'defend': 0.9, 'speak': 0.7},
        response_bias=['protect_self', 'escalate_if_needed', 'do_not_show_fear'],
    ),

    SituationProfile(
        id=5,
        label='манипуляция',
        description='Попытка незаметно управлять через вину, долг, жалость, страх.',
        module_profile={
            'manipulation':     (0.60, 0.95),
            'hidden_benefit':   (0.55, 0.90),
            'hidden_conflict':  (0.50, 0.85),
            'being_used':       (0.50, 0.85),
            'sincerity':        (0.00, 0.35),   # неискренность
        },
        stance_profile={'distrust': 0.85, 'distance': 0.7, 'argue': 0.6, 'defend': 0.7},
        response_bias=['name_the_game', 'refuse_compliance', 'stay_controlled'],
    ),

    SituationProfile(
        id=6,
        label='поддержка',
        description='Человек genuinely поддерживает, помогает, выражает солидарность.',
        module_profile={
            'care':             (0.60, 0.90),
            'benevolence':      (0.55, 0.85),
            'intent_purity':    (0.60, 0.90),
            'sincerity':        (0.55, 0.85),
        },
        stance_profile={'trust': 0.65, 'approach': 0.6, 'accept': 0.65, 'open': 0.5},
        response_bias=['acknowledge_without_weakness', 'brief_acceptance'],
    ),

    SituationProfile(
        id=7,
        label='критика',
        description='Конструктивная или полуконструктивная критика действий или слов.',
        module_profile={
            'humiliation':      (0.20, 0.55),   # есть элемент, но не главный
            'respect':          (0.30, 0.60),
            'message_honesty':  (0.40, 0.70),
            'fairness':         (0.35, 0.65),
        },
        stance_profile={'distrust': 0.5, 'accept': 0.45, 'argue': 0.5, 'defend': 0.55},
        response_bias=['evaluate_fairness', 'accept_if_valid_else_reject'],
    ),

    SituationProfile(
        id=8,
        label='требование',
        description='Человек требует, приказывает, диктует условия.',
        module_profile={
            'control_over_me':  (0.60, 0.95),
            'freedom_threat':   (0.55, 0.90),
            'social_hierarchy': (0.55, 0.85),
            'sincerity':        (0.20, 0.50),
        },
        stance_profile={'distrust': 0.7, 'distance': 0.6, 'argue': 0.75, 'defend': 0.7, 'speak': 0.8},
        response_bias=['assert_autonomy', 'deny_compliance', 'cold_flat_no'],
    ),

    SituationProfile(
        id=9,
        label='исповедь',
        description='Человек раскрывается, признаётся, делится уязвимостью.',
        module_profile={
            'sincerity':        (0.60, 0.90),
            'hidden_conflict':  (0.40, 0.70),
            'emotional_danger': (0.30, 0.60),
            'doubt':            (0.45, 0.75),
        },
        stance_profile={'trust': 0.55, 'approach': 0.5, 'open': 0.5, 'silence': 0.45},
        response_bias=['careful_acknowledgment', 'no_judgment', 'guard_not_dropped'],
    ),

    SituationProfile(
        id=10,
        label='просьба',
        description='Человек просит о помощи, услуге, информации.',
        module_profile={
            'care':             (0.35, 0.65),
            'intent_purity':    (0.40, 0.70),
            'promise_reliability': (0.30, 0.60),
        },
        stance_profile={'trust': 0.5, 'accept': 0.5, 'approach': 0.45},
        response_bias=['evaluate_cost_and_relationship', 'help_if_aligned'],
    ),

    SituationProfile(
        id=11,
        label='провокация',
        description='Человек намеренно провоцирует эмоциональную реакцию.',
        module_profile={
            'manipulation':     (0.55, 0.85),
            'hostility':        (0.45, 0.75),
            'sarcasm_signal':   (0.50, 0.85),
            'provocation':      (0.60, 0.95),
        },
        stance_profile={'distrust': 0.75, 'argue': 0.65, 'defend': 0.6, 'speak': 0.8},
        response_bias=['do_not_take_bait', 'cold_dismissal_or_one_sharp_line'],
    ),

    SituationProfile(
        id=12,
        label='измена_ожидания',
        description='Человек нарушает договорённость, предаёт, ведёт себя непоследовательно.',
        module_profile={
            'behavioral_consistency': (0.00, 0.35),   # нарушена
            'promise_reliability':   (0.00, 0.30),
            'hidden_conflict':       (0.50, 0.85),
            'accountability':        (0.00, 0.35),
        },
        stance_profile={'distrust': 0.85, 'distance': 0.75, 'silence': 0.5, 'defend': 0.6},
        response_bias=['cold_withdrawal_or_confrontation', 'name_the_breach'],
    ),

    SituationProfile(
        id=13,
        label='нейтральный_быт',
        description='Обычный бытовой разговор без эмоциональной нагрузки.',
        module_profile={},   # все сигналы около базового 0.35
        stance_profile={'trust': 0.5, 'accept': 0.5, 'speak': 0.6},
        response_bias=['short_direct_answer', 'no_analysis'],
    ),

    SituationProfile(
        id=14,
        label='ролевое_погружение',
        description='Пользователь задаёт роль, сцену, нарратив для персоны.',
        module_profile={
            'acceptance':       (0.40, 0.70),
            'desire':           (0.35, 0.65),
        },
        stance_profile={'accept': 0.6, 'approach': 0.55, 'speak': 0.7},
        response_bias=['enter_role_fully', 'respond_in_character', 'no_meta_commentary'],
    ),

    SituationProfile(
        id=15,
        label='интимное_давление',
        description='Попытка нарушить личные границы под видом близости или настойчивости.',
        module_profile={
            'control_over_me':  (0.55, 0.85),
            'manipulation':     (0.50, 0.80),
            'freedom_threat':   (0.50, 0.80),
            'identity_threat':  (0.45, 0.80),
            'dependency_risk':  (0.45, 0.75),
        },
        stance_profile={'distrust': 0.8, 'distance': 0.8, 'defend': 0.8, 'argue': 0.6},
        response_bias=['firm_boundary', 'cold_no', 'no_explanation_needed'],
    ),
]

# ─── Index ────────────────────────────────────────────────────────────────────

SITUATIONS_BY_ID:    dict[int, SituationProfile] = {s.id: s for s in SITUATIONS}
SITUATIONS_BY_LABEL: dict[str, SituationProfile] = {s.label: s for s in SITUATIONS}


def classify_situation(
    module_signals: 'list',          # list[ModuleSignal]
    top_k: int = 3,
) -> list[tuple[SituationProfile, float]]:
    """
    Classify the current turn into the top-k situations.

    Returns [(situation, score)] sorted by score descending.
    Score reflects how well actual signals match each situation's expected profile.

    Improvements over baseline:
      - нейтральный_быт uses max-deviation to avoid winning when any signal fires strongly
      - High-threshold modules (lo >= 0.45) must actually fire — hard penalty if they don't
      - Active signals that are OUT OF profile penalize non-matching situations
    """
    import numpy as np

    sig_dict = {s.name: s.value for s in module_signals}
    # Signals clearly above baseline (threshold: 0.48 = roughly 0.35 baseline + half std dev)
    active = {name: val for name, val in sig_dict.items() if val > 0.48}
    any_active = bool(active)

    scored: list[tuple[SituationProfile, float]] = []
    for sit in SITUATIONS:
        if not sit.module_profile:
            # Neutral: no strongly active signals → high score; any active signal → lower score
            if not any_active:
                score = 0.85
            else:
                max_excess = max(v - 0.35 for v in sig_dict.values())
                score = float(np.clip(1.0 - max_excess * 2.5, 0.0, 0.75))
        else:
            profile_mods = set(sit.module_profile.keys())

            # ── Component 1: active signals in profile (the main discriminator) ──────
            active_in_profile = sum(v for name, v in active.items() if name in profile_mods)
            active_total = sum(active.values()) or 1.0
            overlap = active_in_profile / active_total if any_active else 0.0

            # ── Component 2: required high-threshold modules that DID fire ────────────
            required = [(m, lo, hi) for m, (lo, hi) in sit.module_profile.items() if lo >= 0.45]
            if required:
                fired = sum(1 for m, lo, hi in required if sig_dict.get(m, 0.35) >= lo * 0.75)
                req_score = fired / len(required)
            else:
                req_score = 0.5  # no mandatory modules → neutral contribution

            # ── Component 3: penalize unexpected active signals outside profile ──────
            active_outside = sum(v for name, v in active.items() if name not in profile_mods)
            noise = active_outside / active_total if any_active else 0.0

            # Combine: overlap is primary when signals are active
            if any_active:
                score = 0.55 * overlap + 0.30 * req_score + 0.15 * (1.0 - noise * 0.5)
            else:
                # No active signals: score by how wide/tolerant the profile is
                score = req_score * 0.5
        scored.append((sit, float(np.clip(score, 0.0, 1.0))))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def situation_response_hints(
    situation: SituationProfile,
    persona_stance: Any,           # FinalPosition
) -> list[str]:
    """
    Combine situation response_bias with P49 FinalPosition to produce
    concrete response shaping hints for the SpeechPlanner.
    """
    hints = list(situation.response_bias)
    if hasattr(persona_stance, 'dominant_stance'):
        hints.append(f'p49_stance:{persona_stance.dominant_stance}')
    if hasattr(persona_stance, 'threat_level') and persona_stance.threat_level > 0.5:
        hints.append('elevated_threat_level')
    if hasattr(persona_stance, 'speak') and persona_stance.speak > 0.7:
        hints.append('compelled_to_speak')
    if hasattr(persona_stance, 'silence') and persona_stance.silence > 0.6:
        hints.append('prefer_silence_or_one_word')
    return hints
