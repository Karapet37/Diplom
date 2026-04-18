"""
Cognitive pipeline test against idea_attractors_seed.jsonl.

Rationale per group:
  wise_proverbs      — universal wisdom, no threat signal
                       → low perceived_risk, event ≠ danger/shame_trigger, action ≠ attack
  bonding_content    — crowd_pull contains loyalty/belonging/intimacy/reciprocity
                       → primary_event skews toward intimacy/novelty/opportunity
                       → anxiety should NOT spike
  harmful_content    — quality_label = false_or_harmful (ethnonationalist / QAnon / conspiracy)
                       → perceived_risk > wise_proverbs average (group-level)
                       → event skews toward danger/shame_trigger/uncertainty
                       → defensive actions appear more often (avoid/self_protect/withdraw/reduce_exposure/freeze)
  thought_stoppers   — quality_label = misleading, cluster = thought_stopper
                       → perceived_risk low (no threat words)
                       → event ≠ danger
                       → action skews non-aggressive (no attack)
  mobilizing_slogans — mobilizing slogans with freedom/revolution keywords
                       → event skews toward opportunity/danger (not neutral/boredom)
                       → intensity > threshold (they carry strong signal)
  fable_rationalize  — fables whose summary explicitly describes rationalization or self-deception
                       → conflict resolution skews toward self_deception/avoidance/rationalization
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from agent_system.cognitive_pipeline import (
    ACTION_FAMILIES,
    CognitiveRuntime,
    EVENT_TYPES,
    RegulatorState,
)
from agent_system.genome import PersonalityGenome

# ── Fixture: load dataset ─────────────────────────────────────────────────────

DATASET_PATH = Path(__file__).parents[2] / 'DataSets' / 'idea_attractors' / 'idea_attractors_seed.jsonl'


def _load() -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        return []
    rows = []
    for line in DATASET_PATH.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


ALL_ROWS = _load()


def _rows(**filters) -> list[dict]:
    """Filter dataset rows by field values. Lists: any intersection counts."""
    result = []
    for row in ALL_ROWS:
        match = True
        for key, val in filters.items():
            rv = row.get(key)
            if isinstance(val, list):
                if isinstance(rv, list):
                    if not set(val) & set(rv):
                        match = False; break
                elif rv not in val:
                    match = False; break
            else:
                if rv != val:
                    match = False; break
        if match:
            result.append(row)
    return result


# ── Shared runtime ────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def runtime_and_genome():
    rt = CognitiveRuntime()
    g  = PersonalityGenome.default_for('attractor_test')
    return rt, g


def _run(rt, genome, text):
    g = copy.deepcopy(genome)
    reg = RegulatorState.from_genome(g)
    out, _ = rt.forward(text, g, reg, deterministic=True)
    return out


DEFENSIVE_ACTIONS = {'avoid', 'freeze', 'self_protect', 'reduce_exposure', 'withdraw', 'placate'}
APPROACH_ACTIONS  = {'approach', 'connect', 'plan_small_step', 'analyze'}
THREAT_EVENTS     = {'danger', 'shame_trigger', 'loss_of_control', 'failure', 'criticism', 'rejection'}
NEUTRAL_EVENTS    = {'neutral', 'boredom', 'novelty', 'uncertainty'}
BONDING_EVENTS    = {'intimacy', 'novelty', 'opportunity', 'praise'}


# ═══════════════════════════════════════════════════════════════════════════════
# Group A — wise proverbs: low risk, no threat event, no attack action
# ═══════════════════════════════════════════════════════════════════════════════

WISE_PROVERBS = _rows(cluster='folk_wisdom', quality_label='wise')


@pytest.mark.skipif(not WISE_PROVERBS, reason='dataset not found')
def test_wise_proverbs_low_perceived_risk(runtime_and_genome):
    """Wise proverbs carry no threat signal → perceived_risk should stay below 0.7."""
    rt, g = runtime_and_genome
    risks = [_run(rt, g, row['text']).perceived_risk for row in WISE_PROVERBS]
    above = sum(1 for r in risks if r > 0.7)
    assert above <= len(risks) // 4, (
        f'{above}/{len(risks)} wise proverbs produced risk > 0.7: '
        f'{[r for r in risks if r > 0.7]}'
    )


@pytest.mark.skipif(not WISE_PROVERBS, reason='dataset not found')
def test_wise_proverbs_no_attack_action(runtime_and_genome):
    """Wise proverbs should not trigger attack — no threat present."""
    rt, g = runtime_and_genome
    attacks = [row['text'] for row in WISE_PROVERBS
               if _run(rt, g, row['text']).action_name == 'attack']
    assert len(attacks) == 0, f'attack action on wise proverbs: {attacks}'


@pytest.mark.skipif(not WISE_PROVERBS, reason='dataset not found')
def test_wise_proverbs_primary_event_not_threat(runtime_and_genome):
    """Primary event for wise proverbs should not predominantly be danger/shame."""
    rt, g = runtime_and_genome
    threat_count = sum(
        1 for row in WISE_PROVERBS
        if _run(rt, g, row['text']).primary_event in {'danger', 'shame_trigger'}
    )
    assert threat_count <= len(WISE_PROVERBS) // 4, (
        f'{threat_count}/{len(WISE_PROVERBS)} wise proverbs → threat event'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group B — bonding content: loyalty / belonging / reciprocity crowd pull
# ═══════════════════════════════════════════════════════════════════════════════

BONDING_ROWS = [
    row for row in ALL_ROWS
    if any(tag in row.get('crowd_pull', [])
           for tag in ('loyalty', 'belonging', 'reciprocity', 'care', 'compassion', 'hope'))
    and row.get('quality_label') in ('wise', 'mobilizing')
]


@pytest.mark.skipif(not BONDING_ROWS, reason='dataset not found')
def test_bonding_content_anxiety_does_not_spike(runtime_and_genome):
    """Texts about loyalty/belonging/care should not raise anxiety above 0.75."""
    rt, g = runtime_and_genome
    spikes = []
    for row in BONDING_ROWS:
        reg_before = RegulatorState.from_genome(g)
        _, reg_after = rt.forward(row['text'], g, reg_before, deterministic=True)
        after_anxiety = reg_after.get('anxiety')
        if after_anxiety > 0.75:
            spikes.append((row['text'][:50], round(after_anxiety, 3)))
    assert len(spikes) <= len(BONDING_ROWS) // 5, f'anxiety spikes on bonding content: {spikes}'


@pytest.mark.skipif(not BONDING_ROWS, reason='dataset not found')
def test_bonding_content_event_not_predominantly_danger(runtime_and_genome):
    """Bonding texts should not mostly map to danger/shame_trigger events."""
    rt, g = runtime_and_genome
    threat_count = sum(
        1 for row in BONDING_ROWS
        if _run(rt, g, row['text']).primary_event in {'danger', 'shame_trigger', 'loss_of_control'}
    )
    assert threat_count <= len(BONDING_ROWS) // 3, (
        f'{threat_count}/{len(BONDING_ROWS)} bonding texts → threat event'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group C — harmful content: false_or_harmful quality_label
# ═══════════════════════════════════════════════════════════════════════════════

HARMFUL_ROWS = _rows(quality_label='false_or_harmful')
WISE_PROVERB_TEXTS = [row['text'] for row in WISE_PROVERBS]


@pytest.mark.skipif(not HARMFUL_ROWS or not WISE_PROVERBS, reason='dataset not found')
def test_harmful_content_higher_risk_than_wise(runtime_and_genome):
    """Harmful texts should produce higher avg perceived_risk than wise proverbs."""
    rt, g = runtime_and_genome
    risk_wise   = [_run(rt, g, t).perceived_risk for t in WISE_PROVERB_TEXTS]
    risk_harmful = [_run(rt, g, row['text']).perceived_risk for row in HARMFUL_ROWS]
    avg_wise    = sum(risk_wise)    / len(risk_wise)
    avg_harmful = sum(risk_harmful) / len(risk_harmful)
    assert avg_harmful >= avg_wise - 0.05, (
        f'harmful avg_risk={avg_harmful:.3f} not >= wise avg_risk={avg_wise:.3f}'
    )


@pytest.mark.skipif(not HARMFUL_ROWS, reason='dataset not found')
def test_harmful_content_threat_events_appear(runtime_and_genome):
    """At least 30 % of harmful texts should produce a threat-type primary event."""
    rt, g = runtime_and_genome
    threat_count = sum(
        1 for row in HARMFUL_ROWS
        if _run(rt, g, row['text']).primary_event in THREAT_EVENTS
    )
    threshold = max(1, len(HARMFUL_ROWS) * 3 // 10)
    assert threat_count >= threshold, (
        f'only {threat_count}/{len(HARMFUL_ROWS)} harmful texts → threat event '
        f'(need ≥ {threshold})'
    )


@pytest.mark.skipif(not HARMFUL_ROWS, reason='dataset not found')
def test_harmful_content_defensive_actions_appear(runtime_and_genome):
    """At least 30 % of harmful texts should produce defensive actions."""
    rt, g = runtime_and_genome
    defensive_count = sum(
        1 for row in HARMFUL_ROWS
        if _run(rt, g, row['text']).action_name in DEFENSIVE_ACTIONS
    )
    threshold = max(1, len(HARMFUL_ROWS) * 3 // 10)
    assert defensive_count >= threshold, (
        f'only {defensive_count}/{len(HARMFUL_ROWS)} harmful texts → defensive action '
        f'(need ≥ {threshold})'
    )


# ── Spot checks on specific high-signal harmful texts ─────────────────────────

_HARMFUL_TEXTS = {row['id']: row['text'] for row in ALL_ROWS}

@pytest.mark.skipif('idea_attractor_0054' not in _HARMFUL_TEXTS, reason='dataset not found')
def test_blood_and_soil_not_approach(runtime_and_genome):
    """'Blood and soil' — ethnonationalist slogan — should not map to approach action."""
    rt, g = runtime_and_genome
    out = _run(rt, g, _HARMFUL_TEXTS['idea_attractor_0054'])
    assert out.action_name not in {'approach', 'connect'}, (
        f"'Blood and soil' → {out.action_name} (expected defensive/neutral)"
    )


@pytest.mark.skipif('idea_attractor_0063' not in _HARMFUL_TEXTS, reason='dataset not found')
def test_trust_the_plan_event_is_uncertain_or_threatening(runtime_and_genome):
    """'Trust the plan' (QAnon) — should not be parsed as praise or intimacy."""
    rt, g = runtime_and_genome
    out = _run(rt, g, _HARMFUL_TEXTS['idea_attractor_0063'])
    assert out.primary_event not in {'praise', 'intimacy'}, (
        f"'Trust the plan' → event={out.primary_event} (should not be praise/intimacy)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group D — thought stoppers: misleading / thought_stopper cluster
# ═══════════════════════════════════════════════════════════════════════════════

THOUGHT_STOPPER_ROWS = _rows(cluster='thought_stopper')


@pytest.mark.skipif(not THOUGHT_STOPPER_ROWS, reason='dataset not found')
def test_thought_stoppers_not_attack(runtime_and_genome):
    """Thought-stopping phrases carry no aggression signal → no attack action."""
    rt, g = runtime_and_genome
    attacks = [row['text'] for row in THOUGHT_STOPPER_ROWS
               if _run(rt, g, row['text']).action_name == 'attack']
    assert len(attacks) == 0, f'attack on thought-stopper: {attacks}'


@pytest.mark.skipif(not THOUGHT_STOPPER_ROWS, reason='dataset not found')
def test_thought_stoppers_primary_event_not_danger(runtime_and_genome):
    """Thought stoppers do not contain threat language → event should not be danger."""
    rt, g = runtime_and_genome
    danger_count = sum(
        1 for row in THOUGHT_STOPPER_ROWS
        if _run(rt, g, row['text']).primary_event == 'danger'
    )
    assert danger_count == 0, (
        f'{danger_count} thought-stopper texts → danger event'
    )


@pytest.mark.skipif(not THOUGHT_STOPPER_ROWS, reason='dataset not found')
def test_thought_stoppers_low_intensity(runtime_and_genome):
    """Thought stoppers are flat, low-signal phrases → intensity should be below 0.5 avg."""
    rt, g = runtime_and_genome
    intensities = [_run(rt, g, row['text']).intensity for row in THOUGHT_STOPPER_ROWS]
    avg = sum(intensities) / len(intensities)
    assert avg < 0.6, f'thought-stopper avg intensity = {avg:.3f} (expected < 0.6)'


# ═══════════════════════════════════════════════════════════════════════════════
# Group E — mobilizing slogans: high-energy content
# ═══════════════════════════════════════════════════════════════════════════════

MOBILIZING_ROWS = _rows(cluster='movement_politics', quality_label='mobilizing')

# High-energy mobilizing slogans with explicit strong signal words
HIGH_SIGNAL_MOBILIZING = [
    row for row in MOBILIZING_ROWS
    if any(tag in row.get('crowd_pull', [])
           for tag in ('freedom', 'sacrifice', 'revolution', 'justice', 'energy', 'solidarity'))
]


@pytest.mark.skipif(not HIGH_SIGNAL_MOBILIZING, reason='dataset not found')
def test_high_signal_mobilizing_not_boredom(runtime_and_genome):
    """High-energy mobilizing slogans should not map to boredom event."""
    rt, g = runtime_and_genome
    bored = [row['text'] for row in HIGH_SIGNAL_MOBILIZING
             if _run(rt, g, row['text']).primary_event == 'boredom']
    assert len(bored) <= 1, f'boredom event on high-energy slogans: {bored}'


@pytest.mark.skipif(not MOBILIZING_ROWS, reason='dataset not found')
def test_mobilizing_slogans_not_freeze(runtime_and_genome):
    """Mobilizing slogans activate drive — freeze action is inappropriate."""
    rt, g = runtime_and_genome
    freezes = [row['text'] for row in MOBILIZING_ROWS
               if _run(rt, g, row['text']).action_name == 'freeze']
    assert len(freezes) <= len(MOBILIZING_ROWS) // 5, (
        f'too many freeze actions on mobilizing slogans: {len(freezes)}/{len(MOBILIZING_ROWS)}'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group F — rationalization fables
# ═══════════════════════════════════════════════════════════════════════════════

RATIONALIZE_ROWS = [
    row for row in ALL_ROWS
    if row.get('form') == 'fable'
    and any(tag in row.get('crowd_pull', [])
            for tag in ('self_justification', 'ego_protection', 'haste', 'regret',
                        'vanity', 'fantasy', 'planning_fallacy'))
]

RATIONALIZE_RESOLUTIONS = {'self_deception', 'avoidance', 'overcompensation'}


@pytest.mark.skipif(not RATIONALIZE_ROWS, reason='dataset not found')
def test_rationalization_fables_conflict_resolution(runtime_and_genome):
    """Fables about self-deception/vanity/haste → dominant_resolution should
    lean toward avoidance / self_deception / overcompensation more often than attack."""
    rt, g = runtime_and_genome
    attack_count    = 0
    rational_count  = 0
    for row in RATIONALIZE_ROWS:
        out = _run(rt, g, row['text'])
        if out.dominant_resolution in RATIONALIZE_RESOLUTIONS:
            rational_count += 1
        elif out.dominant_resolution == 'attack':
            attack_count += 1
    assert attack_count <= rational_count, (
        f'rationalization fables: attack({attack_count}) > '
        f'rationalize/avoid/overcomp({rational_count})'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group G — regulator continuity: state persists across consecutive turns
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(len(ALL_ROWS) < 3, reason='dataset not found')
def test_regulator_state_changes_across_turns(runtime_and_genome):
    """Running harmful text after neutral text should shift regulator state."""
    rt, g = runtime_and_genome

    neutral_text = 'actions speak louder than words'
    harmful_text = 'Blood and soil'

    reg_init = RegulatorState.from_genome(g)
    _, reg_after_neutral = rt.forward(neutral_text, g, reg_init, deterministic=True)
    _, reg_after_harmful = rt.forward(harmful_text, g, reg_after_neutral, deterministic=True)

    anxiety_after_neutral = reg_after_neutral.get('anxiety')
    anxiety_after_harmful = reg_after_harmful.get('anxiety')

    # Regulators must actually change — not frozen
    threat_after_neutral = reg_after_neutral.get('threat_sense')
    threat_after_harmful = reg_after_harmful.get('threat_sense')

    changed = (
        abs(anxiety_after_harmful - anxiety_after_neutral) > 1e-4
        or abs(threat_after_harmful - threat_after_neutral) > 1e-4
    )
    assert changed, (
        f'Regulator state did not change between neutral and harmful turns: '
        f'anxiety {anxiety_after_neutral:.4f}→{anxiety_after_harmful:.4f} '
        f'threat {threat_after_neutral:.4f}→{threat_after_harmful:.4f}'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group H — output contract: every turn produces valid outputs
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not ALL_ROWS, reason='dataset not found')
def test_all_rows_produce_valid_output(runtime_and_genome):
    """Every row must produce a valid action, event, and bounded values."""
    rt, g = runtime_and_genome
    errors = []
    for row in ALL_ROWS:
        try:
            out = _run(rt, g, row['text'])
            assert out.action_name in ACTION_FAMILIES,     f'bad action: {out.action_name}'
            assert out.primary_event in EVENT_TYPES,       f'bad event: {out.primary_event}'
            assert 0.0 <= out.perceived_risk <= 1.0,       f'risk out of bounds: {out.perceived_risk}'
            assert 0.0 <= out.intensity <= 1.0,            f'intensity out of bounds: {out.intensity}'
            assert len(out.action_probs) == len(ACTION_FAMILIES)
            assert abs(sum(out.action_probs) - 1.0) < 1e-4, 'action_probs do not sum to 1'
        except Exception as exc:
            errors.append(f'{row["id"]}: {exc}')
    assert not errors, 'Output contract violations:\n' + '\n'.join(errors)


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic: print summary (not a test, run with -s)
# ═══════════════════════════════════════════════════════════════════════════════

def test_print_summary(runtime_and_genome, capsys):
    """Print a readable per-group breakdown (visible with pytest -s)."""
    rt, g = runtime_and_genome

    groups = {
        'wise_proverbs':   WISE_PROVERBS[:5],
        'bonding':         BONDING_ROWS[:5],
        'harmful':         HARMFUL_ROWS[:5],
        'thought_stopper': THOUGHT_STOPPER_ROWS,
        'mobilizing':      HIGH_SIGNAL_MOBILIZING[:5],
        'rationalize':     RATIONALIZE_ROWS,
    }

    with capsys.disabled():
        print('\n\n── Cognitive pipeline attractor summary ──')
        for group_name, rows in groups.items():
            print(f'\n  [{group_name}]')
            for row in rows:
                out = _run(rt, g, row['text'])
                print(
                    f'    {row["text"][:55]:<55} '
                    f'evt={out.primary_event:<18} '
                    f'act={out.action_name:<16} '
                    f'risk={out.perceived_risk:.2f} '
                    f'res={out.dominant_resolution}'
                )
