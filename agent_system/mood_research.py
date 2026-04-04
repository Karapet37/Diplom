from __future__ import annotations

import atexit
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .duplicate_resolver import normalize_name
from .graph_store import load_json, write_json
from .models import HeadBundle, MessageAnalysis, MoodCluster, MoodFeatureSnapshot, MoodResearchReport, SocialRoleDecision, Situation
from .runtime_config import get_runtime_config

_MOOD_RESEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='agent-system-mood-research')
_MOOD_RESEARCH_EXECUTOR_CLOSED = False
_MOOD_RESEARCH_LOCK = Lock()
_REPORT_SNAPSHOT_LIMIT = 240


def _shutdown_mood_research_executor() -> None:
    global _MOOD_RESEARCH_EXECUTOR_CLOSED
    with _MOOD_RESEARCH_LOCK:
        if _MOOD_RESEARCH_EXECUTOR_CLOSED:
            return
        shutdown = getattr(_MOOD_RESEARCH_EXECUTOR, 'shutdown', None)
        if callable(shutdown):
            shutdown(wait=False, cancel_futures=True)
        _MOOD_RESEARCH_EXECUTOR_CLOSED = True


atexit.register(_shutdown_mood_research_executor)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _research_paths() -> tuple[Path, Path, Path, Path]:
    paths = get_runtime_config().paths
    return (
        paths.mood_datasets_dir,
        paths.mood_personas_dir,
        paths.mood_sessions_dir,
        paths.mood_reports_dir,
    )


def _global_dataset_path() -> Path:
    return _research_paths()[0] / 'global.jsonl'


def _persona_dataset_path(persona_name: str) -> Path:
    return _research_paths()[1] / f'{normalize_name(persona_name) or "unknown_persona"}.jsonl'


def _session_dataset_path(session_id: str) -> Path:
    return _research_paths()[2] / f'{normalize_name(session_id) or "session"}.jsonl'


def _report_path(*, persona_name: str = '', session_id: str = '') -> Path:
    reports_dir = _research_paths()[3]
    if session_id:
        return reports_dir / f'session__{normalize_name(session_id)}.json'
    if persona_name:
        return reports_dir / f'persona__{normalize_name(persona_name)}.json'
    return reports_dir / 'global.json'


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


def _load_jsonl(path: Path, *, limit: int = _REPORT_SNAPSHOT_LIMIT) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    if limit > 0 and len(rows) > limit:
        return rows[-limit:]
    return rows


def _combined_features(user_features: dict[str, float], persona_features: dict[str, float]) -> dict[str, float]:
    distress = float(user_features.get('distress') or 0.0)
    anger = float(user_features.get('anger') or 0.0)
    curiosity = float(user_features.get('curiosity') or 0.0)
    help_need = float(user_features.get('help_need') or 0.0)
    celebration = float(user_features.get('celebration') or 0.0)
    persona_empathy = float(persona_features.get('empathy') or 0.0)
    persona_confidence = float(persona_features.get('confidence') or 0.0)
    persona_curiosity = float(persona_features.get('curiosity') or 0.0)
    persona_anger = float(persona_features.get('anger') or 0.0)
    return {
        'tension': min((anger + float(user_features.get('insult_pressure') or 0.0) + persona_anger) / 3.0, 1.0),
        'support_pull': min((distress + help_need + persona_empathy) / 3.0, 1.0),
        'challenge_pull': min((anger + persona_confidence + max(1.0 - persona_empathy, 0.0)) / 3.0, 1.0),
        'mentoring_pull': min((curiosity + persona_curiosity + persona_confidence) / 3.0, 1.0),
        'affiliation_pull': min((celebration + persona_empathy + float(user_features.get('trust_bid') or 0.0)) / 3.0, 1.0),
        'ambiguity': min(float(user_features.get('ambiguity') or 0.0), 1.0),
    }


def build_mood_snapshot(
    *,
    analysis: MessageAnalysis,
    persona_bundle: HeadBundle | None,
    social_role: SocialRoleDecision,
    response_style: str,
    session_id: str,
    source: str = 'chat_turn',
) -> MoodFeatureSnapshot:
    signals = dict(analysis.user_state.signals or {})
    user_features = {
        'distress': min(float(signals.get('contains_distress') or 0.0) + (0.18 if analysis.user_state.tone == 'distressed' else 0.0), 1.0),
        'anger': min(float(signals.get('contains_anger') or 0.0) + (0.12 if analysis.user_state.tone == 'angry' else 0.0), 1.0),
        'curiosity': min(float(signals.get('contains_question') or 0.0) + (0.1 if analysis.user_state.intent == 'question' else 0.0), 1.0),
        'help_need': min(float(signals.get('contains_help_request') or 0.0) + (0.12 if analysis.user_state.intent == 'seek_support' else 0.0), 1.0),
        'celebration': min(float(signals.get('contains_celebratory') or 0.0), 1.0),
        'moral_weight': min(float(signals.get('contains_moral_violation') or 0.0), 1.0),
        'trust_bid': min(float(signals.get('contains_persona_reference') or 0.0) * 0.55 + float(signals.get('contains_self_reference') or 0.0) * 0.25, 1.0),
        'ambiguity': 0.75 if analysis.user_state.intent == 'statement' and float(signals.get('contains_question') or 0.0) == 0.0 and analysis.situation.type == 'neutral_query' else 0.18,
        'insult_pressure': min(float(signals.get('contains_insult') or 0.0), 1.0),
    }
    persona_features = {
        'anger': float((persona_bundle.emotion_vector.get('anger') if persona_bundle else 0.0) or 0.0),
        'fear': float((persona_bundle.emotion_vector.get('fear') if persona_bundle else 0.0) or 0.0),
        'curiosity': float((persona_bundle.emotion_vector.get('curiosity') if persona_bundle else 0.0) or 0.0),
        'confidence': float((persona_bundle.emotion_vector.get('confidence') if persona_bundle else 0.0) or 0.0),
        'empathy': float((persona_bundle.emotion_vector.get('empathy') if persona_bundle else 0.0) or 0.0),
    }
    combined = _combined_features(user_features, persona_features)
    summary = (
        f"user={analysis.user_state.tone}/{analysis.user_state.intent}; "
        f"situation={analysis.situation.type}; role={social_role.role}; "
        f"support={combined['support_pull']:.2f}; challenge={combined['challenge_pull']:.2f}; "
        f"mentoring={combined['mentoring_pull']:.2f}"
    )
    return MoodFeatureSnapshot(
        snapshot_id=f'mood:{uuid4().hex[:12]}',
        session_id=session_id,
        persona_name=str(persona_bundle.name if persona_bundle else ''),
        source=source,
        language=analysis.user_state.language,
        user_features=user_features,
        persona_features=persona_features,
        combined_features=combined,
        selected_role=social_role.role,
        response_style=str(response_style or ''),
        situation_type=analysis.situation.type,
        summary=summary,
        created_at=_utc_now(),
    )


def record_mood_snapshot(snapshot: MoodFeatureSnapshot) -> None:
    payload = snapshot.to_dict()
    with _MOOD_RESEARCH_LOCK:
        _append_jsonl(_global_dataset_path(), payload)
        if snapshot.persona_name:
            _append_jsonl(_persona_dataset_path(snapshot.persona_name), payload)
        if snapshot.session_id:
            _append_jsonl(_session_dataset_path(snapshot.session_id), payload)


def _dominant_features(feature_map: dict[str, float], *, limit: int = 2, threshold: float = 0.36) -> list[str]:
    ranked = sorted(
        [(str(key), float(value or 0.0)) for key, value in dict(feature_map or {}).items() if float(value or 0.0) >= threshold],
        key=lambda item: (-item[1], item[0]),
    )
    return [name for name, _value in ranked[:limit]]


def _cluster_signature(snapshot: dict[str, Any]) -> tuple[str, str]:
    features = dict(snapshot.get('combined_features') or {})
    peaks = _dominant_features(features)
    if not peaks:
        return 'cluster:calm_baseline', 'calm baseline'
    key = 'cluster:' + '_plus_'.join(normalize_name(item) for item in peaks)
    label = ' + '.join(item.replace('_', ' ') for item in peaks)
    return key, label


def _centroid(rows: list[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    count = max(len(rows), 1)
    for row in rows:
        for key, value in dict(row.get('combined_features') or {}).items():
            sums[str(key)] += float(value or 0.0)
    return {key: round(total / count, 6) for key, total in sums.items()}


def _nearest_cluster(snapshot: dict[str, Any], clusters: list[MoodCluster]) -> tuple[str, str]:
    if not clusters:
        return '', ''
    features = dict(snapshot.get('combined_features') or {})
    ranked: list[tuple[float, str, str]] = []
    for cluster in clusters:
        keys = set(features) | set(cluster.centroid)
        distance = 0.0
        for key in keys:
            distance += abs(float(features.get(key) or 0.0) - float(cluster.centroid.get(key) or 0.0))
        ranked.append((distance, cluster.cluster_id, cluster.label))
    ranked.sort(key=lambda item: (item[0], item[1]))
    _distance, cluster_id, label = ranked[0]
    return cluster_id, label


def _transition_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: str(item.get('created_at') or ''))
    counter: Counter[tuple[str, str]] = Counter()
    last = ''
    for row in ordered:
        current, _ = _cluster_signature(row)
        if last and current:
            counter[(last, current)] += 1
        last = current
    return [
        {'from': src, 'to': dst, 'count': count}
        for (src, dst), count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ][:12]


def _role_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        role = str(row.get('selected_role') or '').strip()
        if role:
            grouped[role].append(row)
    effects: list[dict[str, Any]] = []
    for role, items in sorted(grouped.items()):
        centroid = _centroid(items)
        fit_score = max(float(centroid.get('support_pull') or 0.0), float(centroid.get('challenge_pull') or 0.0), float(centroid.get('mentoring_pull') or 0.0))
        effects.append(
            {
                'role': role,
                'count': len(items),
                'fit_score': round(fit_score, 6),
                'dominant_features': _dominant_features(centroid, limit=3, threshold=0.2),
            }
        )
    effects.sort(key=lambda item: (-int(item.get('count') or 0), str(item.get('role') or '')))
    return effects[:12]


def _regressions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    feature_names = sorted(
        {
            str(key)
            for row in rows
            for key in dict(row.get('combined_features') or {}).keys()
        }
    )
    roles = sorted({str(row.get('selected_role') or '').strip() for row in rows if str(row.get('selected_role') or '').strip()})
    results: list[dict[str, Any]] = []
    for role in roles:
        role_targets = [1.0 if str(row.get('selected_role') or '').strip() == role else 0.0 for row in rows]
        mean_target = sum(role_targets) / max(len(role_targets), 1)
        for feature in feature_names:
            xs = [float(dict(row.get('combined_features') or {}).get(feature) or 0.0) for row in rows]
            mean_x = sum(xs) / max(len(xs), 1)
            variance = sum((value - mean_x) ** 2 for value in xs)
            if variance <= 1e-9:
                continue
            covariance = sum((x - mean_x) * (y - mean_target) for x, y in zip(xs, role_targets, strict=False))
            slope = covariance / variance
            if abs(slope) < 0.12:
                continue
            results.append(
                {
                    'target': role,
                    'feature': feature,
                    'slope': round(slope, 6),
                    'direction': 'increases' if slope > 0 else 'decreases',
                }
            )
    results.sort(key=lambda item: (-abs(float(item.get('slope') or 0.0)), str(item.get('target') or ''), str(item.get('feature') or '')))
    return results[:16]


def analyze_mood_research(*, persona_name: str = '', session_id: str = '') -> MoodResearchReport:
    if session_id:
        rows = _load_jsonl(_session_dataset_path(session_id))
        scope = f'session:{normalize_name(session_id)}'
    elif persona_name:
        rows = _load_jsonl(_persona_dataset_path(persona_name))
        scope = f'persona:{normalize_name(persona_name)}'
    else:
        rows = _load_jsonl(_global_dataset_path())
        scope = 'global'
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        cluster_id, label = _cluster_signature(row)
        groups[cluster_id].append(row)
        labels[cluster_id] = label
    clusters = [
        MoodCluster(
            cluster_id=cluster_id,
            label=labels.get(cluster_id, cluster_id.replace('cluster:', '').replace('_', ' ')),
            centroid=_centroid(items),
            feature_peaks=_dominant_features(_centroid(items), limit=3, threshold=0.2),
            size=len(items),
            examples=[str(item.get('summary') or '') for item in items[:3] if str(item.get('summary') or '').strip()],
        )
        for cluster_id, items in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    latest = rows[-1] if rows else {}
    latest_cluster_id, latest_cluster_label = _nearest_cluster(latest, clusters) if latest else ('', '')
    report = MoodResearchReport(
        scope=scope,
        snapshot_count=len(rows),
        latest_language=str(latest.get('language') or 'en'),
        latest_cluster_id=latest_cluster_id,
        latest_cluster_label=latest_cluster_label,
        clusters=clusters[:12],
        transition_counts=_transition_counts(rows),
        classification={
            'latest_snapshot_id': str(latest.get('snapshot_id') or ''),
            'selected_role': str(latest.get('selected_role') or ''),
            'response_style': str(latest.get('response_style') or ''),
        },
        regressions=_regressions(rows),
        role_effects=_role_effects(rows),
        updated_at=_utc_now(),
    )
    return report


def refresh_mood_reports(*, persona_name: str = '', session_id: str = '') -> dict[str, Any]:
    global_report = analyze_mood_research()
    write_json(_report_path(), global_report.to_dict())
    result = {'global': str(_report_path())}
    if persona_name:
        persona_report = analyze_mood_research(persona_name=persona_name)
        path = _report_path(persona_name=persona_name)
        write_json(path, persona_report.to_dict())
        result['persona'] = str(path)
    if session_id:
        session_report = analyze_mood_research(session_id=session_id)
        path = _report_path(session_id=session_id)
        write_json(path, session_report.to_dict())
        result['session'] = str(path)
    return result


def schedule_mood_research_refresh(*, persona_name: str = '', session_id: str = '') -> None:
    def _runner() -> None:
        try:
            refresh_mood_reports(persona_name=persona_name, session_id=session_id)
        except Exception:
            return

    with _MOOD_RESEARCH_LOCK:
        if _MOOD_RESEARCH_EXECUTOR_CLOSED:
            return
        try:
            _MOOD_RESEARCH_EXECUTOR.submit(_runner)
        except RuntimeError:
            return


def load_mood_report(*, persona_name: str = '', session_id: str = '') -> MoodResearchReport | None:
    payload = load_json(_report_path(persona_name=persona_name, session_id=session_id), {})
    if not isinstance(payload, dict) or not payload:
        return None
    return MoodResearchReport(
        scope=str(payload.get('scope') or 'global'),
        snapshot_count=int(payload.get('snapshot_count') or 0),
        latest_language=str(payload.get('latest_language') or 'en'),
        latest_cluster_id=str(payload.get('latest_cluster_id') or ''),
        latest_cluster_label=str(payload.get('latest_cluster_label') or ''),
        clusters=[
            MoodCluster(
                cluster_id=str(item.get('cluster_id') or ''),
                label=str(item.get('label') or ''),
                centroid=dict(item.get('centroid') or {}),
                feature_peaks=[str(value).strip() for value in list(item.get('feature_peaks') or []) if str(value).strip()],
                size=int(item.get('size') or 0),
                examples=[str(value).strip() for value in list(item.get('examples') or []) if str(value).strip()],
            )
            for item in list(payload.get('clusters') or [])
            if isinstance(item, dict)
        ],
        transition_counts=[dict(item) for item in list(payload.get('transition_counts') or []) if isinstance(item, dict)],
        classification=dict(payload.get('classification') or {}),
        regressions=[dict(item) for item in list(payload.get('regressions') or []) if isinstance(item, dict)],
        role_effects=[dict(item) for item in list(payload.get('role_effects') or []) if isinstance(item, dict)],
        updated_at=str(payload.get('updated_at') or ''),
    )
