"""
dialog_tracker.py — управление контекстными матрицами диалогов.

Singleton-уровень: хранит per-session DialogContextMatrix в памяти,
и один DialogPatternClusterer (персистируется на диск).

Использование из chat_engine:
    from .dialog_tracker import record_turn, get_logic_context

    # после вычисления P-вектора:
    logic_ctx = record_turn(session_id, text, speaker='user')
    built['dialog_context'] = logic_ctx
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .dialog_context_matrix import DialogContextMatrix, UtteranceRow
from .dialog_pattern_clusterer import DialogPatternClusterer

_MODELS_DIR = Path(os.environ.get(
    'DIALOG_PATTERN_MODELS_DIR',
    str(Path(__file__).parent.parent / 'models' / 'dialog_patterns'),
))
_CLUSTERER_PATH = _MODELS_DIR / 'clusterer.pkl'

# ── In-memory state ───────────────────────────────────────────────────────────

_SESSION_MATRICES: dict[str, DialogContextMatrix] = {}
_CLUSTERER: DialogPatternClusterer | None = None


def _get_clusterer() -> DialogPatternClusterer:
    global _CLUSTERER
    if _CLUSTERER is None:
        if _CLUSTERER_PATH.exists():
            try:
                _CLUSTERER = DialogPatternClusterer.load(_CLUSTERER_PATH)
            except Exception:
                _CLUSTERER = _make_default_clusterer()
        else:
            _CLUSTERER = _make_default_clusterer()
    return _CLUSTERER


def _make_default_clusterer() -> DialogPatternClusterer:
    return DialogPatternClusterer(
        window_size=4,
        min_cluster_size=5,
        min_samples_to_fit=40,
    )


def _save_clusterer_async() -> None:
    """Сохранение в фоне — не блокируем основной тред."""
    import threading
    def _save() -> None:
        try:
            _MODELS_DIR.mkdir(parents=True, exist_ok=True)
            _CLUSTERER.save(_CLUSTERER_PATH)
        except Exception:
            pass
    threading.Thread(target=_save, daemon=True).start()


# ── Публичный API ─────────────────────────────────────────────────────────────

def record_turn(
    session_id: str,
    text: str,
    speaker: str,                        # 'user' | 'persona'
    p_scores: dict[str, float] | None = None,
    p_dominant: dict[str, str] | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    """
    Добавляет ход в матрицу диалога и возвращает dialog_context для built-dict.

    Если p_scores не передан — вычисляет через P-реестр (fast, keyword rules).
    Возвращает:
        {
            'logic_score': float,       # [0..1], 1=логично, 0=аномалия
            'is_logical': bool,
            'noise_ratio': float,
            'pattern_label': str,       # кластер если найден
            'session_length': int,      # реплик в текущей сессии
        }
    """
    if not session_id:
        return _empty_ctx()

    # Проход 1: F1-F49 без кластерного контекста
    _f7_subtext: str = ''
    _f6_directive: str = ''
    if p_scores is None:
        _ctx: dict = {}
        p_scores, p_dominant = _compute_p_scores(text, extra_context=_ctx)
        _f7_subtext   = str(_ctx.get('_f7_subtext') or '')
        _f6_directive = str(_ctx.get('_f6_directive') or '')
    elif p_dominant is None:
        p_dominant = {}

    # Матрица сессии
    if session_id not in _SESSION_MATRICES:
        _SESSION_MATRICES[session_id] = DialogContextMatrix(session_id=session_id)

    mat = _SESSION_MATRICES[session_id]
    mat.add(UtteranceRow(
        text=text,
        speaker=speaker,
        timestamp=ts or time.time(),
        p_scores=p_scores,
        p_dominant=p_dominant,
    ))

    # Контекстная матрица для P39/P51:
    # history   — тексты предыдущих ходов
    # timestamps — временны́е метки предыдущих ходов
    # p_history  — P-скоры предыдущих ходов ((text)(ts)(49 метрик))
    prev_rows = mat._rows[:-1][-8:]  # до 8 предыдущих
    history    = [r.text for r in prev_rows]
    timestamps = [r.timestamp for r in prev_rows]
    p_history  = [r.p_scores for r in prev_rows]

    clusterer = _get_clusterer()

    # Оценка логичности + кластерная информация для P50/P51
    cluster_size = -1
    cluster_id   = -1
    if len(mat) >= clusterer.window_size:
        score_result = clusterer.logic_score(mat)
        is_logical   = score_result['is_logical']
        noise_ratio  = score_result['noise_window_ratio']
        patterns     = score_result.get('patterns', [])
        pattern_label = patterns[0].get('label', '') if patterns else ''
        logic_score  = max(0.0, 1.0 - noise_ratio)

        # Кластер последнего окна → P50/P51 контекст
        cids = score_result.get('cluster_ids', [])
        if cids:
            cluster_id = cids[-1]
            if cluster_id != -1 and cluster_id in clusterer._patterns:
                cluster_size = clusterer._patterns[cluster_id].size
    else:
        is_logical    = True
        noise_ratio   = 0.0
        pattern_label = ''
        logic_score   = 1.0

    # Проход 2: пересчитываем P39/P50/P51 с полным контекстом
    # Матрица: ((text)(ts)(49 метрик)) — источник логики P51
    cluster_ctx = {
        'cluster_id':   cluster_id,
        'cluster_size': cluster_size,
        'noise_ratio':  noise_ratio,
        'history':      history,       # тексты предыдущих ходов
        'timestamps':   timestamps,    # временны́е метки
        'p_history':    p_history,     # P-скоры предыдущих ходов
    }
    try:
        from .p_subsystem_registry import get_p_registry
        reg = get_p_registry()
        for pid in ('F39', 'F50', 'F51'):
            fam = reg.family(pid)
            if fam:
                out = fam.forward(text, cluster_ctx)
                p_scores[pid]   = out.dominant_score
                p_dominant[pid] = out.dominant
                # Обновляем последнюю строку матрицы
                mat._rows[-1].p_scores[pid]   = out.dominant_score
                mat._rows[-1].p_dominant[pid] = out.dominant
    except Exception:
        pass

    # Анализ эмоционального состояния пользователя по контекстной матрице
    user_affect: dict[str, Any] = {}
    if speaker == 'user':
        try:
            from .user_affect_model import analyze_user_affect
            snapshot = analyze_user_affect(
                matrix_rows=mat._rows,
                current_text=text,
                current_p_scores=p_scores,
                current_p_dominant=p_dominant,
                current_ts=ts or time.time(),
            )
            user_affect = snapshot.to_dict()
        except Exception:
            pass

    return {
        'logic_score':    round(logic_score, 3),
        'is_logical':     is_logical,
        'noise_ratio':    round(noise_ratio, 3),
        'pattern_label':  pattern_label,
        'session_length': len(mat),
        # Для ClusterVariant (F50/F51)
        'cluster_id':     cluster_id,
        'cluster_size':   cluster_size,
        # Эмоциональное состояние пользователя (только для user-ходов)
        'user_affect':    user_affect,
        # F7 прагматический подтекст — для инжекции в route_guidance
        'f7_subtext':   _f7_subtext,
        'f6_directive': _f6_directive,
    }


def flush_session(session_id: str) -> None:
    """
    Завершает сессию: добавляет её матрицу в clusterer и сохраняет.
    Вызывать при завершении диалога или смене персоны.
    """
    if session_id not in _SESSION_MATRICES:
        return
    mat = _SESSION_MATRICES.pop(session_id)
    if len(mat) < 2:
        return
    clusterer = _get_clusterer()
    clusterer.add_dialog(mat)
    _save_clusterer_async()


def get_session_matrix(session_id: str) -> DialogContextMatrix | None:
    return _SESSION_MATRICES.get(session_id)


def clusterer_status() -> dict[str, Any]:
    return _get_clusterer().status()


def reset_session(session_id: str) -> None:
    _SESSION_MATRICES.pop(session_id, None)


# ── Вспомогательные ───────────────────────────────────────────────────────────

def _compute_p_scores(
    text: str,
    extra_context: dict | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """
    Вычисляет P-скоры через реестр.
    extra_context передаётся в P50/P51 ClusterVariant (cluster_size, cluster_id).
    """
    try:
        from .p_subsystem_registry import get_p_registry
        reg = get_p_registry()
        ctx = dict(extra_context) if extra_context else {}
        outputs = reg.compute(text, context=ctx, parallel=False)
        p_scores  = {pid: out.dominant_score for pid, out in outputs.items()}
        p_dominant = reg.to_legacy_vector(outputs)
        return p_scores, p_dominant
    except Exception:
        return {}, {}


def _empty_ctx() -> dict[str, Any]:
    return {
        'logic_score': 1.0,
        'is_logical': True,
        'noise_ratio': 0.0,
        'pattern_label': '',
        'session_length': 0,
    }
