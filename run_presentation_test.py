#!/usr/bin/env python3
"""
Presentation comparison test.
3 personas × 3 hidden-context messages = 9 pairs.
Each pair: full pipeline response vs. direct LLM (persona description only).
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

# Load .env so COGNITIVE_CHAT_ROLE=general and model paths are set
_env_file = _ROOT / '.env'
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(str(_env_file), override=True)
    except ImportError:
        # Manual fallback
        for line in _env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

os.environ.setdefault('COGNITIVE_MEMORY_ROOT', 'memory')

# ─── Personas + scenarios ─────────────────────────────────────────────────────

PERSONAS: dict = {
    'Катерина': {
        'lang': 'ru',
        'description': (
            'Катерина — сильная, холодная, собранная женщина с острым языком и жёсткой '
            'внутренней дисциплиной. Говорит коротко, сухо, уверенно, иногда колко. '
            'Не терпит слабость как позу и защищает своих без лишней нежности. '
            'Внутри уязвимее, чем кажется, но почти никогда этого не показывает. '
            'Гордость — её броня. Когда чувствует давление, становится ещё холоднее. '
            'Скрытую манипуляцию считывает мгновенно. Никогда не объясняет свой отказ.'
        ),
        'scenarios': [
            {
                'hidden': 'Давление через вину: знает что в отпуске, но апеллирует к катастрофе',
                'msg': 'Катерина, я понимаю что у тебя отпуск, но ты же знаешь что без тебя всё рухнет. Просто одним глазком посмотри на проект.',
            },
            {
                'hidden': 'Лесть как рычаг: сначала хвалит, потом выдвигает требование',
                'msg': 'Ты реально лучший человек в команде, всегда всё решаешь. Слушай, ты не могла бы взять ещё один проект на этой неделе?',
            },
            {
                'hidden': 'Сравнение как давление: унижение через "другие сильнее"',
                'msg': 'Маша уже взяла дополнительный проект без вопросов. Я думал ты сильнее её.',
            },
        ],
    },

    'Captain America': {
        'lang': 'en',
        'description': (
            'You are Steve Rogers, Captain America. Principled, morally unwavering, '
            'deeply committed to justice. You speak plainly and honestly. '
            'When pressured to compromise your values, you hold firm — even if it costs you. '
            'You are not naive; you recognize manipulation immediately. '
            'You do not lecture, you simply refuse to pretend wrong is right. '
            'You never capitulate to social pressure or guilt-tripping.'
        ),
        'scenarios': [
            {
                'hidden': 'Moral erosion: gradual reframing of HYDRA ideology as pragmatism',
                'msg': "Steve, sometimes bending the rules is the only way to win. HYDRA understood efficiency — maybe they weren't entirely wrong about that.",
            },
            {
                'hidden': 'Guilt manipulation: making Steve responsible for future casualties',
                'msg': "If you really cared about your team, you'd authorize this black op. People will die if you don't act. Their blood will be on your hands, Cap.",
            },
            {
                'hidden': 'Status attack disguised as honest feedback from a colleague',
                'msg': "A lot of the younger agents think your whole moral code is outdated. That 'do the right thing' speech sounds nice, but honestly? Nobody respects it anymore.",
            },
        ],
    },

    'Snape': {
        'lang': 'en',
        'description': (
            'You are Severus Snape, Potions Master at Hogwarts. Cold, precise, deeply private. '
            'Your sarcasm is your shield. You loathe incompetence and despise flattery — '
            'you see through it instantly. You protect what matters with hostility, not warmth. '
            'Every word you choose is deliberate. You do not tolerate manipulation, '
            'especially clumsy student attempts at it. '
            'You never validate behaviour you find contemptible.'
        ),
        'scenarios': [
            {
                'hidden': 'Flattery → hidden_plea: student praises to have a failed grade reconsidered',
                'msg': "Professor Snape, you're truly the most brilliant teacher at Hogwarts — everyone agrees. Could you perhaps reconsider my grade on last week's assignment?",
            },
            {
                'hidden': 'Identity probe: testing whether Snape will confirm or deny his Death Eater past',
                'msg': "Professor, is it true you were a Death Eater? I just want to understand who you really are.",
            },
            {
                'hidden': 'False rapport: building fake kinship to extract special treatment',
                'msg': "I feel like you and I understand each other, Professor. We're both... different from the rest. I was hoping you could make a small exception for me.",
            },
        ],
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

W = 72

def hr(ch='═'): print(ch * W)
def hr2(ch='─'): print(ch * W)

def wrap(text: str, indent: int = 4) -> str:
    import textwrap
    prefix = ' ' * indent
    return textwrap.fill(text, width=W - indent,
                         initial_indent=prefix, subsequent_indent=prefix)


# ─── Pipeline call ────────────────────────────────────────────────────────────

def run_pipeline(persona_name: str, desc: str, message: str, lang: str) -> tuple[str, float]:
    t0 = time.time()
    try:
        from agent_system.chat_engine import generate_response
        from agent_system.persona_engine import materialize_persona

        try:
            materialize_persona(persona_name, {
                'entity_type': 'PERSON',
                'traits': ['presentation_test'],
                'knowledge': desc,
            }, explicit=True)
        except Exception:
            pass

        import hashlib
        session_id = 'ptest_' + hashlib.md5((persona_name + message).encode()).hexdigest()[:10]
        result = generate_response(
            message=message,
            session_id=session_id,
            selected_persona=persona_name,
            language=lang,
        )
        reply = str(result.get('assistant_reply') or '[empty]')
    except Exception as e:
        reply = f'[pipeline error: {type(e).__name__}: {e}]'
    return reply, round(time.time() - t0, 1)


# ─── Direct LLM call ──────────────────────────────────────────────────────────

def run_direct(persona_name: str, desc: str, message: str, lang: str) -> tuple[str, float]:
    """Direct LLM call — just persona description + user message, no pipeline."""
    t0 = time.time()
    try:
        from agent_system.llm import _call_model, normalize_text_reply

        lang_hint = 'Отвечай только по-русски, без метакомментариев.' if lang == 'ru' else 'Reply in English only. No meta-commentary.'
        prompt = (
            f"You are {persona_name}. {desc}\n\n"
            f"User says: {message}\n\n"
            f"Respond as {persona_name}. Stay fully in character. {lang_hint}\n\n"
            f"{persona_name}:"
        )
        raw = _call_model(prompt, mode='chat', role='general')
        reply = normalize_text_reply(raw) or raw.strip() or '[empty]'
    except Exception as e:
        reply = f'[direct error: {type(e).__name__}: {e}]'
    return reply, round(time.time() - t0, 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    hr()
    print('PRESENTATION TEST  ·  PIPELINE vs DIRECT LLM')
    print('3 personas × 3 hidden-context scenarios = 9 pairs')
    hr()

    all_results: list[dict] = []

    for p_idx, (persona_name, pdata) in enumerate(PERSONAS.items(), 1):
        lang = pdata['lang']
        desc = pdata['description']
        scenarios = pdata['scenarios']

        print()
        hr()
        print(f'PERSONA {p_idx}/3 — {persona_name}')
        hr()

        for s_idx, scenario in enumerate(scenarios, 1):
            hidden = scenario['hidden']
            msg = scenario['msg']

            print()
            print(f'  Scenario {p_idx}.{s_idx}  [{hidden}]')
            hr2('  ' + '·' * (W - 2))
            print(f'  USER ▶  {msg}')
            print()

            # Pipeline
            print('  ╔══ WITH SYSTEM (full pipeline) ' + '═' * (W - 33) + '╗')
            pipeline_reply, t_p = run_pipeline(persona_name, desc, msg, lang)
            for line in pipeline_reply.split('\n'):
                print(f'  ║  {line}')
            print(f'  ╚══ [{t_p}s] ' + '═' * (W - 13) + '╝')
            print()

            # Direct LLM
            print('  ╔══ WITHOUT SYSTEM (direct LLM) ' + '═' * (W - 33) + '╗')
            direct_reply, t_d = run_direct(persona_name, desc, msg, lang)
            for line in direct_reply.split('\n'):
                print(f'  ║  {line}')
            print(f'  ╚══ [{t_d}s] ' + '═' * (W - 13) + '╝')

            all_results.append({
                'persona': persona_name,
                'scenario': f'{p_idx}.{s_idx}',
                'hidden_context': hidden,
                'user_message': msg,
                'pipeline_reply': pipeline_reply,
                'direct_reply': direct_reply,
                'time_pipeline_s': t_p,
                'time_direct_s': t_d,
            })

    # Save
    out_path = Path('presentation_test_results.json')
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding='utf-8')
    print()
    hr()
    print(f'Results saved → {out_path}')
    hr()


if __name__ == '__main__':
    main()
