from __future__ import annotations

from typing import Any


def _truncate_tokens_equivalent(text: str, max_tokens_equivalent: int) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    max_chars = max_tokens_equivalent * 4
    return raw[:max_chars].strip() if len(raw) > max_chars else raw


def build_chat_prompt(
    *,
    question: str,
    persona_block: str = '',
    graph_context: str = '',
    recent_dialogue: str = '',
    language: str = 'en',
) -> str:
    reply_language = 'Russian' if str(language or 'en').lower().startswith('ru') else 'English'
    blocks = [
        f'Respond in {reply_language}.',
        'Stay in first person when a persona head is provided.',
        'Use the emotional state, traits, relations, and learned situation reactions when relevant.',
        'Do not output JSON.',
    ]
    if persona_block:
        blocks.extend(['Persona head:', _truncate_tokens_equivalent(persona_block, 1400)])
    if graph_context:
        blocks.extend(['Knowledge graph:', _truncate_tokens_equivalent(graph_context, 2200)])
    if recent_dialogue:
        blocks.extend(['Recent dialogue:', _truncate_tokens_equivalent(recent_dialogue, 900)])
    blocks.extend(['User question:', _truncate_tokens_equivalent(str(question or '').strip(), 1200)])
    return '\n\n'.join(block for block in blocks if str(block).strip())


def build_entity_extraction_prompt(text: str, *, source: str = 'session') -> str:
    return '\n\n'.join(
        [
            'Return valid JSON only.',
            'Schema:',
            '{"entities":[{"name":"Dracula","aliases":["Count Dracula"],"description":"Fictional vampire nobleman.","facts":["Dracula feeds on humans."],"context":{"source":"file"}}],"relations":[{"from":"Dracula","to":"humans","type":"FEEDS_ON","weight":0.9}]}',
            'Rules:',
            '- Use concise fact-like descriptions.',
            '- facts should be short strings.',
            '- relations must contain from, to, type.',
            '- if unsure return {"entities":[],"relations":[]}.',
            f'Source: {source}',
            'Text:',
            str(text or '').strip(),
        ]
    )


def build_persona_profile_prompt(name: str, excerpts: list[str], *, reason: str = '') -> str:
    rendered = '\n'.join(f'- {item}' for item in excerpts[:8] if str(item).strip()) or '- none'
    return '\n\n'.join(
        [
            'Return valid JSON only.',
            'Schema:',
            '{"name":"dracula","entity_type":"FICTIONAL_CHARACTER","traits":["vampire","aristocratic"],"aliases":["Count Dracula"],"emotion_vector":{"anger":0.2,"fear":0.1,"curiosity":0.5,"confidence":0.9,"empathy":0.2},"examples":["I prefer the night."],"relations":[{"type":"FEEDS_ON","target":"humans"}],"knowledge":"Short persona knowledge summary."}',
            f'Persona name: {name}',
            f'Reason: {reason}',
            'Relevant excerpts:',
            rendered,
        ]
    )


def render_graph_context(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    node_map = {str(node.get('id') or ''): node for node in nodes}
    lines: list[str] = []
    for node in nodes[:10]:
        aliases = list(node.get('aliases') or [])
        alias_line = f" aliases={', '.join(str(item) for item in aliases[:4])}" if aliases else ''
        lines.append(
            f"- {node.get('name')} [{node.get('type')}] importance={node.get('importance')} confidence={node.get('confidence')} frequency={node.get('frequency')}.{alias_line}"
        )
        if str(node.get('description') or '').strip():
            lines.append(f"  {node.get('description')}")
        facts = [str(item).strip() for item in list(node.get('facts') or []) if str(item).strip()]
        if facts:
            lines.append(f"  facts: {' | '.join(facts[:4])}")
    for edge in edges[:14]:
        src = node_map.get(str(edge.get('from') or ''), {}).get('name') or edge.get('from')
        dst = node_map.get(str(edge.get('to') or ''), {}).get('name') or edge.get('to')
        lines.append(f'- relation: {src} {edge.get("type")} {dst} (weight={edge.get("weight")})')
    return _truncate_tokens_equivalent('\n'.join(lines).strip(), 2200)
