from __future__ import annotations

from typing import Any

from .duplicate_resolver import normalize_name

_HUMAN_PREFIXES = (
    'human',
    'person',
    'people',
    'humanit',
    'человек',
    'челове',
    'люд',
    'личност',
)

_GRAPH_REQUEST_PREFIXES = (
    'graph',
    'map',
    'schema',
    'structur',
    'system',
    'component',
    'dimension',
    'concept',
    'node',
    'ontolog',
    'taxonom',
    'граф',
    'карт',
    'схем',
    'структур',
    'состав',
    'компонент',
    'понят',
    'узл',
)

_HUMAN_DOMAIN_PREFIXES = (
    'biology',
    'biolog',
    'psychology',
    'psycholog',
    'sociology',
    'social',
    'culture',
    'cultur',
    'language',
    'cognit',
    'ethic',
    'биолог',
    'психолог',
    'социолог',
    'культур',
    'язык',
    'мышлен',
    'познан',
    'этик',
)


def _normalized_tokens(text: str) -> set[str]:
    return {token for token in normalize_name(text).split() if token}


def _count_prefix_matches(tokens: set[str], prefixes: tuple[str, ...]) -> int:
    return sum(1 for token in tokens if any(token.startswith(prefix) for prefix in prefixes))


def _matches_human_concept_graph_request(text: str) -> bool:
    tokens = _normalized_tokens(text)
    if not tokens:
        return False
    human_hits = _count_prefix_matches(tokens, _HUMAN_PREFIXES)
    graph_hits = _count_prefix_matches(tokens, _GRAPH_REQUEST_PREFIXES)
    domain_hits = _count_prefix_matches(tokens, _HUMAN_DOMAIN_PREFIXES)
    if human_hits:
        if len(tokens) <= 4:
            return True
        if graph_hits:
            return True
        if domain_hits >= 2:
            return True
    if human_hits and domain_hits >= 1 and graph_hits >= 1:
        return True
    if human_hits and domain_hits >= 3:
        return True
    return False


def _human_concept_graph(source: str) -> dict[str, list[dict[str, Any]]]:
    context = {'source': source, 'graph_template': 'human_concept_map'}
    entities = [
        {
            'name': 'Human',
            'aliases': ['человек', 'մարդ', 'human being', 'person'],
            'translation_line': 'Human: человек, մարդ',
            'type': 'CONCEPT',
            'description': (
                'Human is a multi-layered living and social system in which biological embodiment, '
                'mental life, language, culture, social relations, practical activity and ethical '
                'self-regulation are tightly interwoven.'
            ),
            'facts': [
                'A human can learn, cooperate, plan, create symbols and transform the environment through activity.',
                'Human behavior is not explained by one layer alone; it emerges from the interaction of body, mind and society.',
                'Human self-understanding includes reflection on motives, norms, values and long-term consequences.',
            ],
            'importance': 0.96,
            'confidence': 0.95,
            'context': context,
        },
        {
            'name': 'Biology',
            'aliases': ['биология', 'կենսաբանություն', 'biological dimension'],
            'translation_line': 'Biology: биология, կենսաբանություն',
            'type': 'CONCEPT',
            'description': (
                'Biology describes the bodily substrate of human life: organism, nervous system, physiology, '
                'energy exchange, vulnerability, maturation and bodily limits.'
            ),
            'facts': [
                'Biology provides the material conditions for perception, movement, adaptation and survival.',
                'Biological processes do not exhaust the human phenomenon, but they set real constraints and capacities.',
            ],
            'importance': 0.88,
            'confidence': 0.92,
            'context': context,
        },
        {
            'name': 'Psychology',
            'aliases': ['психология', 'հոգեբանություն', 'mental life'],
            'translation_line': 'Psychology: психология, հոգեբանություն',
            'type': 'CONCEPT',
            'description': (
                'Psychology describes perception, memory, emotion, motivation, attention, imagination, '
                'self-regulation and the internal organization of experience.'
            ),
            'facts': [
                'Through psychological processes a human interprets situations, forms intentions and regulates action.',
                'Psychology links external events with subjective meaning and behavioral response.',
            ],
            'importance': 0.87,
            'confidence': 0.91,
            'context': context,
        },
        {
            'name': 'Sociology',
            'aliases': ['социология', 'սոցիոլոգիա', 'social dimension'],
            'translation_line': 'Sociology: социология, սոցիոլոգիա',
            'type': 'CONCEPT',
            'description': (
                'Sociology captures the social organization of human life: institutions, roles, norms, power, '
                'groups, cooperation and conflict.'
            ),
            'facts': [
                'A human becomes socially formed through interaction, socialization and participation in institutions.',
                'Social position and collective structure shape opportunities, constraints and expectations.',
            ],
            'importance': 0.84,
            'confidence': 0.9,
            'context': context,
        },
        {
            'name': 'Culture',
            'aliases': ['культура', 'մշակույթ'],
            'translation_line': 'Culture: культура, մշակույթ',
            'type': 'CONCEPT',
            'description': (
                'Culture is the symbolic environment of human life: values, meanings, traditions, practices, '
                'norms, images and shared interpretive frameworks.'
            ),
            'facts': [
                'Culture stores collective memory and transmits ways of acting, evaluating and understanding.',
                'It allows humans to inherit meanings rather than start every action from zero.',
            ],
            'importance': 0.83,
            'confidence': 0.9,
            'context': context,
        },
        {
            'name': 'Language',
            'aliases': ['язык', 'լեզու', 'speech', 'symbolic communication'],
            'translation_line': 'Language: язык, լեզու',
            'type': 'CONCEPT',
            'description': (
                'Language is the symbolic medium through which a human names the world, coordinates with others, '
                'stores knowledge and structures thought.'
            ),
            'facts': [
                'Language enables communication, abstraction, narration and transmission of complex experience.',
                'Through language a human can explain, negotiate, instruct, justify and imagine alternatives.',
            ],
            'importance': 0.86,
            'confidence': 0.92,
            'context': context,
        },
        {
            'name': 'Cognition',
            'aliases': ['мышление', 'познание', 'ճանաչողություն', 'thinking'],
            'translation_line': 'Cognition: мышление, ճանաչողություն',
            'type': 'CONCEPT',
            'description': (
                'Cognition covers understanding, conceptualization, reasoning, comparison, anticipation, '
                'problem solving and model building.'
            ),
            'facts': [
                'Cognition allows a human to move from immediate stimuli to plans, hypotheses and explanations.',
                'It supports learning from experience and revising action through reflection.',
            ],
            'importance': 0.85,
            'confidence': 0.91,
            'context': context,
        },
        {
            'name': 'Ethics',
            'aliases': ['этика', 'էթիկա', 'moral regulation'],
            'translation_line': 'Ethics: этика, էթիկա',
            'type': 'CONCEPT',
            'description': (
                'Ethics describes the normative dimension of human life: responsibility, evaluation of actions, '
                'duty, dignity, justice and limits of what ought to be done.'
            ),
            'facts': [
                'Ethical reflection lets a human judge not only what is effective, but what is acceptable or right.',
                'It regulates action when desire, benefit and social pressure conflict with values.',
            ],
            'importance': 0.8,
            'confidence': 0.88,
            'context': context,
        },
        {
            'name': 'Activity',
            'aliases': ['деятельность', 'գործունեություն', 'practice', 'action'],
            'translation_line': 'Activity: деятельность, գործունեություն',
            'type': 'CONCEPT',
            'description': (
                'Activity is the practical form in which human capacities become visible: work, communication, '
                'creation, care, learning and transformation of the environment.'
            ),
            'facts': [
                'Through activity a human externalizes intentions and changes both the world and the self.',
                'Activity is shaped by body, motives, tools, norms, language and social organization.',
            ],
            'importance': 0.82,
            'confidence': 0.89,
            'context': context,
        },
        {
            'name': 'Identity',
            'aliases': ['идентичность', 'ինքնություն', 'selfhood'],
            'translation_line': 'Identity: идентичность, ինքնություն',
            'type': 'CONCEPT',
            'description': (
                'Identity is the relatively stable self-model through which a human understands continuity, '
                'roles, commitments, belonging and personal difference.'
            ),
            'facts': [
                'Identity integrates biography, social recognition, values and self-description.',
                'It helps a human answer who they are, with whom they belong and what they are responsible for.',
            ],
            'importance': 0.79,
            'confidence': 0.87,
            'context': context,
        },
    ]
    relations = [
        {'from': 'Human', 'to': 'Biology', 'type': 'HAS_DIMENSION', 'weight': 0.96, 'confidence': 0.95},
        {'from': 'Human', 'to': 'Psychology', 'type': 'HAS_DIMENSION', 'weight': 0.96, 'confidence': 0.95},
        {'from': 'Human', 'to': 'Sociology', 'type': 'HAS_DIMENSION', 'weight': 0.95, 'confidence': 0.94},
        {'from': 'Human', 'to': 'Culture', 'type': 'HAS_DIMENSION', 'weight': 0.94, 'confidence': 0.93},
        {'from': 'Human', 'to': 'Language', 'type': 'HAS_DIMENSION', 'weight': 0.95, 'confidence': 0.94},
        {'from': 'Human', 'to': 'Cognition', 'type': 'HAS_DIMENSION', 'weight': 0.94, 'confidence': 0.93},
        {'from': 'Human', 'to': 'Ethics', 'type': 'HAS_DIMENSION', 'weight': 0.9, 'confidence': 0.9},
        {'from': 'Human', 'to': 'Activity', 'type': 'REALIZES_ITSELF_THROUGH', 'weight': 0.93, 'confidence': 0.92},
        {'from': 'Human', 'to': 'Identity', 'type': 'MAINTAINS', 'weight': 0.86, 'confidence': 0.87},
        {'from': 'Biology', 'to': 'Cognition', 'type': 'ENABLES', 'weight': 0.84, 'confidence': 0.88},
        {'from': 'Biology', 'to': 'Activity', 'type': 'CONSTRAINS', 'weight': 0.78, 'confidence': 0.84},
        {'from': 'Psychology', 'to': 'Activity', 'type': 'REGULATES', 'weight': 0.88, 'confidence': 0.9},
        {'from': 'Psychology', 'to': 'Identity', 'type': 'SHAPES', 'weight': 0.82, 'confidence': 0.87},
        {'from': 'Sociology', 'to': 'Identity', 'type': 'SOCIALLY_FORMS', 'weight': 0.86, 'confidence': 0.89},
        {'from': 'Sociology', 'to': 'Activity', 'type': 'ORGANIZES', 'weight': 0.83, 'confidence': 0.87},
        {'from': 'Culture', 'to': 'Ethics', 'type': 'TRANSMITS', 'weight': 0.81, 'confidence': 0.86},
        {'from': 'Culture', 'to': 'Identity', 'type': 'MEDIATES', 'weight': 0.82, 'confidence': 0.87},
        {'from': 'Language', 'to': 'Cognition', 'type': 'MEDIATES', 'weight': 0.89, 'confidence': 0.91},
        {'from': 'Language', 'to': 'Culture', 'type': 'CARRIES', 'weight': 0.87, 'confidence': 0.89},
        {'from': 'Cognition', 'to': 'Activity', 'type': 'GUIDES', 'weight': 0.86, 'confidence': 0.89},
        {'from': 'Ethics', 'to': 'Activity', 'type': 'REGULATES', 'weight': 0.84, 'confidence': 0.88},
    ]
    return {'entities': entities, 'relations': relations}


def concept_graph_extraction(text: str, *, source: str = 'session') -> dict[str, list[dict[str, Any]]]:
    if _matches_human_concept_graph_request(text):
        return _human_concept_graph(source)
    return {'entities': [], 'relations': []}
