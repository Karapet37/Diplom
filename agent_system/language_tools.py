from __future__ import annotations

import re

_ARMENIAN_RE = re.compile(r'[Ա-֏ﬓ-ﬗ]')
_CYRILLIC_RE = re.compile(r'[Ѐ-ӿ]')

# Three supported languages only
SUPPORTED_LANGUAGES = ('en', 'ru', 'hy')

LANGUAGE_LABELS = {
    'en': 'English',
    'ru': 'Russian',
    'hy': 'Armenian',
}


def detect_language_code(text: str, *, fallback: str = 'en') -> str:
    raw = str(text or '').strip()
    if not raw:
        return fallback
    if _ARMENIAN_RE.search(raw):
        return 'hy'
    if _CYRILLIC_RE.search(raw):
        return 'ru'
    return 'en'


def normalize_language_code(value: str, *, fallback: str = 'en') -> str:
    token = str(value or '').strip().lower()
    aliases = {
        'auto':       fallback,
        'english':    'en',
        'английский': 'en',
        'russian':    'ru',
        'русский':    'ru',
        'armenian':   'hy',
        'армянский':  'hy',
        'հայերեն':    'hy',
    }
    # Also match Armenian-script word for Armenian
    if _ARMENIAN_RE.search(token):
        return 'hy'
    normalized = aliases.get(token)
    if normalized:
        return normalized
    if token in LANGUAGE_LABELS:
        return token
    return fallback


def language_label(code: str) -> str:
    normalized = normalize_language_code(code, fallback='en')
    return LANGUAGE_LABELS.get(normalized, 'English')
