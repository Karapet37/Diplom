from __future__ import annotations

import re

_ARMENIAN_RE = re.compile(r'[\u0531-\u058F]')
_CYRILLIC_RE = re.compile(r'[\u0400-\u04FF]')
_CJK_RE = re.compile(r'[\u4E00-\u9FFF]')

LANGUAGE_LABELS = {
    'en': 'English',
    'ru': 'Russian',
    'hy': 'Armenian',
    'zh': 'Chinese',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'tr': 'Turkish',
}


def detect_language_code(text: str, *, fallback: str = 'en') -> str:
    raw = str(text or '').strip()
    if not raw:
        return fallback
    if _ARMENIAN_RE.search(raw):
        return 'hy'
    if _CYRILLIC_RE.search(raw):
        return 'ru'
    if _CJK_RE.search(raw):
        return 'zh'
    return 'en'


def normalize_language_code(value: str, *, fallback: str = 'en') -> str:
    token = str(value or '').strip().lower()
    aliases = {
        'auto': fallback,
        'english': 'en',
        'английский': 'en',
        'russian': 'ru',
        'русский': 'ru',
        'armenian': 'hy',
        'армянский': 'hy',
        'հայերեն': 'hy',
        'chinese': 'zh',
        'french': 'fr',
        'français': 'fr',
        'german': 'de',
        'deutsch': 'de',
        'spanish': 'es',
        'español': 'es',
    }
    normalized = aliases.get(token)
    if normalized:
        return normalized
    if token in LANGUAGE_LABELS:
        return token
    if re.fullmatch(r'[a-z]{2,5}(?:-[a-z]{2,5})?', token):
        return token
    return fallback


def language_label(code: str) -> str:
    normalized = normalize_language_code(code, fallback='en')
    if normalized in LANGUAGE_LABELS:
        return LANGUAGE_LABELS[normalized]
    if normalized:
        return normalized.upper()
    return 'English'
