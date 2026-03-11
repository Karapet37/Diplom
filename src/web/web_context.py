from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

_STOPWORDS = {
    "the", "and", "with", "from", "that", "this", "what", "when", "where", "which", "about",
    "как", "что", "это", "если", "для", "или", "при", "без", "надо", "есть", "быть", "через",
}


def _extract_terms(text: str, limit: int = 3) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in str(text or "").replace("\n", " ").split():
        token = "".join(ch for ch in raw if ch.isalpha() or ch.isdigit() or ch in {"-", "_"}).strip("-_").lower()
        if len(token) < 4 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def _http_json(url: str, timeout: float = 4.0) -> Any:
    request = Request(url, headers={"User-Agent": "AutographCognitiveWorkspace/1.0"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = response.read().decode("utf-8", errors="ignore")
    return json.loads(payload)


def _normalize_wikipedia_language(language: str) -> str:
    value = "".join(ch for ch in str(language or "").lower().strip() if ch.isalpha() or ch == "-")
    if not value:
        return "ru"
    if len(value) > 12:
        return "ru"
    return value


def _wikipedia_summary(language: str, title: str) -> dict[str, str]:
    clean_title = " ".join(str(title or "").split()).strip()
    if not clean_title:
        return {}
    payload = _http_json(
        f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{quote(clean_title)}"
    )
    extract = " ".join(str(payload.get("extract", "") or "").split()).strip()
    return {
        "title": str(payload.get("title", clean_title) or clean_title).strip(),
        "url": str(payload.get("content_urls", {}).get("desktop", {}).get("page", "") or "").strip(),
        "snippet": extract[:800],
        "language": language,
    }


def fetch_random_wikipedia_article(language: str = "ru") -> dict[str, Any]:
    lang = _normalize_wikipedia_language(language)
    payload = _http_json(f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary")
    extract = " ".join(str(payload.get("extract", "") or "").split()).strip()
    return {
        "ok": True,
        "language": lang,
        "article": {
            "title": str(payload.get("title", "") or "").strip(),
            "url": str(payload.get("content_urls", {}).get("desktop", {}).get("page", "") or "").strip(),
            "snippet": extract[:800],
            "source": "wikipedia_random",
            "language": lang,
        },
    }


def search_wikipedia_articles(query: str, *, language: str = "ru", limit: int = 5) -> dict[str, Any]:
    text = " ".join(str(query or "").split()).strip()
    if not text:
        return {"ok": True, "language": _normalize_wikipedia_language(language), "items": []}
    lang = _normalize_wikipedia_language(language)
    safe_limit = max(1, min(int(limit or 5), 8))
    payload = _http_json(
        "https://"
        f"{lang}.wikipedia.org/w/api.php?action=opensearch&search={quote(text)}&limit={safe_limit}&namespace=0&format=json"
    )
    titles = payload[1] if isinstance(payload, list) and len(payload) > 1 and isinstance(payload[1], list) else []
    urls = payload[3] if isinstance(payload, list) and len(payload) > 3 and isinstance(payload[3], list) else []
    items: list[dict[str, str]] = []
    for index, raw_title in enumerate(titles[:safe_limit]):
        title = " ".join(str(raw_title or "").split()).strip()
        if not title:
            continue
        item = _wikipedia_summary(lang, title)
        if not item:
            continue
        if not item.get("url") and index < len(urls):
            item["url"] = str(urls[index] or "").strip()
        item["source"] = "wikipedia_search"
        items.append(item)
    return {
        "ok": True,
        "language": lang,
        "query": text,
        "items": items,
    }


def collect_web_context(message: str, *, limit: int = 3) -> dict[str, Any]:
    terms = _extract_terms(message, limit=limit)
    snippets: list[dict[str, str]] = []
    warnings: list[str] = []
    for term in terms:
        try:
            search_url = (
                "https://en.wikipedia.org/w/api.php?action=opensearch"
                f"&search={quote(term)}&limit=1&namespace=0&format=json"
            )
            search_payload = _http_json(search_url)
            titles = search_payload[1] if isinstance(search_payload, list) and len(search_payload) > 1 else []
            title = str(titles[0] or "").strip() if titles else ""
            if not title:
                continue
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
            summary_payload = _http_json(summary_url)
            extract = " ".join(str(summary_payload.get("extract", "") or "").split()).strip()
            if not extract:
                continue
            snippets.append(
                {
                    "source": "wikipedia",
                    "title": title,
                    "url": str(summary_payload.get("content_urls", {}).get("desktop", {}).get("page", "") or ""),
                    "snippet": extract[:420],
                }
            )
        except Exception as exc:  # network or response failure must not break runtime
            warnings.append(f"{term}: {exc}")
    return {
        "enabled": True,
        "terms": terms,
        "snippets": snippets[:limit],
        "warning": "; ".join(warnings[:3]) if warnings else "",
    }
