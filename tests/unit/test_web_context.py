from unittest.mock import patch

from src.web.web_context import fetch_random_wikipedia_article, search_wikipedia_articles


@patch("src.web.web_context._http_json")
def test_fetch_random_wikipedia_article_returns_summary(mock_http_json):
    mock_http_json.return_value = {
        "title": "Кофе",
        "extract": "Кофе — напиток из обжаренных зёрен.",
        "content_urls": {"desktop": {"page": "https://ru.wikipedia.org/wiki/Кофе"}},
    }

    result = fetch_random_wikipedia_article("ru")

    assert result["ok"] is True
    assert result["language"] == "ru"
    assert result["article"]["title"] == "Кофе"
    assert "обжаренных" in result["article"]["snippet"]


@patch("src.web.web_context._http_json")
def test_search_wikipedia_articles_builds_results(mock_http_json):
    mock_http_json.side_effect = [
        ["coffee", ["Кофе", "Эспрессо"], [], ["https://ru.wikipedia.org/wiki/Кофе", "https://ru.wikipedia.org/wiki/Эспрессо"]],
        {
            "title": "Кофе",
            "extract": "Кофе — напиток.",
            "content_urls": {"desktop": {"page": "https://ru.wikipedia.org/wiki/Кофе"}},
        },
        {
            "title": "Эспрессо",
            "extract": "Эспрессо — способ приготовления кофе.",
            "content_urls": {"desktop": {"page": "https://ru.wikipedia.org/wiki/Эспрессо"}},
        },
    ]

    result = search_wikipedia_articles("кофе", language="ru", limit=2)

    assert result["ok"] is True
    assert result["query"] == "кофе"
    assert len(result["items"]) == 2
    assert result["items"][0]["title"] == "Кофе"
    assert result["items"][1]["title"] == "Эспрессо"
