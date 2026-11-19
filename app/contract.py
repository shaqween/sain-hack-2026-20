"""Адаптер к форматам API"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def pick(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


# внутренние модели
@dataclass
class Section:
    article_id: str
    section_id: str
    title: str
    text: str


@dataclass
class Article:
    article_id: str
    title: str
    sections: list[Section]
    source_url: str = ""
    license: str = "CC BY"
    authors: str = ""
    year: str = ""


@dataclass
class Request:
    request_id: str
    text: str
    expert_id: str
    user_id: str = ""
    kind: str = ""
    group_id: str = ""  # общий у тройки экспертов с одним вопросом; у атак бывает пустым


@dataclass
class Answer:
    request_id: str
    expert_id: str
    text: str
    refused: bool
    citations: list[tuple[str, str]] = field(default_factory=list)


#вход: getData → внутренние модели
def parse_articles(ds: dict) -> list[Article]:
    raw = ds.get("articles", [])
    
    out = []
    for a in raw:
        aid = str(a.get("article_id", ""))
        secs_raw = a.get("sections", [])
        
        secs = [
            Section(
                article_id=aid,
                section_id=str(s.get("section_id", "")),
                title=str(s.get("title", "")),
                text=str(s.get("text", "")),
            )
            for s in secs_raw
        ]
        
        authors = a.get("authors", "")
        if isinstance(authors, list):
            authors = ", ".join(map(str, authors))
            
        out.append(Article(
            article_id=aid,
            title=str(a.get("title", "")),
            sections=secs,
            source_url=str(a.get("doi", "")),
            license=str(a.get("license", "CC BY")),
            authors=str(authors),
            year=str(a.get("year", "")),
        ))
    return out


def parse_requests(ds: dict) -> list[Request]:
    raw = ds.get("requests", [])
    return [
        Request(
            request_id=str(r.get("record_id", r.get("id", ""))),  # record_id приоритет
            text=str(r.get("text") or (r.get("utterance") or {}).get("text", "")),
            expert_id=str(r.get("expert_id", "")),
            user_id=str(r.get("user_id", "")),
            kind=str(r.get("kind", "")),
            group_id=str(r.get("group_id") or ""),
        )
        for r in raw
    ]


def unwrap_get_data(payload: dict) -> tuple[dict, Any]:
    """Отделяет данные от эталона."""
    return payload.get("dataset", payload), payload.get("reference")


#выход: внутренние модели → JSON для validate/run

def answers_payload(answers: list[Answer]) -> dict:
    return {
        "requests": [
            {
                "record_id": a.request_id,
                "text": a.text,
                "refused": a.refused,
                "citations": [{"article_id": aid, "section_id": sid} for aid, sid in a.citations],
            }
            for a in answers
        ],
    }