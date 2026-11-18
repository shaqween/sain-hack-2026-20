"""Пакетный прогон набора: то, что делает POST /run на онлайн-показе и в финале."""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import config
from .agent import answer_group
from .contract import Answer, answers_payload, parse_articles, parse_requests, unwrap_get_data
from .experts import resolve_expert
from .kb import KnowledgeBase

log = logging.getLogger("runner")


def fetch_dataset(dataset_code: str, authorization: str, case: int = 4) -> dict:
    """GET getData с тем же Authorization, что пришёл в /run. URL и токен в лог не пишем."""
    url = f"{config.HACK_API_URL}/case/{case}/getData?dataset_code={urllib.parse.quote(dataset_code)}"
    req = urllib.request.Request(url, headers={"Authorization": authorization})
    last = None
    for pause in (3, 6, 12, None):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # 5xx/обрыв — повторяем
            last = e
            log.warning("getData: %s", type(e).__name__)
            if pause:
                time.sleep(pause)
    raise RuntimeError(f"getData failed: {type(last).__name__}")


def run_dataset(ds: dict, kb: KnowledgeBase | None = None) -> tuple[dict, KnowledgeBase]:
    """Посчитать ответы на весь набор. Возвращает (тело ответа /run, база этого набора)."""
    t0 = time.monotonic()
    if kb is None:
        kb = KnowledgeBase()
        kb.load(parse_articles(ds), source=f"dataset:{(ds.get('meta') or {}).get('dataset_id', '?')}")
    experts_meta = []
    requests = parse_requests(ds)

    # группируем тройки: один и тот же текст, адресованный разным экспертам
    groups: dict[str, list] = {}
    for r in requests:
        groups.setdefault(" ".join(r.text.lower().split()), []).append(r)

    answers: dict[str, Answer] = {}

    def work(items):
        allow_llm = time.monotonic() - t0 < config.RUN_DEADLINE_S  # время на исходе → заглушка
        persona = {r.request_id: resolve_expert(r.expert_id, experts_meta) for r in items}
        eids = list(dict.fromkeys(persona.values()))
        try:
            g = answer_group(kb, items[0].text, eids, allow_llm=allow_llm)
        except Exception as e:
            log.exception("group failed: %s", type(e).__name__)
            g = answer_group(kb, items[0].text, eids, allow_llm=False)
        for r in items:
            rep = g.replies[persona[r.request_id]]
            answers[r.request_id] = Answer(r.request_id, r.expert_id, rep.text, rep.refused, rep.citations)

    with ThreadPoolExecutor(max_workers=config.LLM_CONCURRENCY) as pool:
        list(pool.map(work, groups.values()))

    ordered = [answers[r.request_id] for r in requests]  # ответ на КАЖДЫЙ запрос, в исходном порядке
    log.info("run: %d запросов, %d групп, %.1f с", len(requests), len(groups), time.monotonic() - t0)
    return answers_payload(ds, ordered), kb


def run_remote(dataset_code: str, authorization: str, case: int = 4) -> tuple[dict, KnowledgeBase, dict]:
    payload = fetch_dataset(dataset_code, authorization, case)
    ds, _ref = unwrap_get_data(payload)
    body, kb = run_dataset(ds)
    return body, kb, ds
