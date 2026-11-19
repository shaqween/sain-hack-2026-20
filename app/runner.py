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
from .retrieve import retrieve_batch

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
    chat_history_raw = ds.get("chat_history", [])
    user_histories = {}
    for msg in chat_history_raw:
        uid = msg.get("user_id")
        if uid not in user_histories:
            user_histories[uid] = []
        who = "Пользователь" if msg.get("role") == "user" else "Ассистент Sain"  # агент приложения, не наш эксперт
        user_histories[uid].append((who, msg.get("text", "")))
    # группируем тройки: один вопрос, адресованный разным экспертам. ключ — group_id,
    groups: dict[str, list] = {}
    for r in requests:
        key = r.group_id or " ".join(r.text.lower().split())
        groups.setdefault(key, []).append(r)

    # разделы выбираем на группу, а не на запрос: если ссылки у тройки разойдутся, V = 0.
    # поиск — один вызов на весь набор (статьи на английском, вопросы на русском — см. app/retrieve.py).
    # модели отдаём короткие id вместо ключей групп: ключ может быть целым текстом вопроса
    qid_to_key = {f"q{i}": key for i, key in enumerate(groups, 1)}
    picks = retrieve_batch(kb, {qid: groups[key][0].text for qid, key in qid_to_key.items()})
    cands_by_key: dict[str, list] = {}
    for qid, key in qid_to_key.items():
        sections = (kb.section(aid, sid) for aid, sid in picks.get(qid, []))
        cands_by_key[key] = [s for s in sections if s]

    answers: dict[str, Answer] = {}

    def work(kv):
        key, items = kv
        allow_llm = time.monotonic() - t0 < config.RUN_DEADLINE_S  # время на исходе → заглушка
        persona = {r.request_id: resolve_expert(r.expert_id, experts_meta) for r in items}
        eids = list(dict.fromkeys(persona.values()))
        cands = cands_by_key.get(key) or None  # пусто → None: агент возьмёт свой BM25

        uid = items[0].user_id
        history_for_user = user_histories.get(uid, [])
        try:
            g = answer_group(kb, items[0].text, eids, history=history_for_user, 
                             allow_llm=allow_llm, cands_override=cands)
        except Exception as e:
            log.exception("group failed: %s", type(e).__name__)
            g = answer_group(kb, items[0].text, eids, history=history_for_user, 
                             allow_llm=False, cands_override=cands)
        for r in items:
            rep = g.replies[persona[r.request_id]]
            answers[r.request_id] = Answer(r.request_id, r.expert_id, rep.text, rep.refused, rep.citations)

    with ThreadPoolExecutor(max_workers=config.LLM_CONCURRENCY) as pool:
        list(pool.map(work, groups.items()))

    ordered = [answers[r.request_id] for r in requests]  # ответ на каждый запрос, в исходном порядке
    log.info("run: %d запросов, %d групп, %.1f с", len(requests), len(groups), time.monotonic() - t0)
    return answers_payload(ordered), kb


def run_remote(dataset_code: str, authorization: str, case: int = 4) -> tuple[dict, KnowledgeBase, dict]:
    payload = fetch_dataset(dataset_code, authorization, case)
    ds, _ref = unwrap_get_data(payload)
    body, kb = run_dataset(ds)
    return body, kb, ds
