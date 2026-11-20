"""Состояние процесса: активная база, сессии чата, кеш ответов. Всё в памяти —
на стенде контейнер пересоздаётся при каждом деплое, а писать можно только в /tmp."""

from __future__ import annotations

import glob
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict

from . import config
from .contract import parse_articles, unwrap_get_data
from .kb import KnowledgeBase

log = logging.getLogger("state")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

kb = KnowledgeBase()
final_user: dict | None = None  # в финале getData отдаёт одного пользователя — кладём сюда


def load_initial_base() -> None:
    """Старт: dev-набор через API (нужен HACK_TEAM_TOKEN) плюс статьи из data/extra_articles."""
    source_articles, source = [], "empty"
    if config.HACK_TEAM_TOKEN:
        try:
            from .runner import fetch_dataset
            payload = fetch_dataset(config.DEV_DATASET_CODE, "Bearer " + config.HACK_TEAM_TOKEN)
            ds, _ = unwrap_get_data(payload)
            source_articles, source = parse_articles(ds), "api:" + config.DEV_DATASET_CODE
        except Exception as e:
            log.warning("не удалось загрузить dev-набор: %s — в базе только добавленные статьи", type(e).__name__)
    kb.load(source_articles, source)
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "extra_articles", "*.json"))):
        with open(path, encoding="utf-8") as f:
            for a in parse_articles({"articles": [json.load(f)]}):
                kb.add_article(a)
    log.info("база загружена: %s", kb.stats())


def load_media() -> dict[str, list[dict]]:
    path = os.path.join(ROOT, "data", "media.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        items = json.load(f).get("items", [])
    out: dict[str, list[dict]] = {}
    for it in items:
        out.setdefault(f"{it['article_id']}#{it.get('section_id', '')}", []).append(it)
    return out


media = load_media()


class Sessions:
    """session_id → {expert_id, history}. Смена эксперта = новая сессия (так проще фронтенду)."""

    def __init__(self, limit: int = 2000) -> None:
        self._d: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self.limit = limit

    def get_or_create(self, session_id: str | None, expert_id: str) -> tuple[str, dict]:
        with self._lock:
            s = self._d.get(session_id or "")
            if s is None or s["expert_id"] != expert_id:
                session_id = uuid.uuid4().hex
                s = {"expert_id": expert_id, "history": [], "created": time.time()}
                self._d[session_id] = s
                while len(self._d) > self.limit:
                    self._d.popitem(last=False)
            return session_id, s


sessions = Sessions()


class Budget:
    """Сколько токенов уже съел чат. Нужен, чтобы клиент не выел лимит, общий с /run."""

    def __init__(self, limit: int) -> None:
        self.limit, self.used = limit, 0

    def ok(self) -> bool:
        return self.used < self.limit


chat_budget = Budget(config.CHAT_TOKEN_BUDGET)
answer_cache: OrderedDict[str, dict] = OrderedDict()
