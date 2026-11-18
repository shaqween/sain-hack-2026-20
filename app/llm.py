"""Клиент нейросети

Написан на urllib, чтобы не тянуть пакет openai: протокол тот же самый —
POST {LLM_BASE_URL}/chat/completions с заголовком Authorization: Bearer <ключ>.

В режиме stub метод chat_json() возвращает None, и агент строит ответ шаблоном из найденного
раздела. Тот же путь используется как запасной, если нейросеть ответила ошибкой или 429.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request

from . import config

log = logging.getLogger("llm")


class RateLimiter:
    """Не больше N запросов за скользящие 60 секунд на весь процесс."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < 60]
                if len(self._calls) < self.per_minute:
                    self._calls.append(now)
                    return
                sleep = 60 - (now - self._calls[0]) + 0.05
            time.sleep(sleep)


class LLMClient:
    def __init__(self) -> None:
        self.mode = config.LLM_MODE
        self.limiter = RateLimiter(config.LLM_MAX_RPM)
        self.tokens_used = 0
        self.calls = 0
        self.failures = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.mode == "real" and bool(config.LLM_API_KEY and config.LLM_BASE_URL and config.LLM_MODEL)

    def chat_json(self, messages: list[dict], max_tokens: int | None = None) -> dict | None:
        """Вызов модели, ожидаем JSON-объект в ответе. None — использовать заглушку."""
        if not self.enabled:
            return None
        body = json.dumps({
            "model": config.LLM_MODEL,          # имя модели только из env
            "messages": messages,
            "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
            "temperature": 0.4,
        }).encode()
        for attempt, pause in enumerate([5, 10, 20, None]):
            self.limiter.wait()
            req = urllib.request.Request(
                config.LLM_BASE_URL + "/chat/completions", data=body, method="POST",
                headers={"Authorization": "Bearer " + config.LLM_API_KEY, "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT_S) as r:
                    resp = json.load(r)
                with self._lock:
                    self.calls += 1
                    self.tokens_used += int((resp.get("usage") or {}).get("total_tokens", 0))
                choice = resp["choices"][0]
                if choice.get("finish_reason") == "length":
                    log.warning("ответ обрезан по max_tokens — поднимите LLM_MAX_TOKENS")
                parsed = extract_json(choice["message"].get("content") or "")
                if parsed is not None:
                    return parsed
                log.warning("модель вернула не-JSON (попытка %s)", attempt + 1)
            except urllib.error.HTTPError as e:
                log.warning("LLM HTTP %s (попытка %s)", e.code, attempt + 1)  # тело/ключ не логируем
                if e.code in (400, 401, 403):
                    break  # повтор не поможет
            except Exception as e:  # таймаут, обрыв
                log.warning("LLM %s (попытка %s)", type(e).__name__, attempt + 1)
            if pause is None:
                break
            time.sleep(pause)
        with self._lock:
            self.failures += 1
        return None


def extract_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                v = json.loads(m.group(0))
                return v if isinstance(v, dict) else None
            except json.JSONDecodeError:
                return None
    return None


llm = LLMClient()
