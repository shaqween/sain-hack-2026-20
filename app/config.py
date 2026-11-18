"""Все настройки в одном месте. Секреты берутся ТОЛЬКО из env, в код их не пишем."""

import os

TEAM = os.environ.get("TEAM", "local")

# API хакатона
HACK_API_URL = os.environ.get("HACK_API_URL", "https://hack-api.be-healthy.tech").rstrip("/")
HACK_TEAM_TOKEN = os.environ.get("HACK_TEAM_TOKEN", "")
DEV_DATASET_CODE = "case4-dev-v2"  # из README стенда, раздел «Наборы данных»

# Нейросеть
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

# Режим нейросети
LLM_MODE = os.environ.get("LLM_MODE") or ("real" if LLM_API_KEY else "stub")

LLM_MAX_RPM = 25            # лимит прокси — 30 запросов/мин на команду, оставляем запас
LLM_CONCURRENCY = 3         # параллельных вызовов в /run
LLM_MAX_TOKENS = 2600       # ответ трёх экспертов ~900 токенов + ~500 скрытых рассуждений + запас
LLM_TIMEOUT_S = 120

# Бюджет токенов на чат 
CHAT_TOKEN_BUDGET = int(os.environ.get("CHAT_TOKEN_BUDGET", "60000"))

# run 
RUN_DEADLINE_S = 40 * 60    # стенд ждёт 45 минут; после 40 оставшееся добиваем заглушкой

# RAG
TOP_K = 4                   # сколько разделов-кандидатов показывать нейросети
SNIPPET_CHARS = 700         # сколько символов раздела отдавать в промпт
