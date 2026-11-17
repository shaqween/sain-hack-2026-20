#!/usr/bin/env bash
# Точка входа контейнера: сервер должен слушать 0.0.0.0 и порт из переменной PORT.
# uvicorn держит долгий запрос /run (до 45 минут) без дополнительных настроек.
set -euo pipefail
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --no-access-log
