"""Пример бэкенда команды для стенда хакатона Sain x Школа 21."""

import os

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

TEAM = os.environ.get("TEAM", "unknown")

app = FastAPI(title=f"hackathon {TEAM}")


@app.get("/")
def root() -> dict[str, str]:
    """Главная: показывает, какой команде принадлежит сервис."""
    return {"team": TEAM, "status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    """Проверка живости для организаторов и мониторинга."""
    return {"team": TEAM, "status": "ok"}


class RunRequest(BaseModel):
    """Что стенд присылает в теле POST /run."""

    case: int = Field(ge=1, le=4)  # номер кейса, 1..4
    dataset_code: str = Field(min_length=1, max_length=128)  # код набора этапа (online или final)


@app.post("/run")
def run(req: RunRequest, authorization: str | None = Header(default=None)) -> JSONResponse:
    """Онлайн-проверка 20.11 и финал 22.11: этот метод вызывает сам стенд, без участия команды.

    Что придёт:
      - тело {"case": N, "dataset_code": "<код набора>"};
      - заголовок Authorization: Bearer <токен проверки>. Это не HACK_TEAM_TOKEN: токен действует
        только на время проверки и только для наборов этого этапа.

    Что сделать:
      1. Взять данные: GET {HACK_API_URL}/case/{case}/getData?dataset_code={dataset_code}
         с тем же заголовком Authorization, что пришёл сюда. Один раз, весь набор целиком.
      2. Посчитать ответ кейса (нейросеть — через LLM_BASE_URL, LLM_API_KEY, LLM_MODEL из env).

    Что вернуть: сам ответ кейса в теле этого ответа — HTTP 200, JSON в том же формате, что принимает
    POST {HACK_API_URL}/case/{case}/validate. В API ответ не сдавайте: стенд сохранит его сам.
    Стенд ждёт до 45 минут, потом попытка засчитывается как таймаут. В online одна попытка на кейс.

    Не печатайте токен, данные набора и ответы в логи и не возвращайте их в ответе.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"status": "error", "error": "нет Authorization: Bearer"})
    # Здесь ваша логика кейса req.case. Пока её нет — честно отвечаем 501, а не «done».
    return JSONResponse(status_code=501, content={"status": "not_implemented", "case": req.case})
