"""
Маршруты для фронтенда — под /api, маршрут проверки — POST /run (его вызывает стенд).
Ядро (app/*) написано на стандартной библиотеке, здесь только HTTP-слой на FastAPI.
"""

from __future__ import annotations

import hashlib
import logging
import threading

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


from app import config, state
from app.agent import answer_group
from app.contract import parse_articles
from app.experts import EXPERTS, public_card
from app.llm import llm
from app.runner import run_remote

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # не пишем URL запросов в лог

app = FastAPI(title=f"Sain case 4 — {config.TEAM}")
# клиент в финале запускается на компьютере организаторов → другой origin; cookies не используем
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup() -> None:
    # сервер отвечает сразу, база грузится в фоне (стенд ждёт ответ на GET / не дольше 60 с)
    threading.Thread(target=state.load_initial_base, daemon=True).start()


# служебное
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

#@app.get("/")
#def root() -> dict:
#    return {"team": config.TEAM, "status": "ok", "service": "sain-case4"}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_mode": "real" if llm.enabled else "stub",
        "kb": state.kb.stats(),
        "llm_calls": llm.calls,
        "llm_tokens": llm.tokens_used,
        "chat_tokens": state.chat_budget.used,
    }


#экран 1: выбор эксперта

@app.get("/api/experts")
def experts() -> list[dict]:
    return [public_card(eid) for eid in EXPERTS]


#экран 2: разговор

class ChatIn(BaseModel):
    expert_id: str = Field(description="clinician | biohacker | nutritionist")
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


def _citation_view(aid: str, sid: str) -> dict:
    a = state.kb.articles.get(aid)
    s = state.kb.section(aid, sid)
    return {
        "article_id": aid,
        "section_id": sid,
        "article_title": a.title if a else aid,
        "section_title": s.title if s else sid,
        "quote": (s.text[:280] + "…") if s and len(s.text) > 280 else (s.text if s else ""),
        "api_url": f"/api/articles/{aid}#{sid}",     # фронтенд открывает свою страницу статьи на этом разделе
        "source_url": a.source_url if a else "",     # оригинал (DOI) — вторичная ссылка «читать в журнале»
        "license": a.license if a else "",
        "authors": a.authors if a else "",
    }


@app.post("/api/chat")
def chat(body: ChatIn) -> dict:
    if body.expert_id not in EXPERTS:
        raise HTTPException(404, "unknown expert")
    if not state.kb.articles:
        raise HTTPException(503, "база ещё загружается, повторите через пару секунд")
    sid, sess = state.sessions.get_or_create(body.session_id, body.expert_id)

    key = hashlib.sha1(f"{body.expert_id}|{body.message.strip().lower()}|{len(sess['history'])==0}".encode()).hexdigest()
    cached = state.answer_cache.get(key)
    if cached:
        reply_view = cached
    else:
        before = llm.tokens_used
        g = answer_group(state.kb, body.message, [body.expert_id], history=sess["history"],
                         allow_llm=state.chat_budget.ok())
        state.chat_budget.used += llm.tokens_used - before
        r = g.replies[body.expert_id]
        reply_view = {
            "text": r.text,
            "refused": r.refused,
            "citations": [_citation_view(a, s) for a, s in r.citations],
            "related_media": [m for a, s in r.citations for m in state.media.get(f"{a}#{s}", [])],
            "source": r.source,
        }
        if not sess["history"]:  # кешируем только первые реплики: без контекста они одинаковые
            state.answer_cache[key] = reply_view
            while len(state.answer_cache) > 500:
                state.answer_cache.popitem(last=False)

    sess["history"] += [("Пользователь", body.message), ("Эксперт", reply_view["text"])]
    return {"session_id": sid, "expert": public_card(body.expert_id), "reply": reply_view}


#статьи (горячие ссылки ведут сюда)

@app.get("/api/articles")
def articles() -> list[dict]:
    return [{"article_id": a.article_id, "title": a.title, "authors": a.authors, "year": a.year,
             "sections": len(a.sections), "source_url": a.source_url, "license": a.license}
            for a in state.kb.articles.values()]


@app.get("/api/articles/{article_id}")
def article(article_id: str) -> dict:
    a = state.kb.articles.get(article_id)
    if not a:
        raise HTTPException(404, "article not found")
    return {
        "article_id": a.article_id, "title": a.title, "authors": a.authors, "year": a.year,
        "license": a.license, "source_url": a.source_url,
        "attribution": f"{a.authors + '. ' if a.authors else ''}{a.title}. Лицензия {a.license}.",
        "sections": [{"section_id": s.section_id, "title": s.title, "text": s.text,
                      "media": state.media.get(f"{a.article_id}#{s.section_id}", [])} for s in a.sections],
    }


class ArticleIn(BaseModel):
    article: dict  # тот же формат, что у статьи в наборе (см. data/demo_base.json)


@app.post("/api/admin/articles")
def add_article(body: ArticleIn, authorization: str | None = Header(default=None)) -> dict:
    """Добавить статью «на лету». Доступ — по командному токену (он есть только у команды)."""
    # локально (токена в env нет) — открыто; на стенде — только с командным токеном
    if config.HACK_TEAM_TOKEN and authorization != "Bearer " + config.HACK_TEAM_TOKEN:
        raise HTTPException(401, "admin only")
    parsed = parse_articles({"articles": [body.article]})
    if not parsed or not parsed[0].sections:
        raise HTTPException(400, "в статье нет разделов")
    state.kb.add_article(parsed[0])
    return {"ok": True, "kb": state.kb.stats()}


#онлайн-проверка и финал

class RunRequest(BaseModel):
    case: int = Field(ge=1, le=4)
    dataset_code: str = Field(min_length=1, max_length=128)

@app.post("/run")
def run(req: RunRequest, authorization: str | None = Header(default=None)) -> JSONResponse:
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"status": "error", "error": "нет Authorization: Bearer"})
    if req.case != 4:
        return JSONResponse(status_code=400, content={"status": "error", "error": "команда решает кейс 4"})
    body, kb, ds = run_remote(req.dataset_code, authorization, req.case)
    # финал: после ответа бэкенд продолжает работать на базе этого набора, жюри задаёт вопросы
    state.kb.load(list(kb.articles.values()), kb.source)
    state.final_user = ds.get("user") or ds.get("users")
    return JSONResponse(status_code=200, content=body)
