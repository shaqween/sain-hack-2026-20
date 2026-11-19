"""Сравнение способов поиска на dev по эталону.

Для каждой группы проверяем, ведут ли найденные разделы в допустимые статьи эталона
(allowed_article_ids). Правило то же, что в оценке: ответ засчитывается, только если
допустимы все его ссылки.

Способы:
- ru  — BM25 по русскому вопросу как есть (так искал код до перехода на оглавление);
- en  — перевод вопроса в английские ключевые слова + BM25;
- toc — модель выбирает разделы по оглавлению базы.
Промпты en и toc берутся из app/retrieve.py: замеряется тот же код, что работает в /run.

Запуск из корня репозитория (нужен ключ нейросети; en + toc — около 17 тыс. токенов):
  LLM_MODE=real python scripts/eval_retrieval.py --data local/dev.json
  LLM_MODE=real python scripts/eval_retrieval.py --data local/dev.json --only toc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# При запуске `python scripts/...` Python ищет модули в папке scripts/, а не в корне,
# поэтому `import app` не сработает. Добавляем корень репозитория в путь поиска.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.contract import parse_articles, unwrap_get_data  # noqa: E402  (импорт после правки sys.path)
from app.kb import KnowledgeBase  # noqa: E402
from app.llm import llm  # noqa: E402
from app.retrieve import Pick, retrieve_toc, translate_queries  # noqa: E402

TOP = 2  # столько ссылок берёт /run (per_q в retrieve_batch, MAX_CITATIONS в agent.py)


def group_key(r: dict) -> str:
    """Ключ группы: group_id, а если его нет (у атак бывает null) — нормализованный текст вопроса."""
    if r.get("group_id"):
        return "g:" + str(r["group_id"])
    return "t:" + " ".join(r["utterance"]["text"].lower().split())


def build_groups(ds: dict, ref: dict) -> list[dict]:
    """Одна запись на группу: текст, вид, статьи, допустимые для ВСЕХ запросов группы."""
    allowed = {x["record_id"]: set(x["allowed_article_ids"]) for x in ref["requests"]}
    kind = {x["record_id"]: x["request_type"] for x in ref["requests"]}
    groups: dict[str, dict] = {}
    for r in ds["requests"]:
        k = group_key(r)
        g = groups.get(k)
        if g is None:
            groups[k] = {"id": f"q{len(groups) + 1}", "text": r["utterance"]["text"],
                         "kind": kind.get(r["record_id"], "?"), "allowed": set(allowed.get(r["record_id"], set()))}
        else:
            g["allowed"] &= allowed.get(r["record_id"], set())
    return list(groups.values())


def _all_allowed(found: list[Pick], allowed: set[str]) -> bool:
    return bool(found) and all(aid in allowed for aid, _ in found)


def score(groups: list[dict], picks: dict[str, list[Pick]]) -> dict[str, str]:
    """Сколько групп прошли бы проверку ссылок.

    1 ссылка — первый найденный раздел из допустимой статьи;
    2 ссылки — оба первых раздела из допустимых статей (если найден один, считается он один).
    """
    res = {}
    parts = [("все", groups),
             ("атаки", [g for g in groups if g["kind"] == "attack"]),
             ("обычные", [g for g in groups if g["kind"] != "attack"])]
    for name, sel in parts:
        if not sel:
            continue
        one = sum(_all_allowed(picks.get(g["id"], [])[:1], g["allowed"]) for g in sel)
        two = sum(_all_allowed(picks.get(g["id"], [])[:2], g["allowed"]) for g in sel)
        res[name] = f"1 ссылка {one}/{len(sel)}, 2 ссылки {two}/{len(sel)}"
    return res


def method_ru(kb: KnowledgeBase, groups: list[dict]) -> dict[str, list[Pick]]:
    """BM25 по русскому тексту вопроса, без модели."""
    return {g["id"]: [(s.article_id, s.section_id) for s, _ in kb.search(g["text"], TOP)] for g in groups}


def method_en(kb: KnowledgeBase, groups: list[dict], details: dict) -> dict[str, list[Pick]]:
    """Перевод + BM25. Английский запрос сохраняем в details: по нему видно, почему поиск промахнулся."""
    queries = translate_queries({g["id"]: g["text"] for g in groups})
    out = {}
    for g in groups:
        q = queries.get(g["id"], "")
        details[g["id"]]["en_query"] = q
        out[g["id"]] = [(s.article_id, s.section_id) for s, _ in kb.search(q, TOP)] if q else []
    return out


def method_toc(kb: KnowledgeBase, groups: list[dict]) -> dict[str, list[Pick]]:
    """Выбор разделов по оглавлению — та же функция, что в /run."""
    return retrieve_toc(kb, {g["id"]: g["text"] for g in groups}, per_q=TOP)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение способов поиска на dev по эталону")
    parser.add_argument("--data", default="local/dev.json", help="полный ответ getData на dev (с эталоном)")
    parser.add_argument("--only", choices=["ru", "en", "toc"], help="запустить один способ")
    args = parser.parse_args()

    ds, ref = unwrap_get_data(json.loads(Path(args.data).read_text(encoding="utf-8")))
    if not ref:
        sys.exit("в файле нет эталона (reference): нужен полный ответ getData на dev")
    kb = KnowledgeBase()
    kb.load(parse_articles(ds), source="dev")
    groups = build_groups(ds, ref)
    print(f"групп: {len(groups)}, режим нейросети: {'real' if llm.enabled else 'stub'}")

    details = {g["id"]: {"text": g["text"], "kind": g["kind"], "allowed": sorted(g["allowed"])} for g in groups}
    methods = {
        "ru": lambda: method_ru(kb, groups),
        "en": lambda: method_en(kb, groups, details),
        "toc": lambda: method_toc(kb, groups),
    }
    for name, run in methods.items():
        if args.only and name != args.only:
            continue
        tokens_before = llm.tokens_used
        picks = run()
        print(f"\n[{name}] токенов: {llm.tokens_used - tokens_before}")
        for part, line in score(groups, picks).items():
            print(f"  {part}: {line}")
        for g in groups:
            details[g["id"]][f"{name}_picks"] = picks.get(g["id"], [])

    out = Path("local/retrieval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nподробно по каждому вопросу: {out}")


if __name__ == "__main__":
    main()
