"""Добавить новую статью в базу ОДНОЙ командой и измерить, через сколько эксперт на неё сошлётся.

    python scripts/add_article.py path/to/article.json                       # в репозиторий (data/extra_articles)
    python scripts/add_article.py article.json --live https://team20.hack.be-healthy.tech  # ещё и в работающий сервер

Формат файла — как у статьи в наборе (см. data/demo_base.json): id, title, license, sections[{id, title, text}].
Скрипт проверяет формат, кладёт файл в data/extra_articles/, перестраивает индекс и сразу задаёт
эксперту вопрос по теме статьи — печатает время до первой ссылки на неё (это число — в README).
"""

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LLM_MODE", "stub")
from app import config  # noqa: E402
from app.agent import answer_group  # noqa: E402
from app.contract import parse_articles  # noqa: E402
from app.kb import KnowledgeBase  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--live", help="адрес работающего бэкенда, чтобы добавить и туда")
ap.add_argument("--question", help="вопрос для проверки; по умолчанию — заголовок первого раздела")
args = ap.parse_args()

with open(args.file, encoding="utf-8") as f:
    raw = json.load(f)
parsed = parse_articles({"articles": [raw]})
if not parsed or not parsed[0].sections or not all(s.text for s in parsed[0].sections):
    sys.exit("ошибка: нужны id, title и непустые sections[{id, title, text}]")
art = parsed[0]
if "BY" not in art.license.upper():
    print("ВНИМАНИЕ: лицензия не CC BY — проверьте, можно ли показывать текст в приложении")

dst_dir = os.path.join(ROOT, "data", "extra_articles")
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(args.file, os.path.join(dst_dir, f"{art.article_id}.json"))

t0 = time.perf_counter()
kb = KnowledgeBase()
with open(os.path.join(ROOT, "data", "demo_base.json"), encoding="utf-8") as f:
    kb.load(parse_articles(json.load(f)), "demo")
kb.add_article(art)
t_index = time.perf_counter() - t0
q = args.question or f"{art.sections[0].title}: что об этом известно?"
g = answer_group(kb, q, ["somnologist"])
t_total = time.perf_counter() - t0
cited = any(a == art.article_id for a, _ in g.replies["somnologist"].citations)
print(f"индекс перестроен за {t_index*1000:.1f} мс; первый ответ {'ССЫЛАЕТСЯ' if cited else 'не ссылается'} "
      f"на новую статью через {t_total*1000:.1f} мс после добавления (вопрос: «{q}»)")

if args.live:
    req = urllib.request.Request(
        args.live.rstrip("/") + "/api/admin/articles", data=json.dumps({"article": raw}).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + config.HACK_TEAM_TOKEN})
    with urllib.request.urlopen(req, timeout=30) as r:
        print("live:", json.load(r))
print("Не забудьте закоммитить data/extra_articles/ — содержимое контейнера пропадает при деплое.")
