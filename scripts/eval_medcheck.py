"""Локальная копия medcheck: проверка медицинских формулировок тем же словарём, промптом и моделью,
что у организаторов (medcheck-resources.json). Тратит наш лимит нейросети (~600 токенов на текст),
а не квоту /case/4/medcheck — она доступна только на dev и быстро кончается.

Два слоя, как у организаторов: словарь регулярных выражений (бесплатно) и модель. Как организаторы
подают текст модели, не опубликовано; гипотеза — промпт в system, проверяемый текст в user.

Запуск из корня репозитория (по умолчанию --resources local/medcheck-resources.json):
  python scripts/eval_medcheck.py --answers local/answers_final.json                # выбор и оценка, 0 токенов
  python scripts/eval_medcheck.py --answers local/answers_final.json --run          # проверить моделью
  python scripts/eval_medcheck.py --texts local/texts_iter1b.txt --lines 1-6 --run  # тексты из файла
Без --kind проверяются ответы dev на обычные вопросы и атаки. Уже проверенные тексты (по хэшу)
берутся из local/eval_medcheck.jsonl и не тратят токены повторно.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402  (импорт после правки sys.path)
from app.contract import unwrap_get_data  # noqa: E402

LOG = Path("local/eval_medcheck.jsonl")
TOKENS_PER_TEXT = 600  # грубая оценка: промпт + текст + скрытое рассуждение


def load_dictionary(res: dict) -> list[tuple[str, re.Pattern]]:
    """Словарь организаторов написан для модуля regex (\\p{L}); в стандартном re то же самое — \\w."""
    rules = []
    for d in res["dictionary"]:
        expr = d["expression"].replace("[\\p{L}\\p{N}_]", "\\w")
        rules.append((d["category"], re.compile(expr, re.IGNORECASE)))
    return rules


def call_model(res: dict, text: str, json_mode: bool = True) -> tuple[dict | None, int]:
    prompt = res["prompt"]
    payload = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "system", "content": prompt["text"]}, {"role": "user", "content": text}],
        "max_tokens": 1200,
        "temperature": prompt.get("temperature", 0),
    }
    if json_mode:  # у организаторов режим не описан; откалиброванная копия работала без него
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(config.LLM_BASE_URL + "/chat/completions", data=body, method="POST",
                                 headers={"Authorization": "Bearer " + config.LLM_API_KEY,
                                          "Content-Type": "application/json"})
    for pause in (5, 15, None):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            used = int((resp.get("usage") or {}).get("total_tokens", 0))
            content = resp["choices"][0]["message"].get("content") or ""
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())  # без режима JSON бывают ```
            try:
                return json.loads(content, strict=False), used
            except json.JSONDecodeError:
                return {"category": "?", "quote": "не JSON: " + content[:120]}, used
        except urllib.error.HTTPError as e:
            print(f"   HTTP {e.code}")
            if e.code in (400, 401, 403) or pause is None:
                return None, 0
        except OSError as e:  # сеть, DNS, обрыв соединения
            print(f"   сеть: {getattr(e, 'reason', e)}")
            if pause is None:
                return None, 0
        time.sleep(pause)
    return None, 0


def parse_lines(spec: str, total: int) -> list[int]:
    if not spec:
        return list(range(1, total + 1))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальная копия medcheck")
    parser.add_argument("--resources", default="local/medcheck-resources.json", help="материалы medcheck организаторов")
    parser.add_argument("--data", default="local/dev_plus.json", help="набор с эталоном dev")
    parser.add_argument("--answers", help="ответы eval_dev.py")
    parser.add_argument("--kind", choices=["normal", "attack", "all"], default="all", help="какие ответы dev")
    parser.add_argument("--texts", help="вместо ответов: файл с текстами по одному на строку")
    parser.add_argument("--lines", default="", help="номера строк файла --texts, например 1-6 или 1,3,5")
    parser.add_argument("--run", action="store_true", help="вызвать модель (без флага — только выбор, 0 токенов)")
    parser.add_argument("--plain", action="store_true", help="без режима JSON — как откалиброванная копия")
    parser.add_argument("--fresh", action="store_true", help="не брать вердикты из журнала")
    args = parser.parse_args()

    res = json.loads(Path(args.resources).read_text(encoding="utf-8"))
    rules = load_dictionary(res)

    items: list[tuple[str, str]] = []  # (метка, текст)
    if args.texts:
        lines = [t.strip() for t in Path(args.texts).read_text(encoding="utf-8").splitlines()]
        items = [(f"строка {n}", lines[n - 1]) for n in parse_lines(args.lines, len(lines)) if lines[n - 1]]
    else:
        _, ref = unwrap_get_data(json.loads(Path(args.data).read_text(encoding="utf-8")))
        kind = {x["record_id"]: x["request_type"] for x in (ref or {}).get("requests", [])}
        for a in json.loads(Path(args.answers).read_text(encoding="utf-8"))["requests"]:
            k = kind.get(a["record_id"])
            if k and (args.kind == "all" or k == args.kind):
                items.append((a["record_id"], a["text"]))

    done: dict[str, dict] = {}
    if LOG.exists() and not args.fresh:
        for line in LOG.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("category") not in (None, "?", "сбой"):
                done[rec["sha"]] = rec
    todo = [(label, t) for label, t in items if hashlib.sha256(t.encode("utf-8")).hexdigest() not in done]
    print(f"medcheck {res['version']}: текстов {len(items)}, уже в журнале {len(items) - len(todo)}, "
          f"проверить {len(todo)} — оценка ~{len(todo) * TOKENS_PER_TEXT // 1000} тыс. токенов "
          "(словарь бесплатно, модель — только если словарь чист)")
    if not args.run:
        print("это холостой запуск: добавь --run, чтобы проверить моделью")
        return

    total, counts = 0, {}
    for label, text in items:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if sha in done:
            v = done[sha]
        else:
            hit = next(((cat, m.group()) for cat, rx in rules if (m := rx.search(text))), None)
            if hit:
                v = {"category": hit[0], "quote": hit[1], "layer": "словарь", "tokens": 0}
            else:
                verdict, used = call_model(res, text, json_mode=not args.plain)
                total += used
                verdict = verdict or {"category": "сбой", "quote": None}
                v = {"category": verdict.get("category", "?"), "quote": verdict.get("quote"),
                     "layer": "модель", "mode": "plain" if args.plain else "json", "tokens": used}
            with LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"label": label, "sha": sha, **v}, ensure_ascii=False) + "\n")
        counts[v["category"]] = counts.get(v["category"], 0) + 1
        if v["category"] != "clean":
            print(f"✗ {label:22s} {v['category']} — «{v.get('quote')}»")
    print(f"\nитого: {counts}; токенов в этом запуске: {total}")


if __name__ == "__main__":
    main()
