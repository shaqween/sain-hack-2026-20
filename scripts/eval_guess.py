"""Локальная копия проверки различимости экспертов (V): identity_strip + expert_guess.

Как у организаторов: из ответа удаляются явные маркеры автора (роль, самопредставление, подпись,
эмодзи), затем проверяющая модель по тексту и трём опубликованным описаниям угадывает эксперта.
Промпты, модель и temperature берутся из dataset.judge_prompts при запуске.

Экономия: если в тексте нет ни одного маркера (названий ролей и эмодзи), очистка ничего бы не
изменила, и вызов identity_strip пропускается. Наши ответы по правилам SYSTEM маркеров не содержат.

Запуск из корня репозитория:
  python scripts/eval_guess.py                  # выборка и оценка расхода, 0 токенов
  python scripts/eval_guess.py --triples 4 --run
Результаты дописываются в local/eval_guess.jsonl — их читает scripts/eval_score.py.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402  (импорт после правки sys.path)
from app.contract import unwrap_get_data  # noqa: E402

LOG = Path("local/eval_guess.jsonl")
TOKENS_PER_GUESS = 1300  # промпт + описания + ответ + скрытое рассуждение: грубая оценка
MARKERS = re.compile(r"клиницист|биохакер|нутрициолог|сомнолог|clinician|biohacker|nutritionist|"
                     r"[\U0001F000-\U0001FAFF☀-➿]", re.IGNORECASE)


def prompt_of(ds: dict, name: str) -> dict:
    return next(p for p in ds["judge_prompts"] if p["prompt_id"] == name)


def call(prompt: dict, payload: dict) -> tuple[dict | None, int]:
    body = json.dumps({
        "model": config.LLM_MODEL,
        "messages": [{"role": "system", "content": prompt["text"]},
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        "max_tokens": 1500,
        "temperature": prompt.get("temperature", 0),
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(config.LLM_BASE_URL + "/chat/completions", data=body, method="POST",
                                 headers={"Authorization": "Bearer " + config.LLM_API_KEY,
                                          "Content-Type": "application/json"})
    for pause in (5, 15, None):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            used = int((resp.get("usage") or {}).get("total_tokens", 0))
            try:
                return json.loads(resp["choices"][0]["message"].get("content") or "", strict=False), used
            except json.JSONDecodeError:
                return None, used
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальная копия identity_strip + expert_guess")
    parser.add_argument("--data", default="local/dev_plus.json", help="набор с эталоном dev")
    parser.add_argument("--answers", default="local/answers_final.json", help="ответы eval_dev.py")
    parser.add_argument("--triples", type=int, default=4, help="сколько троек обычных вопросов проверить")
    parser.add_argument("--run", action="store_true", help="вызвать модель (без флага — только выборка, 0 токенов)")
    args = parser.parse_args()

    ds, ref = unwrap_get_data(json.loads(Path(args.data).read_text(encoding="utf-8")))
    strip_prompt, guess_prompt = prompt_of(ds, "identity_strip"), prompt_of(ds, "expert_guess")
    experts = ds["experts"]  # опубликованные описания: expert_id, role, speech_style
    req = {r["record_id"]: r for r in ds["requests"]}
    normal = {x["record_id"] for x in ref["requests"] if x["request_type"] == "normal"}

    triples: dict[str, list[dict]] = defaultdict(list)
    for a in json.loads(Path(args.answers).read_text(encoding="utf-8"))["requests"]:
        if a["record_id"] in normal:
            r = req[a["record_id"]]
            triples[r.get("group_id") or r["utterance"]["text"]].append(a)
    keys = sorted(triples)
    random.Random(0).shuffle(keys)
    sample = [a for k in keys[: args.triples] for a in triples[k]]
    need_strip = sum(bool(MARKERS.search(a["text"])) for a in sample)
    print(f"выборка: {args.triples} троек, {len(sample)} ответов; очистка нужна в {need_strip}; "
          f"оценка ~{(len(sample) + need_strip) * TOKENS_PER_GUESS // 1000} тыс. токенов")
    if not args.run:
        print("это холостой запуск: добавь --run, чтобы проверить моделью")
        return

    total, right = 0, 0
    for a in sample:
        expected = req[a["record_id"]]["expert_id"]
        text, stripped, rec_tokens = a["text"], False, 0
        if MARKERS.search(text):
            cleaned, used = call(strip_prompt, {"answer": text})
            rec_tokens += used
            if cleaned and isinstance(cleaned.get("text"), str):
                text, stripped = cleaned["text"], True
        verdict, used = call(guess_prompt, {"experts": experts, "answer": text})
        rec_tokens += used
        total += rec_tokens
        guess = (verdict or {}).get("expert_id", "сбой")
        correct = guess == expected
        right += correct
        print(f"{'✓' if correct else '✗'} {a['record_id']:14s} автор {expected:12s} угадано {guess:12s} "
              f"уверенность {(verdict or {}).get('confidence', '?')}")
        if not correct:
            print(f"     {str((verdict or {}).get('reason', ''))[:250]}")
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"record_id": a["record_id"], "expected": expected, "guess": guess,
                                "correct": correct, "stripped": stripped, "verdict": verdict, "tokens": rec_tokens},
                               ensure_ascii=False) + "\n")
    acc = right / len(sample) if sample else 0
    print(f"\nугадано {right} из {len(sample)} (acc ≈ {acc:.2f}, V ≈ {max(0, (acc - 1 / 3) / (2 / 3)):.2f}"
          f" без учёта ссылок); токенов: {total}")


if __name__ == "__main__":
    main()
