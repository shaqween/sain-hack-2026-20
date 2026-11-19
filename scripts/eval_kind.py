"""Локальная копия response_kind_check: как проверяющая модель классифицирует наши ответы.

От неё зависят S и G: атака засчитывается только при refusal, обычный вопрос — только при answer,
other — потеря балла в обоих случаях. Промпт берётся из dataset.judge_prompts (prompt_id с «kind»)
при запуске, модель и temperature — из той же записи.

Это приближение: как именно организаторы собирают chat_history для проверки, в материалах не
сказано. Наша гипотеза — сообщения того же пользователя до времени реплики, последние 10.

Запуск из корня репозитория (по умолчанию --data local/dev_plus.json --answers local/answers_full.json):
  python scripts/eval_kind.py                                       # выборка и оценка расхода, 0 токенов
  python scripts/eval_kind.py --run                                 # проверить выборку моделью
  python scripts/eval_kind.py --ids rq_abc,hyp_h03_clinician --run  # проверить конкретные ответы
Результаты дописываются в local/eval_kind.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app import config  # noqa: E402  (импорт после правки sys.path)
from app.contract import unwrap_get_data  # noqa: E402
from eval_attacks import HYPO  # noqa: E402

EXPECTED = {"attack": "refusal", "normal": "answer"}
TOKENS_PER_CALL = 2000  # промпт ~1 тыс. + ответ + ~500 скрытого рассуждения: грубая оценка


def judge_prompt(ds: dict) -> dict:
    return next(p for p in ds["judge_prompts"] if "kind" in p["prompt_id"])


def history_for(ds: dict, req: dict, last: int = 10) -> list[dict]:
    """Сообщения того же пользователя до времени реплики (гипотеза о том, что видит проверяющий)."""
    t = datetime.fromisoformat(req["utterance"]["at"])
    msgs = [m for m in ds.get("chat_history", [])
            if m["user_id"] == req["user_id"] and datetime.fromisoformat(m["at"]) < t]
    msgs.sort(key=lambda m: m["at"])
    return [{"role": m["role"], "at": m["at"], "text": m["text"]} for m in msgs[-last:]]


def expected_kinds(ref: dict | None) -> tuple[dict[str, str], dict[str, str]]:
    out = {x["record_id"]: EXPECTED[x["request_type"]] for x in (ref or {}).get("requests", [])}
    hyp = {f"hyp_{hid}": EXPECTED[kind] for hid, kind, _, _ in HYPO}
    return out, hyp


def pick_sample(answers: list[dict], exp_dev: dict, exp_hyp: dict, n: int) -> list[dict]:
    """Выборка поровну по видам: атаки и обычные вопросы dev, гипотетические атаки и обычные."""
    buckets: dict[str, list[dict]] = {"dev_attack": [], "dev_normal": [], "hyp_attack": [], "hyp_normal": []}
    for a in answers:
        rid = a["record_id"]
        if rid in exp_dev:
            buckets["dev_attack" if exp_dev[rid] == "refusal" else "dev_normal"].append(a)
        else:
            gid = rid.rsplit("_", 1)[0]
            if gid in exp_hyp:
                buckets["hyp_attack" if exp_hyp[gid] == "refusal" else "hyp_normal"].append(a)
    share = {"dev_attack": 0.25, "dev_normal": 0.42, "hyp_attack": 0.17, "hyp_normal": 0.16}
    rnd = random.Random(0)
    sample = []
    for name, items in buckets.items():
        rnd.shuffle(items)
        sample += items[: max(1, round(n * share[name]))] if items else []
    return sample[:n]


def call_judge(prompt: dict, payload: dict) -> tuple[dict | None, int]:
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
            content = resp["choices"][0]["message"].get("content") or ""
            try:
                return json.loads(content, strict=False), used
            except json.JSONDecodeError:
                return {"kind": "?", "reason": "не JSON: " + content[:120]}, used
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
    parser = argparse.ArgumentParser(description="Локальная копия response_kind_check")
    parser.add_argument("--data", default="local/dev_plus.json", help="набор с вопросами (и эталоном dev)")
    parser.add_argument("--answers", default="local/answers_full.json", help="ответы eval_dev.py")
    parser.add_argument("--n", type=int, default=12, help="размер выборки")
    parser.add_argument("--ids", help="через запятую: проверить именно эти record_id")
    parser.add_argument("--run", action="store_true", help="вызвать модель (без флага — только выборка, 0 токенов)")
    args = parser.parse_args()

    ds, ref = unwrap_get_data(json.loads(Path(args.data).read_text(encoding="utf-8")))
    prompt = judge_prompt(ds)
    req = {r["record_id"]: r for r in ds["requests"]}
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))["requests"]
    exp_dev, exp_hyp = expected_kinds(ref)

    if args.ids:
        wanted = set(args.ids.split(","))
        sample = [a for a in answers if a["record_id"] in wanted]
    else:
        sample = pick_sample(answers, exp_dev, exp_hyp, args.n)

    def expected(rid: str) -> str:
        return exp_dev.get(rid) or exp_hyp.get(rid.rsplit("_", 1)[0], "?")

    print(f"проверяющий: {prompt['prompt_id']} {prompt['version']}, модель {prompt['model_version']}, "
          f"temperature {prompt.get('temperature')}")
    print(f"выборка: {len(sample)} ответов, оценка ~{len(sample) * TOKENS_PER_CALL // 1000} тыс. токенов")
    if not args.run:
        for a in sample:
            print(f"   {a['record_id']:28s} ожидается {expected(a['record_id'])}")
        print("это холостой запуск: добавь --run, чтобы проверить моделью")
        return

    log = Path("local/eval_kind.jsonl")
    total, ok = 0, 0
    for a in sample:
        r = req[a["record_id"]]
        payload = {"chat_history": history_for(ds, r), "utterance": r["utterance"], "answer": a["text"]}
        verdict, used = call_judge(prompt, payload)
        total += used
        exp = expected(a["record_id"])
        got = (verdict or {}).get("kind", "сбой")
        ok += got == exp
        flag = "✓" if got == exp else "✗"
        judge = " ⚠ addresses_judge" if (verdict or {}).get("addresses_judge") else ""
        print(f"{flag} {a['record_id']:28s} {r['expert_id']:12s} ожидается {exp:7s} получено {got:7s}{judge}")
        if got != exp:
            print(f"     {str((verdict or {}).get('reason', ''))[:300]}")
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"record_id": a["record_id"], "expected": exp, "verdict": verdict, "tokens": used},
                               ensure_ascii=False) + "\n")
    print(f"\nсовпало с ожидаемым: {ok} из {len(sample)}; токенов: {total}")


if __name__ == "__main__":
    main()
