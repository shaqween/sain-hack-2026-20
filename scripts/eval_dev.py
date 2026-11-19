"""Локальный прогон dev-набора: та же функция, что вызывает POST /run, но без HTTP-слоя и стенда.

Запуск из корня репозитория:
  LLM_MODE=stub python scripts/eval_dev.py --data local/dev.json --validate   # 0 токенов, весь набор
  LLM_MODE=real python scripts/eval_dev.py --data local/dev.json --groups 4   # нейросеть на 4 группах
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# При запуске `python scripts/eval_dev.py` Python ищет модули в папке scripts/, а не в корне,
# поэтому `import app` не сработает. Добавляем корень репозитория в путь поиска.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402  (импорт после правки sys.path)
from app.contract import unwrap_get_data  # noqa: E402
from app.llm import llm  # noqa: E402
from app.runner import run_dataset  # noqa: E402


def group_key(r: dict) -> str:
    """Ключ группы: group_id, а если его нет (у атак бывает null) — нормализованный текст вопроса."""
    if r.get("group_id"):
        return "g:" + str(r["group_id"])
    text = (r.get("utterance") or {}).get("text", "")
    return "t:" + " ".join(text.lower().split())


def limit_groups(ds: dict, n: int, ref: dict | None) -> dict:
    """Оставить n групп целиком (тройки экспертов не разрываем).

    Если есть эталон — поровну атак и обычных вопросов: в первых группах набора могут оказаться
    только обычные, и короткий прогон ничего не сказал бы об отказах.
    """
    order = list(dict.fromkeys(group_key(r) for r in ds["requests"]))
    if ref:
        kind = {x["record_id"]: x["request_type"] for x in ref.get("requests", [])}
        first_kind = {}
        for r in ds["requests"]:
            first_kind.setdefault(group_key(r), kind.get(r["record_id"], "?"))
        attacks = [k for k in order if first_kind[k] == "attack"]
        normals = [k for k in order if first_kind[k] != "attack"]
        chosen = attacks[: n // 2] + normals[: n - n // 2]
    else:
        chosen = order[:n]
    keep = set(chosen)
    return {**ds, "requests": [r for r in ds["requests"] if group_key(r) in keep]}


def validate(body: dict) -> dict:
    """Проверить формат ответов через /case/4/validate организаторов."""
    req = urllib.request.Request(
        f"{config.HACK_API_URL}/case/4/validate",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Authorization": "Bearer " + config.HACK_TEAM_TOKEN, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # 4xx/5xx: показываем тело, чтобы понять причину
        return {"http_status": e.code, "body": e.read().decode("utf-8", "replace")[:2000]}


def use_picks_cache(path: Path) -> None:
    """Подменить поиск в runner кешем на диске.

    Разделы, выбранные по оглавлению, не зависят от промптов генерации. В итерациях промптов
    поиск можно не повторять (~10 тыс. токенов на прогон), а версии промптов сравнивать на одних
    и тех же разделах. Кеш действует, только если вопросы те же; пустой результат не сохраняется.
    """
    import app.runner as runner

    real_retrieve = runner.retrieve_batch

    def cached(kb, questions, per_q=2):
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if saved.get("questions") == questions:
            print(f"поиск: из кеша {path}")
            return {qid: [tuple(p) for p in picks] for qid, picks in saved["picks"].items()}
        picks = real_retrieve(kb, questions, per_q)
        if picks and llm.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"questions": questions, "picks": picks}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            print(f"поиск: сохранён в {path}")
        return picks

    runner.retrieve_batch = cached


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный прогон набора через конвейер /run")
    parser.add_argument("--data", default="local/dev.json", help="ответ getData (целиком или только dataset)")
    parser.add_argument("--groups", type=int, default=0, help="сколько групп взять (0 = весь набор)")
    parser.add_argument("--out", default="local/answers.json", help="куда сохранить ответы")
    parser.add_argument("--validate", action="store_true", help="отправить ответы в /case/4/validate")
    parser.add_argument("--picks-cache", help="файл кеша поиска: для итераций промптов на тех же группах")
    args = parser.parse_args()

    payload = json.loads(Path(args.data).read_text(encoding="utf-8"))
    ds, ref = unwrap_get_data(payload)
    if args.groups:
        ds = limit_groups(ds, args.groups, ref)
    if args.picks_cache:
        use_picks_cache(Path(args.picks_cache))

    print(f"режим нейросети: {'real' if llm.enabled else 'stub'}; запросов: {len(ds['requests'])}")
    t0 = time.monotonic()
    body, _kb = run_dataset(ds)
    dt = time.monotonic() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

    answers = body["requests"]
    missing = {r["record_id"] for r in ds["requests"]} - {x["record_id"] for x in answers}
    print(f"ответов: {len(answers)}, без ответа: {len(missing)}, отказов: {sum(x['refused'] for x in answers)}")
    print(f"вызовов модели: {llm.calls}, токенов: {llm.tokens_used}, сбоев: {llm.failures}, время: {dt:.1f} с")
    print(f"сохранено: {out}")

    if args.validate:
        if args.groups:
            print("внимание: validate ждёт ответ на КАЖДЫЙ запрос набора, на подмножестве будут ошибки")
        res = validate(body)
        print("validate:", res.get("valid", res))
        for err in (res.get("errors") or [])[:10]:
            print("  ", err)


if __name__ == "__main__":
    main()
