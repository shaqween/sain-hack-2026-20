"""База знаний и поиск (R в RAG).

Поиск — BM25 на чистом Python: без GPU, без моделей эмбеддингов и без сторонних пакетов,
индекс 20 статей строится за миллисекунды. Поэтому новая статья доступна экспертам сразу после
add_article(), а не «через несколько дней».
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter

from .contract import Article, Section

_STOP = set(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было "
    "вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас "
    "нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их "
    "чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой "
    "совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при "
    "наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три "
    "эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно "
    "всю между the of and to in a is for on with that are by as".split()
)
_WORD = re.compile(r"[a-zа-я0-9]+")


def tokenize(text: str) -> list[str]:
    words = _WORD.findall(text.lower().replace("ё", "е"))
    return [w[:6] for w in words if w not in _STOP and len(w) > 1]


class KnowledgeBase:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.articles: dict[str, Article] = {}
        self._docs: list[Section] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._avgdl = 1.0
        self.source = "empty"

    # наполнение
    def load(self, articles: list[Article], source: str) -> None:
        with self._lock:
            self.articles = {a.article_id: a for a in articles}
            self.source = source
            self._reindex()

    def add_article(self, article: Article) -> None:
        with self._lock:
            self.articles[article.article_id] = article
            self._reindex()

    def _reindex(self) -> None:
        self._docs, self._tf, self._df = [], [], Counter()
        for a in self.articles.values():
            for s in a.sections:
                # заголовок статьи и раздела повторяем дважды
                tokens = tokenize(f"{a.title} {s.title} {a.title} {s.title} {s.text}")
                tf = Counter(tokens)
                self._docs.append(s)
                self._tf.append(tf)
                self._df.update(tf.keys())
        n = max(len(self._docs), 1)
        self._avgdl = sum(sum(t.values()) for t in self._tf) / n or 1.0

    def search(self, query: str, k: int = 4) -> list[tuple[Section, float]]:
        q = tokenize(query)
        if not q:
            return []
        n = len(self._docs)
        k1, b = 1.5, 0.75
        scores = []
        for i, tf in enumerate(self._tf):
            dl = sum(tf.values())
            s = 0.0
            for t in set(q):
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / self._avgdl))
            if s > 0:
                scores.append((self._docs[i], s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    # справочное
    def exists(self, article_id: str, section_id: str) -> bool:
        a = self.articles.get(article_id)
        return bool(a and any(s.section_id == section_id for s in a.sections))

    def section(self, article_id: str, section_id: str) -> Section | None:
        a = self.articles.get(article_id)
        if not a:
            return None
        return next((s for s in a.sections if s.section_id == section_id), None)

    def stats(self) -> dict:
        return {"source": self.source, "articles": len(self.articles), "sections": len(self._docs)}
