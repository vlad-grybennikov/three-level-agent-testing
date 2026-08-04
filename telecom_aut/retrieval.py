"""Config-driven BM25 retrieval over policy_documents.

Defect surface #4: every behavioural knob lives in
RetrievalConfig (k, BM25 parameters, and the category filter). Injecting a
retrieval defect means constructing the Retriever with a perturbed config
(e.g. k=1, or category_filter=["billing"] which hides the cancellation rules).

Pure Python and fully deterministic: fixed tokeniser, fixed scoring, ties
broken by document id. No embedding service, no network.
"""

from __future__ import annotations

import math
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .environment import TelecomEnv

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Baseline k=4: measured on the 2026-07-28 run. Natural cancel queries
    # ranked a distractor above the unpaid-balance doc at k=3.
    k: int = Field(default=4, ge=1)
    k1: float = Field(default=1.5, ge=0.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    category_filter: Optional[list[str]] = None  # None = whole corpus


class Retriever:
    """BM25 over title+body of policy_documents, read live from the env DB."""

    def __init__(self, env: TelecomEnv, config: RetrievalConfig | None = None):
        self.env = env
        self.config = config or RetrievalConfig()

    def _corpus(self) -> list[dict]:
        rows = [
            dict(r) for r in self.env.conn.execute(
                "SELECT * FROM policy_documents ORDER BY id"
            )
        ]
        if self.config.category_filter is not None:
            allowed = set(self.config.category_filter)
            rows = [r for r in rows if r["category"] in allowed]
        return rows

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Top-k documents by BM25, deterministic (ties broken by id)."""
        cfg = self.config
        k = cfg.k if k is None else k
        docs = self._corpus()
        if not docs:
            return []

        doc_tokens = [tokenize(d["title"] + " " + d["body"]) for d in docs]
        n = len(docs)
        avgdl = sum(len(t) for t in doc_tokens) / n
        df: dict[str, int] = {}
        for tokens in doc_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        query_terms = tokenize(query)
        scored: list[tuple[float, int, dict]] = []
        for doc, tokens in zip(docs, doc_tokens):
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
                num = tf[term] * (cfg.k1 + 1.0)
                den = tf[term] + cfg.k1 * (1.0 - cfg.b + cfg.b * len(tokens) / avgdl)
                score += idf * num / den
            if score > 0.0:
                scored.append((score, doc["id"], doc))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "id": doc["id"],
                "slug": doc["slug"],
                "title": doc["title"],
                "category": doc["category"],
                "score": round(score, 4),
                "body": doc["body"],
            }
            for score, _, doc in scored[:k]
        ]
