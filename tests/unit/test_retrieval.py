"""Step-3 tests: policy corpus + deterministic config-driven retrieval."""

import pytest

from telecom_aut.environment import TelecomEnv
from telecom_aut.retrieval import RetrievalConfig, Retriever
from telecom_aut.tools import TelecomAPI, invoke


@pytest.fixture()
def env():
    e = TelecomEnv.fresh()
    yield e
    e.close()


def test_corpus_is_seeded_and_deterministic(env):
    docs = env.snapshot()["tables"]["policy_documents"]
    assert len(docs) == 12
    assert [d["id"] for d in docs] == list(range(600, 612))


class TestBM25:
    def test_finds_the_load_bearing_cancellation_rules(self, env):
        r = Retriever(env)
        hits = r.search("cancel my order unpaid balance invoice")
        slugs = [h["slug"] for h in hits]
        assert slugs[0] == "cancellation-unpaid-balance"
        assert len(hits) == 4  # default k

    def test_finds_allowed_plans_rule(self, env):
        hits = Retriever(env).search(
            "which plans are allowed for a service update"
        )
        assert hits[0]["slug"] == "plan-allowed-list"

    def test_scores_are_descending_and_rounded(self, env):
        hits = Retriever(env).search("cancellation confirmation termination")
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(round(s, 4) == s for s in scores)

    def test_no_match_returns_empty(self, env):
        assert Retriever(env).search("zebra quantum snowboard") == []

    def test_determinism(self, env):
        r = Retriever(env)
        q = "terminate subscription confirmation appointments"
        assert r.search(q) == r.search(q)


class TestRetrievalConfigSurface:
    """Defect surface #4: behaviour must change from config alone."""

    def test_k_is_config_driven(self, env):
        query = "order plan appointment cancellation customer"
        assert len(Retriever(env, RetrievalConfig(k=1)).search(query)) == 1
        assert len(Retriever(env, RetrievalConfig(k=5)).search(query)) == 5

    def test_category_filter_hides_the_cancellation_rules(self, env):
        # The canonical injected retrieval defect: same query, corpus
        # restricted to billing docs -> the balance rule can no longer surface.
        clean = Retriever(env)
        faulty = Retriever(env, RetrievalConfig(category_filter=["billing"]))
        query = "cancel order unpaid balance"
        assert clean.search(query)[0]["slug"] == "cancellation-unpaid-balance"
        assert all(h["category"] == "billing" for h in faulty.search(query))
        assert "cancellation-unpaid-balance" not in [
            h["slug"] for h in faulty.search(query)
        ]

    def test_config_rejects_unknown_fields(self):
        with pytest.raises(Exception):
            RetrievalConfig(krank=3)


def test_search_policy_tool_reads_without_events(env):
    api = TelecomAPI(env)
    before = env.snapshot()
    hits = invoke(api, "search_policy",
                  {"query": "confirmation before cancelling an order"})
    assert hits and hits[0]["category"] == "cancellation"
    assert env.snapshot() == before  # read: no event, no state change
