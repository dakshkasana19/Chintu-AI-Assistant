from __future__ import annotations

from scripts.train_rag_models import _bundle_text, _ranking_metrics


def test_answerability_features_ignore_candidate_order() -> None:
    passages = {
        "p-a": {"title": "Alpha", "text": "Alpha passage."},
        "p-b": {"title": "Beta", "text": "Beta passage."},
    }
    first = {
        "query": "What is alpha?",
        "candidate_labels": [
            {"passage_id": "p-a"},
            {"passage_id": "p-b"},
        ],
    }
    reordered = {
        **first,
        "candidate_labels": list(reversed(first["candidate_labels"])),
    }

    assert _bundle_text(first, passages) == _bundle_text(reordered, passages)


def test_ranking_metrics_reward_correct_top_passage() -> None:
    example = {
        "candidate_labels": [
            {"passage_id": "p-a", "relevance": 3},
            {"passage_id": "p-b", "relevance": 1},
            {"passage_id": "p-c", "relevance": 0},
        ],
    }

    metrics = _ranking_metrics(
        [example],
        lambda _: {"p-a": 0.9, "p-b": 0.4, "p-c": 0.1},
    )

    assert metrics == {
        "queries": 1.0,
        "hit_at_1": 1.0,
        "mrr": 1.0,
        "ndcg_at_3": 1.0,
    }
