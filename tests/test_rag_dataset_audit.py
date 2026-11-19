from __future__ import annotations

from scripts.audit_rag_dataset import _build_review_rows
from scripts.build_rag_dataset import MANIFEST_PATH, _read_jsonl


def test_review_rows_include_same_split_distractors_without_relabeling() -> None:
    topics = _read_jsonl(MANIFEST_PATH)
    passages = _read_jsonl(MANIFEST_PATH.parent / "expanded_passages.jsonl")
    examples = _read_jsonl(MANIFEST_PATH.parent / "expanded_queries.jsonl")
    rows = _build_review_rows(topics, passages, examples)
    split_by_topic = {
        str(example["topic_group"]): str(example["split"])
        for example in examples
    }
    topic_by_url = {
        str(passage["source_url"]): str(passage["topic_id"])
        for passage in passages
    }

    assert len(rows) == 100
    for row in rows:
        distractor_url = str(row["bm25_top_distractor_url"])
        if distractor_url:
            distractor_topic = topic_by_url[distractor_url]
            assert split_by_topic[distractor_topic] == row["split"]
        else:
            assert row["bm25_top_distractor_score"] == ""
        assert row["reviewed_source_support"] == ""
        assert row["reviewed_hard_negative_relevance"] == ""
        assert row["reviewed_distractor_relevance"] == ""
        assert row["review_status"] == "needs_human_review"
