from __future__ import annotations

import csv

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


def test_review_draft_covers_only_held_out_topics_and_is_approved() -> None:
    examples = _read_jsonl(MANIFEST_PATH.parent / "expanded_queries.jsonl")
    expected_topics = {
        str(example["topic_group"])
        for example in examples
        if example["split"] == "test"
    }
    draft_path = MANIFEST_PATH.parent / "test_label_review_draft.csv"
    with draft_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    assert {row["topic_id"] for row in rows} == expected_topics
    assert all(
        row["review_status"] == "user_approved_provisional"
        for row in rows
    )
