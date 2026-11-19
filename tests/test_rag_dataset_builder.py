from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_rag_dataset import (
    MANIFEST_PATH,
    _build_examples,
    _read_jsonl,
    _write_question_catalog,
)


def test_candidate_passages_stay_within_their_query_split() -> None:
    topics = _read_jsonl(MANIFEST_PATH)
    examples = _build_examples(topics)
    topic_splits = {
        str(example["topic_group"]): str(example["split"])
        for example in examples
    }

    for example in examples:
        for candidate in example["candidate_labels"]:
            topic_id = str(candidate["passage_id"])[2:]
            assert topic_splits[topic_id] == example["split"]


def test_question_catalog_has_one_traceable_row_per_topic(tmp_path: Path) -> None:
    topics = _read_jsonl(MANIFEST_PATH)
    examples = _build_examples(topics)
    passages = {
        str(passage["topic_id"]): passage
        for passage in _read_jsonl(MANIFEST_PATH.parent / "expanded_passages.jsonl")
    }
    catalog_path = tmp_path / "questions.csv"

    _write_question_catalog(catalog_path, topics, passages, examples)

    with catalog_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 100
    assert len({row["topic_id"] for row in rows}) == 100
    assert rows[0]["question"] == topics[0]["query"]
    revision_id = str(passages[str(topics[0]["topic_id"])]["revision_id"])
    assert rows[0]["source_url"].endswith(revision_id)
