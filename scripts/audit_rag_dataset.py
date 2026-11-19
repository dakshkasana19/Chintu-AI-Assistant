from __future__ import annotations

import csv
import json
from pathlib import Path

from chintu.rag.reranker import BM25Reranker, tokenize
from chintu.rag.sources import SearchResult

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "rag"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Expected JSON objects in {path}")
    return records


def _coverage(query: str, text: str) -> tuple[float, list[str]]:
    query_tokens = set(tokenize(query))
    overlap = sorted(query_tokens.intersection(tokenize(text)))
    return len(overlap) / max(1, len(query_tokens)), overlap


def _clean_excerpt(text: object) -> str:
    return " ".join(str(text).split())


def _build_review_rows(
    topics: list[dict[str, object]],
    passages: list[dict[str, object]],
    examples: list[dict[str, object]],
) -> list[dict[str, object]]:
    passages_by_topic = {str(passage["topic_id"]): passage for passage in passages}
    split_by_topic: dict[str, str] = {}
    matched_examples: dict[str, dict[str, object]] = {}
    for example in examples:
        topic_id = str(example["topic_group"])
        split = str(example["split"])
        previous_split = split_by_topic.setdefault(topic_id, split)
        if previous_split != split:
            raise ValueError(f"Topic {topic_id} appears in multiple splits")
        if example["evidence_condition"] == "matched_source_included":
            matched_examples[topic_id] = example

    if set(matched_examples) != set(passages_by_topic):
        raise ValueError("Expected one matched-source example per passage topic")

    reranker = BM25Reranker()
    rows: list[dict[str, object]] = []
    for topic in topics:
        topic_id = str(topic["topic_id"])
        query = str(topic["query"])
        split = split_by_topic[topic_id]
        source = passages_by_topic[topic_id]
        candidate_pool = [
            passage
            for candidate_topic, passage in passages_by_topic.items()
            if split_by_topic[candidate_topic] == split
        ]
        search_results = [
            SearchResult(
                title=str(passage["title"]),
                extract=str(passage["text"]),
                url=str(passage["source_url"]),
                source="wikipedia",
            )
            for passage in candidate_pool
        ]
        ranked = reranker.rank(query, search_results)
        score_by_url: dict[str, float] = {}
        for item in ranked:
            url = item.source_result.url
            score_by_url[url] = max(score_by_url.get(url, 0.0), item.score)
        ordered_pool = sorted(
            candidate_pool,
            key=lambda passage: (
                -score_by_url.get(str(passage["source_url"]), 0.0),
                str(passage["title"]),
            ),
        )
        positive_score = score_by_url.get(str(source["source_url"]), 0.0)
        positive_rank = (
            next(
                index + 1
                for index, passage in enumerate(ordered_pool)
                if str(passage["topic_id"]) == topic_id
            )
            if positive_score > 0
            else None
        )
        distractor = next(
            (
                passage
                for passage in ordered_pool
                if str(passage["topic_id"]) != topic_id
                and score_by_url.get(str(passage["source_url"]), 0.0) > 0
            ),
            None,
        )

        hard_negative_id = str(topic.get("hard_negative_topic_id", ""))
        hard_negative = passages_by_topic.get(hard_negative_id)
        hard_negative_used = bool(
            hard_negative
            and split_by_topic.get(hard_negative_id) == split
        )
        source_coverage, source_overlap = _coverage(query, str(source["text"]))
        hard_coverage, hard_overlap = (
            _coverage(query, str(hard_negative["text"]))
            if hard_negative
            else (0.0, [])
        )
        distractor_coverage, distractor_overlap = (
            _coverage(query, str(distractor["text"]))
            if distractor
            else (0.0, [])
        )

        review_notes: list[str] = []
        if source_coverage < 0.5:
            review_notes.append("matched source has low literal query-term coverage")
        if hard_negative_used and not hard_overlap:
            review_notes.append("used hard negative has no literal query-term overlap")
        if positive_rank is not None and positive_rank > 1:
            review_notes.append("BM25 ranks a different page above the matched source")
        if positive_rank is None:
            review_notes.append("matched source has no literal BM25 score in this pool")
        if distractor is None:
            review_notes.append("no non-source page has a positive BM25 score")

        rows.append(
            {
                "topic_id": topic_id,
                "category": str(topic["category"]),
                "split": split,
                "question": query,
                "source_title": str(source["title"]),
                "source_url": str(source["source_url"]),
                "source_query_term_coverage": f"{source_coverage:.3f}",
                "source_overlap_terms": "|".join(source_overlap),
                "source_excerpt": _clean_excerpt(source["text"]),
                "reviewed_source_support": "",
                "curated_hard_negative_id": hard_negative_id,
                "curated_hard_negative_title": (
                    str(hard_negative["title"]) if hard_negative else ""
                ),
                "curated_hard_negative_url": (
                    str(hard_negative["source_url"]) if hard_negative else ""
                ),
                "curated_hard_negative_used_in_split": hard_negative_used,
                "curated_hard_negative_query_term_coverage": (
                    f"{hard_coverage:.3f}" if hard_negative else ""
                ),
                "curated_hard_negative_overlap_terms": "|".join(hard_overlap),
                "curated_hard_negative_excerpt": (
                    _clean_excerpt(hard_negative["text"]) if hard_negative else ""
                ),
                "curated_hard_negative_draft_relevance": (
                    1 if hard_negative_used else ""
                ),
                "reviewed_hard_negative_relevance": "",
                "bm25_top_distractor_title": (
                    str(distractor["title"]) if distractor else ""
                ),
                "bm25_top_distractor_url": (
                    str(distractor["source_url"]) if distractor else ""
                ),
                "bm25_top_distractor_score": (
                    f"{score_by_url.get(str(distractor['source_url']), 0.0):.6f}"
                    if distractor
                    else ""
                ),
                "bm25_positive_rank_in_split_pool": (
                    positive_rank if positive_rank is not None else ""
                ),
                "bm25_distractor_query_term_coverage": (
                    f"{distractor_coverage:.3f}" if distractor else ""
                ),
                "bm25_distractor_overlap_terms": "|".join(distractor_overlap),
                "bm25_distractor_excerpt": (
                    _clean_excerpt(distractor["text"]) if distractor else ""
                ),
                "reviewed_distractor_relevance": "",
                "review_notes": "; ".join(review_notes),
                "review_status": "needs_human_review",
            }
        )
    return rows


def main() -> None:
    topics = _read_jsonl(DATA_DIR / "topic_manifest.jsonl")
    passages = _read_jsonl(DATA_DIR / "expanded_passages.jsonl")
    examples = _read_jsonl(DATA_DIR / "expanded_queries.jsonl")
    rows = _build_review_rows(topics, passages, examples)

    output_path = DATA_DIR / "annotation_review.csv"
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    notes = [str(row["review_notes"]) for row in rows]
    print(
        f"Wrote {len(rows)} review rows; "
        f"source-match flags={sum('low literal' in note for note in notes)}, "
        f"hard-negative overlap flags={sum('hard negative has no' in note for note in notes)}, "
        f"BM25 source-rank flags={sum('BM25 ranks' in note for note in notes)}"
    )
    print(f"Review sheet: {output_path}")


if __name__ == "__main__":
    main()
