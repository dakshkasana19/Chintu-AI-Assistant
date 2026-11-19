from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "rag"
MANIFEST_PATH = DATA_DIR / "topic_manifest.jsonl"
API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "Chintu-AI-Assistant/1.0 (educational dataset; Wikipedia API)"
SEED = 20261119


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_question_catalog(
    path: Path,
    topics: list[dict[str, object]],
    passages: dict[str, dict[str, object]],
    examples: list[dict[str, object]],
) -> None:
    split_by_topic = {
        str(example["topic_group"]): str(example["split"])
        for example in examples
    }
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "topic_id",
                "category",
                "question",
                "source_title",
                "source_url",
                "split",
            ],
        )
        writer.writeheader()
        for topic in topics:
            topic_id = str(topic["topic_id"])
            passage = passages[topic_id]
            writer.writerow(
                {
                    "topic_id": topic_id,
                    "category": topic["category"],
                    "question": topic["query"],
                    "source_title": passage["title"],
                    "source_url": passage["source_url"],
                    "split": split_by_topic[topic_id],
                }
            )


def _load_passages(topics: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    passages: dict[str, dict[str, object]] = {}

    batch_size = 20
    for start in range(0, len(topics), batch_size):
        batch = topics[start : start + batch_size]
        response = session.get(
            API_URL,
            params={
                "action": "query",
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "exsentences": 3,
                "exlimit": "max",
                "inprop": "url",
                "redirects": 1,
                "format": "json",
                "formatversion": 2,
                "titles": "|".join(str(topic["title"]) for topic in batch),
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"Wikipedia API error: {payload['error'].get('info', 'unknown error')}")

        query_data = payload["query"]
        pages = query_data["pages"]
        by_title = {str(page["title"]).casefold(): page for page in pages}
        aliases = {
            str(item["from"]).casefold(): str(item["to"])
            for item in query_data.get("normalized", []) + query_data.get("redirects", [])
        }

        for topic in batch:
            requested_title = str(topic["title"])
            resolved_title = aliases.get(requested_title.casefold(), requested_title)
            page = by_title.get(resolved_title.casefold())
            if page is None or "missing" in page:
                raise RuntimeError(f"No Wikipedia page found for {requested_title}")
            extract = str(page.get("extract", "")).strip()
            revision_id = page.get("lastrevid")
            if len(extract) < 40 or not revision_id:
                raise RuntimeError(f"Wikipedia returned an unusable summary for {requested_title}")

            title = str(page["title"])
            page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            topic_id = str(topic["topic_id"])
            passages[topic_id] = {
                "passage_id": f"p-{topic_id}",
                "topic_id": topic_id,
                "title": title,
                "source_url": f"{page_url}?oldid={revision_id}",
                "revision_id": str(revision_id),
                "source": "Wikipedia",
                "license": "CC BY-SA 4.0",
                "attribution": "Wikipedia contributors",
                "retrieved_at": date.today().isoformat(),
                "text": extract,
            }

        if start + batch_size < len(topics):
            time.sleep(1)

    page_ids = [str(passage["revision_id"]) for passage in passages.values()]
    if len(passages) != len(topics) or len(page_ids) != len(set(page_ids)):
        raise ValueError("Topic manifest contains duplicate or unresolved Wikipedia pages")
    return passages


def _topic_splits(topics: list[dict[str, object]]) -> dict[str, str]:
    randomizer = random.Random(SEED)
    categories: dict[str, list[str]] = defaultdict(list)
    for topic in topics:
        categories[str(topic["category"])].append(str(topic["topic_id"]))

    splits: dict[str, str] = {}
    for category_topics in categories.values():
        randomizer.shuffle(category_topics)
        for index, topic_id in enumerate(category_topics):
            splits[topic_id] = "train" if index < 14 else "validation" if index < 17 else "test"
    return splits


def _build_examples(
    topics: list[dict[str, object]],
) -> list[dict[str, object]]:
    topic_by_id = {str(topic["topic_id"]): topic for topic in topics}
    if len(topic_by_id) != len(topics):
        raise ValueError("Topic manifest contains duplicate topic IDs")

    related_pairs: set[frozenset[str]] = set()
    for topic in topics:
        hard_negative_id = topic.get("hard_negative_topic_id")
        if hard_negative_id:
            if str(hard_negative_id) not in topic_by_id:
                raise ValueError(f"Unknown hard-negative topic ID: {hard_negative_id}")
            related_pairs.add(frozenset((str(topic["topic_id"]), str(hard_negative_id))))

    splits = _topic_splits(topics)
    randomizer = random.Random(SEED)
    records: list[dict[str, object]] = []
    for topic in topics:
        topic_id = str(topic["topic_id"])
        category = str(topic["category"])
        split = splits[topic_id]
        hard_negative_id = (
            str(topic["hard_negative_topic_id"])
            if topic.get("hard_negative_topic_id")
            else None
        )
        if hard_negative_id and splits[hard_negative_id] != split:
            hard_negative_id = None
        unrelated_ids = [
            str(candidate["topic_id"])
            for candidate in topics
            if candidate["topic_id"] != topic_id
            and candidate["category"] != category
            and splits[str(candidate["topic_id"])] == split
            and (hard_negative_id is None or candidate["topic_id"] != hard_negative_id)
            and frozenset((topic_id, str(candidate["topic_id"]))) not in related_pairs
        ]
        if len(unrelated_ids) < 3:
            raise ValueError(f"Not enough unrelated candidates for topic {topic_id}")

        if hard_negative_id:
            answerable_negatives = [hard_negative_id, *randomizer.sample(unrelated_ids, 1)]
            insufficient_negatives = [
                hard_negative_id,
                *randomizer.sample([item for item in unrelated_ids if item not in answerable_negatives], 2),
            ]
        else:
            answerable_negatives = randomizer.sample(unrelated_ids, 2)
            insufficient_negatives = randomizer.sample(unrelated_ids, 3)

        def candidate_label(candidate_id: str) -> dict[str, object]:
            is_hard_negative = (
                hard_negative_id is not None and candidate_id == hard_negative_id
            )
            return {
                "passage_id": f"p-{candidate_id}",
                "relevance": 1 if is_hard_negative else 0,
                "origin": "curated_hard_negative" if is_hard_negative else "curated_random_negative",
            }

        positive = {
            "passage_id": f"p-{topic_id}",
            "relevance": 3,
            "origin": "matched_source",
        }
        records.extend(
            [
                {
                    "example_id": f"{topic_id}-supported",
                    "topic_group": topic_id,
                    "category": category,
                    "split": split,
                    "query": topic["query"],
                    "evidence_condition": "matched_source_included",
                    "candidate_labels": [
                        positive,
                        *(candidate_label(candidate_id) for candidate_id in answerable_negatives),
                    ],
                    "answerability": "answerable",
                    "supporting_passage_ids": [f"p-{topic_id}"],
                    "annotation_status": "assistant_draft",
                },
                {
                    "example_id": f"{topic_id}-missing",
                    "topic_group": topic_id,
                    "category": category,
                    "split": split,
                    "query": topic["query"],
                    "evidence_condition": "matched_source_omitted",
                    "candidate_labels": [
                        candidate_label(candidate_id)
                        for candidate_id in insufficient_negatives
                    ],
                    "answerability": "insufficient",
                    "supporting_passage_ids": [],
                    "annotation_status": "assistant_draft",
                },
            ]
        )
    return records


def main() -> None:
    topics = _read_jsonl(MANIFEST_PATH)
    if len(topics) != 100:
        raise ValueError(f"Expected 100 topics in {MANIFEST_PATH}, found {len(topics)}")

    passages = _load_passages(topics)
    examples = _build_examples(topics)
    _write_jsonl(
        DATA_DIR / "expanded_passages.jsonl",
        [passages[str(topic["topic_id"])] for topic in topics],
    )
    _write_jsonl(DATA_DIR / "expanded_queries.jsonl", examples)
    _write_question_catalog(
        DATA_DIR / "questions.csv",
        topics,
        passages,
        examples,
    )

    split_counts = {
        split: sum(record["split"] == split for record in examples)
        for split in ("train", "validation", "test")
    }
    print(
        f"Built {len(passages)} passages and {len(examples)} evidence bundles; "
        f"splits={split_counts}"
    )


if __name__ == "__main__":
    main()
