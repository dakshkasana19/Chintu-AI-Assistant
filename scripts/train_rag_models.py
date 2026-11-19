from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from chintu.rag.reranker import BM25Reranker
from chintu.rag.sources import SearchResult
from chintu.rag.verifier import GroundingVerifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "rag"
MODEL_DIR = ROOT / "models" / "rag"
SEED = 20261119


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path}")
            records.append(cast(dict[str, object], record))
    return records


def _candidate_records(example: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw_candidates = example.get("candidate_labels")
    if not isinstance(raw_candidates, list):
        raise ValueError("Evidence bundle is missing candidate_labels")
    if not all(isinstance(candidate, Mapping) for candidate in raw_candidates):
        raise ValueError("Evidence bundle contains an invalid candidate label")
    return cast(list[Mapping[str, object]], raw_candidates)


def _candidate_ids(example: Mapping[str, object]) -> list[str]:
    return [str(candidate["passage_id"]) for candidate in _candidate_records(example)]


def _pair_text(
    query: str,
    passage: Mapping[str, object],
) -> str:
    return (
        f"query: {query}\n"
        f"title: {passage['title']}\n"
        f"passage: {passage['text']}"
    )


def _bundle_text(
    example: Mapping[str, object],
    passages: Mapping[str, Mapping[str, object]],
) -> str:
    evidence = [
        _pair_text(str(example["query"]), passages[passage_id])
        for passage_id in sorted(_candidate_ids(example))
    ]
    return f"question: {example['query']}\nevidence:\n" + "\n".join(evidence)


def _make_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _positive_probability(model: Pipeline, texts: list[str]) -> list[float]:
    classifier = model.named_steps["classifier"]
    positive_index = list(classifier.classes_).index(1)
    return [float(score) for score in model.predict_proba(texts)[:, positive_index]]


def _bm25_ranking(
    example: Mapping[str, object],
    passages: Mapping[str, Mapping[str, object]],
    reranker: BM25Reranker,
) -> tuple[dict[str, float], list[object]]:
    passage_ids = _candidate_ids(example)
    source_results = [
        SearchResult(
            title=str(passages[passage_id]["title"]),
            extract=str(passages[passage_id]["text"]),
            url=str(passages[passage_id]["source_url"]),
            source="wikipedia",
        )
        for passage_id in passage_ids
    ]
    ranked = reranker.rank(str(example["query"]), source_results)
    score_by_url: dict[str, float] = {}
    for item in ranked:
        url = item.source_result.url
        score_by_url[url] = max(score_by_url.get(url, 0.0), item.score)
    scores = {
        passage_id: score_by_url.get(str(passages[passage_id]["source_url"]), 0.0)
        for passage_id in passage_ids
    }
    return scores, ranked


def _ranking_metrics(
    examples: list[dict[str, object]],
    score_example: Callable[[Mapping[str, object]], Mapping[str, float]],
) -> dict[str, float]:
    if not examples:
        raise ValueError("Cannot evaluate ranking with an empty split")

    hit_at_one: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    for example in examples:
        labels = {
            str(candidate["passage_id"]): int(candidate["relevance"])
            for candidate in _candidate_records(example)
        }
        scores = score_example(example)
        ordered_ids = sorted(
            labels,
            key=lambda passage_id: (-scores.get(passage_id, 0.0), passage_id),
        )
        ordered_relevance = [labels[passage_id] for passage_id in ordered_ids]
        relevant_ranks = [
            index + 1
            for index, relevance in enumerate(ordered_relevance)
            if relevance == 3
        ]
        hit_at_one.append(float(bool(relevant_ranks) and relevant_ranks[0] == 1))
        reciprocal_ranks.append(1.0 / relevant_ranks[0] if relevant_ranks else 0.0)

        def dcg(relevance: list[int]) -> float:
            return sum(
                (2**value - 1) / math.log2(index + 2)
                for index, value in enumerate(relevance)
            )

        ideal_dcg = dcg(sorted(ordered_relevance, reverse=True))
        ndcg_scores.append(dcg(ordered_relevance) / ideal_dcg if ideal_dcg else 0.0)

    count = len(examples)
    return {
        "queries": float(count),
        "hit_at_1": sum(hit_at_one) / count,
        "mrr": sum(reciprocal_ranks) / count,
        "ndcg_at_3": sum(ndcg_scores) / count,
    }


def _classification_metrics(
    examples: list[dict[str, object]],
    predictions: list[int],
) -> dict[str, object]:
    expected = [int(example["answerability"] == "answerable") for example in examples]
    return {
        "bundles": len(examples),
        "accuracy": float(accuracy_score(expected, predictions)),
        "precision_answerable": float(precision_score(expected, predictions, zero_division=0)),
        "recall_answerable": float(recall_score(expected, predictions, zero_division=0)),
        "f1_answerable": float(f1_score(expected, predictions, zero_division=0)),
        "confusion_matrix": {
            "true_insufficient": sum(y == 0 and p == 0 for y, p in zip(expected, predictions)),
            "false_answerable": sum(y == 0 and p == 1 for y, p in zip(expected, predictions)),
            "false_insufficient": sum(y == 1 and p == 0 for y, p in zip(expected, predictions)),
            "true_answerable": sum(y == 1 and p == 1 for y, p in zip(expected, predictions)),
        },
    }


def _model_ranking_scores(
    model: Pipeline,
    example: Mapping[str, object],
    passages: Mapping[str, Mapping[str, object]],
) -> dict[str, float]:
    passage_ids = _candidate_ids(example)
    texts = [
        _pair_text(str(example["query"]), passages[passage_id])
        for passage_id in passage_ids
    ]
    return dict(zip(passage_ids, _positive_probability(model, texts)))


def _baseline_answerability(
    examples: list[dict[str, object]],
    passages: Mapping[str, Mapping[str, object]],
    reranker: BM25Reranker,
    verifier: GroundingVerifier,
) -> list[int]:
    predictions: list[int] = []
    for example in examples:
        _, ranked = _bm25_ranking(example, passages, reranker)
        result = verifier.verify_and_format(str(example["query"]), ranked)
        predictions.append(int(result.verified))
    return predictions


def main() -> None:
    passages = _read_jsonl(DATA_DIR / "expanded_passages.jsonl")
    examples = _read_jsonl(DATA_DIR / "expanded_queries.jsonl")
    passages_by_id = {
        str(passage["passage_id"]): passage
        for passage in passages
    }
    for example in examples:
        unknown_ids = set(_candidate_ids(example)) - passages_by_id.keys()
        if unknown_ids:
            raise ValueError(f"Unknown candidate passage IDs: {sorted(unknown_ids)}")

    train_examples = [example for example in examples if example["split"] == "train"]
    validation_examples = [
        example for example in examples if example["split"] == "validation"
    ]
    test_examples = [example for example in examples if example["split"] == "test"]
    ranker_train = [
        example
        for example in train_examples
        if example["evidence_condition"] == "matched_source_included"
    ]
    ranker_validation = [
        example
        for example in validation_examples
        if example["evidence_condition"] == "matched_source_included"
    ]
    ranker_test = [
        example
        for example in test_examples
        if example["evidence_condition"] == "matched_source_included"
    ]
    if not ranker_train or not validation_examples or not test_examples:
        raise ValueError("Dataset must include train, validation, and test examples")

    ranker_texts: list[str] = []
    ranker_labels: list[int] = []
    for example in ranker_train:
        for candidate in _candidate_records(example):
            passage_id = str(candidate["passage_id"])
            ranker_texts.append(
                _pair_text(str(example["query"]), passages_by_id[passage_id])
            )
            ranker_labels.append(int(int(candidate["relevance"]) == 3))

    passage_ranker = _make_model()
    passage_ranker.fit(ranker_texts, ranker_labels)

    answerability_model = _make_model()
    answerability_model.fit(
        [_bundle_text(example, passages_by_id) for example in train_examples],
        [int(example["answerability"] == "answerable") for example in train_examples],
    )

    reranker = BM25Reranker()
    verifier = GroundingVerifier()
    evaluation: dict[str, object] = {
        "dataset": {
            "topics": len({str(example["topic_group"]) for example in examples}),
            "evidence_bundles": len(examples),
            "ranker_training_pairs": len(ranker_texts),
            "split_bundles": dict(Counter(str(example["split"]) for example in examples)),
            "annotation_status": "assistant_draft",
            "caveat": (
                "Labels are generated from matched-source inclusion and curated negatives, "
                "not independently human-judged. Metrics are preliminary and do not establish "
                "real-world answer quality."
            ),
        },
        "versions": {"scikit_learn": sklearn.__version__},
        "passage_ranker": {},
        "answerability": {},
    }

    for split_name, split_examples in (
        ("validation", ranker_validation),
        ("test", ranker_test),
    ):
        model_scores = lambda example: _model_ranking_scores(  # noqa: E731
            passage_ranker, example, passages_by_id
        )
        bm25_scores = lambda example: _bm25_ranking(  # noqa: E731
            example, passages_by_id, reranker
        )[0]
        evaluation["passage_ranker"][split_name] = {
            "tfidf_logistic_regression": _ranking_metrics(
                split_examples,
                model_scores,
            ),
            "bm25_baseline": _ranking_metrics(split_examples, bm25_scores),
        }

    for split_name, split_examples in (
        ("validation", validation_examples),
        ("test", test_examples),
    ):
        probabilities = _positive_probability(
            answerability_model,
            [_bundle_text(example, passages_by_id) for example in split_examples],
        )
        model_predictions = [int(probability >= 0.5) for probability in probabilities]
        evaluation["answerability"][split_name] = {
            "tfidf_logistic_regression": _classification_metrics(
                split_examples,
                model_predictions,
            ),
            "bm25_grounding_baseline": _classification_metrics(
                split_examples,
                _baseline_answerability(
                    split_examples,
                    passages_by_id,
                    reranker,
                    verifier,
                ),
            ),
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(passage_ranker, MODEL_DIR / "passage_ranker.joblib")
    joblib.dump(answerability_model, MODEL_DIR / "answerability_classifier.joblib")
    report_path = DATA_DIR / "model_evaluation.json"
    report_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    print(f"Saved models to {MODEL_DIR}")
    print(f"Saved evaluation to {report_path}")


if __name__ == "__main__":
    main()
