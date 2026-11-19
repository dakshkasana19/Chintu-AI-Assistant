# RAG dataset

These files are a prototype dataset for Chintu's passage-ranking and answerability models. They are not large enough to support claims of reliable real-world performance. All labels are assistant-generated drafts; the prototype trained on them must be human-reviewed before integration or any accuracy claims.

## Files and schema

- `pilot_passages.jsonl`: one source passage per line. Each record includes a stable passage ID, text, page URL, Wikipedia revision ID, retrieval date, attribution, and license.
- `pilot_queries.jsonl`: one question per line, candidate passage IDs with graded relevance labels, an answerability label, and the passage IDs that support an answer.
- `topic_manifest.jsonl`: 100 curated topics, questions, categories, and selected hard-negative topics.
- `questions.csv`: a readable, spreadsheet-friendly list of the 100 questions, their topic/category, source revision URL, and split.
- `expanded_passages.jsonl`: one Wikipedia summary passage for each manifest topic.
- `expanded_queries.jsonl`: 200 evidence bundles (supported and evidence-missing per question), each with three candidate passages and draft relevance labels.
- `model_evaluation.json`: preliminary held-out metrics for the TF-IDF/logistic-regression models and existing BM25/grounding baselines.
- [`build_rag_dataset.py`](../../scripts/build_rag_dataset.py): regenerates the expanded files using the Wikipedia API and a fixed random seed.
- [`train_rag_models.py`](../../scripts/train_rag_models.py): trains the local prototype models without paid services or API keys.

The 100-row `questions.csv` is the professor-friendly question list. From the project root, activate the project virtual environment and run `python -m scripts.train_rag_models` to retrain. Model files are written under the Git-ignored `models/rag/` directory.

The expanded data has 100 answerable and 100 insufficient bundles, with 70/15/15 topic groups in train/validation/test. All candidates in a bundle come from that bundle's split; the paired supported and evidence-missing bundles for a question remain together. There are no conflicting-evidence examples yet.

Relevance is graded from `0` to `3`:

- `0`: unrelated or does not help answer the question.
- `1`: topically related, but does not substantively support the answer.
- `2`: partially useful; it supports only part of the answer.
- `3`: directly answers or materially supports the answer.

Answerability is assessed for the candidate evidence bundle, not for all knowledge available on the internet:

- `answerable`: the listed supporting passages jointly provide enough evidence.
- `insufficient`: the bundle does not answer the question.
- `conflicting`: the bundle contains material evidence that disagrees. This class is not yet represented in the pilot.

Keep all candidate passages and paired evidence conditions within their topic group's split. Splitting individual passage pairs can leak evidence between train and held-out evaluation.

## Provenance and license

The passages are short excerpts from the cited English Wikipedia article summaries. They are attributed to Wikipedia contributors and linked to the exact article revision used. Wikipedia text is available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); these excerpts retain that attribution and license. Each passage's `source_url` contains its revision ID.

The pilot's `p-tv-season` passage was an observed wrong-topic Wikipedia hit for the related query "What causes seasons on Earth?" It is reused as a hard negative for `q-season-definition`. The expanded passages are short excerpts from the cited Wikipedia summaries. Hard negatives are manually selected; random negatives are selected from other categories. A hard-negative pair that crosses a split boundary is not used. Neither dataset represents a full multi-source search distribution.

The expanded labels are constructed from the matched source and curated negatives, not independently human-judged relevance. In particular, they are useful for validating the training/evaluation pipeline, but do not by themselves establish that a learned model will improve answer quality.

## Preliminary model comparison

The held-out test contains only 15 ranking questions and 30 answerability bundles. On this generated test set, TF-IDF/logistic regression reached passage-ranking Hit@1 of 0.27 and answerability accuracy of 0.50. The existing BM25/grounding baseline reached 1.00 on both metrics. These results are preliminary and artificially easy because the labels are built around matched-source inclusion; the trained models are not integrated into Chintu because they did not outperform the baseline.
