from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .metrics import anomaly_metrics, normal_score_threshold
from .rescoring import empirical_normal_percentile
from .trainer import Trainer


def fit_markov_counts(
    sequences: list[list[int]], order: int
) -> tuple[Counter[tuple[int, ...]], Counter[tuple[int, ...]]]:
    counts: Counter[tuple[int, ...]] = Counter()
    contexts: Counter[tuple[int, ...]] = Counter()
    prefix = [-1] * (order - 1)
    for sequence in sequences:
        padded = prefix + sequence
        for index in range(order - 1, len(padded)):
            gram = tuple(padded[index - order + 1 : index + 1])
            counts[gram] += 1
            contexts[gram[:-1]] += 1
    return counts, contexts


def markov_nll_scores(
    sequences: list[list[int]],
    counts: Counter[tuple[int, ...]],
    contexts: Counter[tuple[int, ...]],
    *,
    order: int,
    vocabulary_size: int,
) -> np.ndarray:
    values = []
    prefix = [-1] * (order - 1)
    for sequence in sequences:
        padded = prefix + sequence
        losses = []
        for index in range(order - 1, len(padded)):
            gram = tuple(padded[index - order + 1 : index + 1])
            probability = (counts[gram] + 1) / (contexts[gram[:-1]] + vocabulary_size)
            losses.append(-np.log(probability))
        values.append(float(np.mean(losses)) if losses else 0.0)
    return np.asarray(values, dtype=np.float64)


def _sequence_text(sequences: list[list[int]]) -> list[str]:
    return [" ".join(map(str, sequence)) for sequence in sequences]


def _read_fixed_ids(path: str | Path | None) -> set[int] | None:
    if path is None:
        return None
    values = Path(path).read_text(encoding="utf-8").splitlines()
    return {int(value.strip()) for value in values if value.strip()}


def rescore_adfa_hybrid(
    config: dict[str, Any],
    checkpoint: str | Path,
    *,
    fixed_test_ids: str | Path | None = None,
    unigram_weight: float = 0.5,
    markov_weight: float = 0.25,
    markov_order: int = 3,
) -> dict[str, Any]:
    if markov_order < 1:
        raise ValueError("markov_order must be at least 1")
    if unigram_weight < 0 or markov_weight < 0:
        raise ValueError("Hybrid score weights must be non-negative")
    trainer = Trainer(config)
    trainer.load_checkpoint(checkpoint)
    normal = trainer.evaluate("train", return_details=True)
    test = trainer.evaluate("test", return_details=True)
    if any(int(label) != 0 for label in normal["_labels"]):
        raise ValueError("ADFA hybrid calibration requires a normal-only train split")
    if not normal["_graph_ids"]:
        raise ValueError("ADFA hybrid calibration requires at least one training graph")
    graph_index = {
        trainer.dataset[index].graph_id: index for index in range(len(trainer.dataset))
    }

    def sequences(graph_ids: list[int]) -> list[list[int]]:
        return [
            trainer.dataset[graph_index[int(graph_id)]].node_type.tolist()
            for graph_id in graph_ids
        ]

    normal_sequences = sequences(normal["_graph_ids"])
    test_sequences = sequences(test["_graph_ids"])
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 1),
        min_df=2,
        sublinear_tf=True,
    )
    normal_tfidf = vectorizer.fit_transform(_sequence_text(normal_sequences))
    test_tfidf = vectorizer.transform(_sequence_text(test_sequences))
    neighbor_count = min(2, normal_tfidf.shape[0])
    neighbors = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
    ).fit(normal_tfidf)
    normal_unigram = neighbors.kneighbors(normal_tfidf)[0][:, -1]
    test_unigram = neighbors.kneighbors(test_tfidf, n_neighbors=1)[0].reshape(-1)

    markov_counts, markov_contexts = fit_markov_counts(normal_sequences, markov_order)
    vocabulary_size = int(config["dataset"]["num_node_types"])
    normal_markov = markov_nll_scores(
        normal_sequences,
        markov_counts,
        markov_contexts,
        order=markov_order,
        vocabulary_size=vocabulary_size,
    )
    test_markov = markov_nll_scores(
        test_sequences,
        markov_counts,
        markov_contexts,
        order=markov_order,
        vocabulary_size=vocabulary_size,
    )

    normal_svdd_percentile = empirical_normal_percentile(
        normal["_svdd_scores"], normal["_svdd_scores"]
    )
    test_svdd_percentile = empirical_normal_percentile(
        normal["_svdd_scores"], test["_svdd_scores"]
    )
    normal_unigram_percentile = empirical_normal_percentile(
        normal_unigram.tolist(), normal_unigram.tolist()
    )
    test_unigram_percentile = empirical_normal_percentile(
        normal_unigram.tolist(), test_unigram.tolist()
    )
    normal_markov_percentile = empirical_normal_percentile(
        normal_markov.tolist(), normal_markov.tolist()
    )
    test_markov_percentile = empirical_normal_percentile(
        normal_markov.tolist(), test_markov.tolist()
    )
    normal_scores = (
        normal_svdd_percentile
        + unigram_weight * normal_unigram_percentile
        + markov_weight * normal_markov_percentile
    )
    test_scores = (
        test_svdd_percentile
        + unigram_weight * test_unigram_percentile
        + markov_weight * test_markov_percentile
    )
    threshold = normal_score_threshold(
        normal_scores.tolist(),
        float(config["evaluation"].get("threshold_quantile", 0.99)),
    )
    metric_arguments = {
        "threshold": threshold,
        "alert_fraction": float(config["evaluation"].get("alert_fraction", 0.01)),
        "target_fpr": float(config["evaluation"].get("target_fpr", 0.01)),
    }
    full_metrics = anomaly_metrics(
        test["_labels"], test_scores.tolist(), **metric_arguments
    )

    selected_ids = _read_fixed_ids(fixed_test_ids)
    mask = np.ones(len(test_scores), dtype=bool)
    if selected_ids is not None:
        mask = np.asarray(
            [int(graph_id) in selected_ids for graph_id in test["_graph_ids"]]
        )
        if int(mask.sum()) != len(selected_ids):
            raise ValueError("Some fixed test graph IDs are absent from the test split")
    labels = np.asarray(test["_labels"], dtype=np.int64)[mask]
    if np.unique(labels).size < 2:
        raise ValueError("The selected test set must contain both classes")
    metrics = anomaly_metrics(labels, test_scores[mask].tolist(), **metric_arguments)

    output = trainer.output_dir
    pd.DataFrame(
        {
            "graph_id": test["_graph_ids"],
            "label": test["_labels"],
            "svdd_percentile": test_svdd_percentile,
            "unigram_knn_percentile": test_unigram_percentile,
            "markov_nll_percentile": test_markov_percentile,
            "hybrid_score": test_scores,
            "selected_for_fixed_test": mask,
        }
    ).to_csv(output / "adfa_hybrid_predictions.csv", index=False)
    summary = {
        **metrics,
        "full_auc": full_metrics["auc"],
        "full_ap": full_metrics["ap"],
        "dataset": config["dataset"]["name"],
        "seed": trainer.seed,
        "score_mode": "normal_ecdf_svdd_unigram_markov",
        "unigram_weight": unigram_weight,
        "markov_weight": markov_weight,
        "markov_order": markov_order,
        "calibration_source": "normal_train_only",
        "fixed_test_ids": None if fixed_test_ids is None else str(fixed_test_ids),
        "num_fixed_test_graphs": int(mask.sum()),
        "checkpoint": str(checkpoint),
        "experimental_stage": "candidate_extension",
    }
    (output / "adfa_hybrid_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
