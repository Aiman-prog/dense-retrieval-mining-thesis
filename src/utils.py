"""Utility functions for dataset loading and indexing."""

import json
import os
from typing import Optional, Dict, Any

import pyterrier as pt
import pandas as pd


def init_pyterrier() -> None:
    """Initialize PyTerrier (idempotent - safe to call multiple times)."""
    if not pt.started():
        pt.init()


def load_dataset(name: str = "msmarco_passage") -> pt.datasets.Dataset:
    """Load a PyTerrier dataset.

    Args:
        name: Dataset name (default: "msmarco_passage").

    Returns:
        PyTerrier dataset object.
    """
    init_pyterrier()
    return pt.get_dataset(name)


def load_corpus_sample(
    dataset: pt.datasets.Dataset,
    max_docs: Optional[int] = None
) -> pd.DataFrame:
    """Load a sample of documents from a dataset.

    Args:
        dataset: PyTerrier dataset.
        max_docs: Maximum number of documents to load. If None, loads all.

    Returns:
        DataFrame with 'docno' and 'text' columns.
    """
    corpus_iter = dataset.get_corpus_iter()
    if max_docs is not None:
        docs = [doc for i, doc in enumerate(corpus_iter) if i < max_docs]
    else:
        docs = list(corpus_iter)
    return pd.DataFrame(docs)


def load_topics(
    dataset: pt.datasets.Dataset,
    variant: Optional[str] = None
) -> pd.DataFrame:
    """Load topics (queries) from a dataset.

    Args:
        dataset: PyTerrier dataset.
        variant: Dataset variant (e.g., 'dev.small', 'train', 'dev').

    Returns:
        DataFrame with 'qid' and 'query' columns.
    """
    topics = dataset.get_topics(variant=variant)
    if not isinstance(topics, pd.DataFrame):
        topics = pd.DataFrame(topics)
    return topics


def load_qrels(
    dataset: pt.datasets.Dataset,
    variant: Optional[str] = None
) -> pd.DataFrame:
    """Load relevance judgments (qrels) from a dataset.

    Args:
        dataset: PyTerrier dataset.
        variant: Dataset variant (e.g., 'dev.small', 'train', 'dev').

    Returns:
        DataFrame with 'qid', 'docno', and 'label' columns.
    """
    qrels = dataset.get_qrels(variant=variant)
    if not isinstance(qrels, pd.DataFrame):
        qrels = pd.DataFrame(qrels)
    return qrels


def save_evaluation_results(
    results: Dict[str, Any],
    output_file: str,
    model_path: str,
    variant: str,
    num_queries: int,
    num_qrels: int
) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: Dictionary containing evaluation metrics (dense_ndcg, dense_mrr, etc.).
        output_file: Path to save the JSON file.
        model_path: Path or identifier of the model used.
        variant: Dataset variant used for evaluation.
        num_queries: Number of queries evaluated.
        num_qrels: Number of relevance judgments.
    """
    results_to_save = {
        'model_path': model_path,
        'variant': variant,
        'dense_ndcg': float(results['dense_ndcg']),
        'dense_mrr': float(results['dense_mrr']),
        'bm25_ndcg': float(results['bm25_ndcg']),
        'bm25_mrr': float(results['bm25_mrr']),
        'num_queries': num_queries,
        'num_qrels': num_qrels
    }
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)



