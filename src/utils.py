"""Utility functions for dataset loading and indexing."""

from typing import Optional

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



