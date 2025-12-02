"""Utility functions for dataset loading and indexing."""

import json
import os
from typing import Optional, Dict, Any

import pyterrier as pt
import pandas as pd


def init_pyterrier() -> None:
    """Initialize PyTerrier (idempotent - safe to call multiple times).
    
    Uses pt.java.init() directly to avoid online version checks.
    Assumes PyTerrier was pre-initialized on login node (see CLUSTER_SETUP.md Step 2).
    """
    if not pt.started():
        # Use explicit version to avoid online version check (required for offline mode)
        # Version 5.11 matches what was downloaded during Step 2 initialization
        # Helper version 0.0.8 was shown in the error message
        pt.init(version="5.11", helper_version="0.0.8")


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


def load_corpus_by_docnos(
    dataset: pt.datasets.Dataset,
    docnos: set
) -> Dict[str, str]:
    """Load corpus documents by their docnos.
    
    Args:
        dataset: PyTerrier dataset.
        docnos: Set of document IDs to load.
        
    Returns:
        Dictionary mapping docno to text.
    """
    corpus_dict = {}
    corpus_iter = dataset.get_corpus_iter()
    for doc in corpus_iter:
        docno = doc['docno']
        if docno in docnos:
            corpus_dict[docno] = doc['text']
            if len(corpus_dict) >= len(docnos):
                break
    return corpus_dict


def prepare_evaluation_data(
    dataset: pt.datasets.Dataset,
    variant: str = 'eval.small'
) -> tuple:
    """Prepare evaluation data for InformationRetrievalEvaluator.
    
    Args:
        dataset: PyTerrier dataset.
        variant: Dataset variant (default: 'eval.small').
        
    Returns:
        Tuple of (queries_dict, corpus_dict, relevant_docs_dict).
    """
    eval_topics = load_topics(dataset, variant=variant)
    eval_qrels = load_qrels(dataset, variant=variant)
    
    # Load corpus for evaluation (only documents in eval qrels)
    eval_docnos = set(eval_qrels['docno'].unique())
    eval_corpus = load_corpus_by_docnos(dataset, eval_docnos)
    
    # Format: {query_id: query_text}
    queries_dict = dict(zip(eval_topics['qid'], eval_topics['query']))
    
    # Build relevance dict: {qid: {docno: score}}
    relevant_docs = {}
    for _, row in eval_qrels.iterrows():
        qid = row['qid']
        docno = row['docno']
        score = row['label']
        if qid not in relevant_docs:
            relevant_docs[qid] = {}
        relevant_docs[qid][docno] = float(score)
    
    return queries_dict, eval_corpus, relevant_docs


def plot_training_loss(
    loss_history: list,
    output_path: str
) -> None:
    """Plot and save training loss graph.
    
    Args:
        loss_history: List of (step, loss) tuples or list of losses.
        output_path: Path to save the plot.
    """
    import matplotlib.pyplot as plt
    
    if not loss_history:
        print("Note: No loss history to plot.")
        return
    
    try:
        # Extract steps and losses
        if isinstance(loss_history[0], (tuple, list)) and len(loss_history[0]) == 2:
            steps, losses = zip(*loss_history)
        else:
            steps = range(len(loss_history))
            losses = loss_history
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title('Training Loss Over Time', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training loss plot saved to: {output_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot training loss: {e}")


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



