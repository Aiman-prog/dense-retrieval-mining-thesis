"""Utility functions for dataset loading and indexing."""

import json
import os
from typing import Optional, Dict, Any

import pyterrier as pt
import pandas as pd
import torch


def get_device() -> str:
    """Get the best available device for PyTorch (CUDA > MPS > CPU).
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_and_print_device() -> str:
    """Get the best available device and print device information.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    device = get_device()
    if device == 'cuda':
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    elif device == 'mps':
        print(f"✓ Using Apple Silicon GPU (MPS)")
    else:
        print(f"✓ Using CPU")
    return device


def init_pyterrier() -> None:
    """Initialize PyTerrier (idempotent - safe to call multiple times).
    
    Uses pt.terrier.set_version() and pt.java.init() directly to avoid online version checks.
    Assumes PyTerrier was pre-initialized on login node (see CLUSTER_SETUP.md Step 2).
    """
    if not pt.started():
        # Use explicit version to avoid online version check (required for offline mode)
        # Version 5.11 matches what was downloaded during Step 2 initialization
        # Helper version 0.0.8 was shown in the error message
        pt.terrier.set_version('5.11')
        pt.terrier.set_helper_version('0.0.8')
        pt.java.init()  # optional, forces java initialisation


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


def build_query_document_mapping(
    qrels: pd.DataFrame,
    corpus_dict: Optional[Dict[str, str]] = None,
    min_label: int = 1,
    return_texts: bool = True
) -> Dict[str, Any]:
    """Build a mapping from query IDs to their relevant documents.
    
    Can return either passage texts (for training) or docno-to-score mappings (for evaluation).
    
    Args:
        qrels: DataFrame with 'qid', 'docno', and 'label' columns.
        corpus_dict: Optional dictionary mapping document ID to text. Required if return_texts=True.
        min_label: Minimum label value to include (default: 1 for relevant only).
        return_texts: If True, returns {qid: [text1, text2, ...]}. 
                     If False, returns {qid: {docno: score}}.
        
    Returns:
        Dictionary mapping query ID to either list of texts or dict of {docno: score}.
    """
    # Filter qrels by minimum label
    filtered_qrels = qrels[qrels['label'] >= min_label]
    
    if return_texts:
        if corpus_dict is None:
            raise ValueError("corpus_dict is required when return_texts=True")
        query_mapping = {}
        for _, row in filtered_qrels.iterrows():
            qid = row['qid']
            docno = row['docno']
            if qid not in query_mapping:
                query_mapping[qid] = []
            if docno in corpus_dict:
                query_mapping[qid].append(corpus_dict[docno])
        return query_mapping
    else:
        query_mapping = {}
        for _, row in filtered_qrels.iterrows():
            qid = row['qid']
            docno = row['docno']
            score = row['label']
            if qid not in query_mapping:
                query_mapping[qid] = {}
            query_mapping[qid][docno] = float(score)
        return query_mapping


def build_query_positives_mapping(
    qrels: pd.DataFrame,
    corpus_dict: Dict[str, str]
) -> Dict[str, list]:
    """Build a mapping from query IDs to their positive passages.
    
    Filters qrels to only label=1 (relevant) and maps query IDs to lists
    of positive passage texts from the corpus.
    
    Args:
        qrels: DataFrame with 'qid', 'docno', and 'label' columns.
        corpus_dict: Dictionary mapping document ID to text.
        
    Returns:
        Dictionary mapping query ID to list of positive passage texts.
    """
    return build_query_document_mapping(qrels, corpus_dict, min_label=1, return_texts=True)


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
    
    # Build relevance dict using the general mapping function
    relevant_docs = build_query_document_mapping(
        eval_qrels, 
        corpus_dict=None, 
        min_label=0,  # Include all labels for evaluation
        return_texts=False
    )
    
    return queries_dict, eval_corpus, relevant_docs


def plot_training_loss(
    loss_history: list,
    output_path: str
) -> None:
    """Plot and save training loss graph with smoothing and statistics.
    
    Args:
        loss_history: List of (step, loss) tuples or list of losses.
        output_path: Path to save the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not loss_history:
        print("Note: No loss history to plot.")
        return
    
    try:
        # Extract steps and losses
        if isinstance(loss_history[0], (tuple, list)) and len(loss_history[0]) == 2:
            steps, losses = list(zip(*loss_history))
            steps = list(steps)
            losses = list(losses)
        else:
            steps = list(range(len(loss_history)))
            losses = list(loss_history)
        
        plt.figure(figsize=(12, 6))
        
        # Plot raw loss values
        plt.plot(steps, losses, 'b-', linewidth=1.5, alpha=0.7, label='Training Loss')
        
        # Add smoothed curve for better visualization (moving average)
        if len(losses) > 50:
            window_size = max(50, len(losses) // 100)
            smoothed_loss = np.convolve(
                losses,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            smoothed_steps = steps[window_size - 1:]
            plt.plot(smoothed_steps, smoothed_loss, 'r-', linewidth=2, label='Smoothed Loss')
        
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Steps', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text box
        min_loss = min(losses)
        final_loss = losses[-1]
        initial_loss = losses[0] if len(losses) > 0 else 0
        stats_text = f'Initial: {initial_loss:.4f}\nFinal: {final_loss:.4f}\nMin: {min_loss:.4f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
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



