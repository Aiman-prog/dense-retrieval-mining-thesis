"""Evaluation module using PyTerrier's high-level API."""

from typing import Dict, Any, Optional

import pyterrier as pt
import pandas as pd

from src.engine import DenseEngine
from src.utils import (
    init_pyterrier,
    load_dataset,
    load_topics,
    load_qrels
)

def create_bm25_baseline(
    dataset: pt.datasets.Dataset,
    index_path: str = "./bm25_index",
    top_k: int = 10
) -> pt.Transformer:
    """Create a BM25 baseline retrieval pipeline for evaluation.

    Args:
        dataset: PyTerrier dataset to index.
        index_path: Path to store the BM25 index.
        top_k: Number of documents to retrieve.

    Returns:
        PyTerrier transformer pipeline for BM25 retrieval.
    """
    indexer = pt.IterDictIndexer(index_path, overwrite=True, fields=['text'])
    indexref = indexer.index(dataset.get_corpus_iter())
    # Use pt.terrier.Retriever as shown in PyTerrier documentation
    return pt.terrier.Retriever(indexref, wmodel="BM25") % top_k


def evaluate(model_path: str, variant: str = 'dev.small', max_docs: Optional[int] = None) -> Dict[str, Any]:
    """Evaluate a dense retrieval model against BM25 baseline.

    Args:
        model_path: Path to SentenceTransformer model (or HuggingFace ID).
        variant: Dataset variant to use (default: 'dev.small' for quick testing).

    Returns:
        Dictionary containing experiment results with metrics.
    """
    # Initialize PyTerrier and load dataset
    init_pyterrier()
    dataset = load_dataset()
    topics = load_topics(dataset, variant=variant)
    qrels = load_qrels(dataset, variant=variant)

    print(f"Loaded {len(topics)} queries and {len(qrels)} relevance judgments")

    # Create and index DenseEngine
    print(f"\nInitializing DenseEngine with model: {model_path}")
    dense_engine = DenseEngine(model_path)
    
    if max_docs is not None:
        print(f"Indexing corpus sample ({max_docs} documents)...")
        from src.utils import load_corpus_sample
        corpus_df = load_corpus_sample(dataset, max_docs=max_docs)
        dense_engine.index(corpus_df)
    else:
        print("Indexing full corpus (this may take a while)...")
        dense_engine.index(dataset)

    # Create dense retrieval pipeline using pt.apply.by_query
    # This processes queries one at a time and retrieves documents (Q → Q×D)
    dense_pipeline = pt.apply.by_query(
        lambda queries_df: dense_engine.search(queries_df, top_k=10)
    )
    
    print("\nBuilding BM25 baseline index...")
    # Use same corpus sample for BM25 if max_docs is set
    if max_docs is not None:
        from src.utils import load_corpus_sample
        corpus_df = load_corpus_sample(dataset, max_docs=max_docs)
        corpus_iter = (row.to_dict() for _, row in corpus_df.iterrows())
        indexer = pt.IterDictIndexer("./bm25_index", overwrite=True, fields=['text'])
        indexref = indexer.index(corpus_iter)
        bm25_pipeline = pt.terrier.Retriever(indexref, wmodel="BM25") % 10
    else:
        bm25_pipeline = create_bm25_baseline(dataset, top_k=10)

    # Run experiment
    print("\nRunning experiment: Dense vs BM25...")
    experiment = pt.Experiment(
        [dense_pipeline, bm25_pipeline],
        topics,
        qrels,
        eval_metrics=["ndcg_cut_10", "recip_rank"],
        names=["Dense", "BM25"],
        verbose=True
    )

    # Extract results
    results = {
        'dense_ndcg': experiment[experiment['name'] == 'Dense']['ndcg_cut_10'].values[0],
        'dense_mrr': experiment[experiment['name'] == 'Dense']['recip_rank'].values[0],
        'bm25_ndcg': experiment[experiment['name'] == 'BM25']['ndcg_cut_10'].values[0],
        'bm25_mrr': experiment[experiment['name'] == 'BM25']['recip_rank'].values[0],
        'experiment_df': experiment
    }

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Dense - nDCG@10: {results['dense_ndcg']:.4f}, MRR@10: {results['dense_mrr']:.4f}")
    print(f"BM25  - nDCG@10: {results['bm25_ndcg']:.4f}, MRR@10: {results['bm25_mrr']:.4f}")
    print("="*60)

    return results


if __name__ == "__main__":
    """Run evaluation with naive (untuned) model."""
    model_path = 'sentence-transformers/all-MiniLM-L6-v2'
    # Use max_docs=10000 for quick testing, None for full corpus
    evaluate(model_path, variant='dev.small', max_docs=10000)

