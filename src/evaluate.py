"""Evaluation module using PyTerrier's high-level API."""

import argparse
import os
from typing import Dict, Any, Optional

import pyterrier as pt
import pandas as pd

from src.engine import DenseEngine
from src.utils import (
    init_pyterrier,
    load_dataset,
    load_topics,
    load_qrels,
    save_evaluation_results
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
    indexer = pt.IterDictIndexer(index_path, overwrite=True, fields=['text'])  # type: ignore[arg-type]
    indexref = indexer.index(dataset.get_corpus_iter())  # type: ignore[assignment]
    assert indexref is not None  # Type narrowing for type checker
    # Use pt.terrier.Retriever as shown in PyTerrier documentation
    retriever = pt.terrier.Retriever(indexref, wmodel="BM25")  # type: ignore[assignment]
    assert retriever is not None  # Type narrowing for type checker
    return retriever % top_k


def evaluate(
    model_path: str,
    variant: str = 'dev.small',
    max_docs: Optional[int] = None,
    index_path: str = "./bm25_index",
    dense_index_path: Optional[str] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate a dense retrieval model against BM25 baseline.

    Args:
        model_path: Path to SentenceTransformer model (or HuggingFace ID).
        variant: Dataset variant to use (default: 'dev.small' for quick testing).
        max_docs: Maximum number of documents to index (None for full corpus).
        index_path: Path to store BM25 index (use scratch space on cluster).
        dense_index_path: Optional path to save/load dense index. If provided and exists,
                        index will be loaded instead of re-indexing.
        output_file: Optional path to save results as JSON.

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
    
    # Use provided path, or fallback to scratch if not provided
    # Note: Directory should already exist from CLUSTER_SETUP.md or run_evaluation.sh
    # Handle case where argparse receives string 'None' instead of Python None
    if not dense_index_path or dense_index_path.lower() == 'none':
        scratch_dir = os.environ.get('SCRATCH_DIR')
        if scratch_dir:
            dense_index_path = f"{scratch_dir}/dense_index"  # FAISS creates .faiss and .meta files
        else:
            dense_index_path = "./dense_index"  # Local fallback
    
    print(f"Dense index path: {dense_index_path}")
    dense_engine = DenseEngine(model_path)
    
    # Index (will load from disk if exists, otherwise index and save)
    if max_docs is not None:
        print(f"Indexing corpus sample ({max_docs} documents) for DenseEngine...")
        from src.utils import load_corpus_sample
        corpus_df = load_corpus_sample(dataset, max_docs=max_docs)
        dense_engine.index(corpus_df, index_path=dense_index_path)
    else:
        print("Indexing full corpus (this may take a while) for DenseEngine...")
        dense_engine.index(dataset, index_path=dense_index_path)

    # Create dense retrieval pipeline using pt.apply.by_query
    # This processes queries one at a time and retrieves documents (Q → Q×D)
    dense_pipeline = pt.apply.by_query(
        lambda queries_df: dense_engine.search(queries_df, top_k=10)
    )
    
    print(f"\nBuilding BM25 baseline index at {index_path}...")
    # Ensure index directory exists
    os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else '.', exist_ok=True)
    # Use same corpus sample for BM25 if max_docs is set
    if max_docs is not None:
        from src.utils import load_corpus_sample
        corpus_df = load_corpus_sample(dataset, max_docs=max_docs)
        corpus_iter = (row.to_dict() for _, row in corpus_df.iterrows())
        indexer = pt.IterDictIndexer(index_path, overwrite=True, fields=['text'])  # type: ignore[arg-type]
        indexref = indexer.index(corpus_iter)  # type: ignore[assignment]
        assert indexref is not None  # Type narrowing for type checker
        retriever = pt.terrier.Retriever(indexref, wmodel="BM25")  # type: ignore[assignment]
        assert retriever is not None  # Type narrowing for type checker
        bm25_pipeline = retriever % 10
    else:
        bm25_pipeline = create_bm25_baseline(dataset, index_path=index_path, top_k=10)

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
    dense_row = experiment[experiment['name'] == 'Dense']
    bm25_row = experiment[experiment['name'] == 'BM25']
    # Access pandas Series values (type checker may not recognize DataFrame column access)
    dense_ndcg_series: pd.Series = dense_row['ndcg_cut_10']  # type: ignore[assignment]
    dense_mrr_series: pd.Series = dense_row['recip_rank']  # type: ignore[assignment]
    bm25_ndcg_series: pd.Series = bm25_row['ndcg_cut_10']  # type: ignore[assignment]
    bm25_mrr_series: pd.Series = bm25_row['recip_rank']  # type: ignore[assignment]
    results = {
        'dense_ndcg': float(dense_ndcg_series.iloc[0]),
        'dense_mrr': float(dense_mrr_series.iloc[0]),
        'bm25_ndcg': float(bm25_ndcg_series.iloc[0]),
        'bm25_mrr': float(bm25_mrr_series.iloc[0]),
        'experiment_df': experiment
    }

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Dense - nDCG@10: {results['dense_ndcg']:.4f}, MRR@10: {results['dense_mrr']:.4f}")
    print(f"BM25  - nDCG@10: {results['bm25_ndcg']:.4f}, MRR@10: {results['bm25_mrr']:.4f}")
    print("="*60)

    # Save results to file if specified
    if output_file:
        save_evaluation_results(
            results=results,
            output_file=output_file,
            model_path=model_path,
            variant=variant,
            num_queries=len(topics),
            num_qrels=len(qrels)
        )
        print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    """Run evaluation with command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate dense retrieval model against BM25 baseline')
    parser.add_argument(
        '--model-path',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Path to SentenceTransformer model or HuggingFace ID'
    )
    parser.add_argument(
        '--variant',
        type=str,
        default='dev.small',
        help='Dataset variant (default: dev.small for local testing, use test-2019 for cluster)'
    )
    parser.add_argument(
        '--index-path',
        type=str,
        default='./bm25_index',
        help='Path to store BM25 index (default: ./bm25_index, use scratch space on cluster)'
    )
    parser.add_argument(
        '--dense-index-path',
        type=str,
        default=None,
        help='Path to save/load dense index (default: auto-detect scratch space, or ./dense_index)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Path to save results as JSON (optional)'
    )

    args = parser.parse_args()
    
    # Debug: show what arguments were parsed
    print(f"DEBUG: dense_index_path argument: '{args.dense_index_path}'")

    # Use subset for testing (100k documents - large enough for meaningful results)
    # Set to None for full corpus, or an integer for testing with a subset
    max_docs = 100000  # 100k documents for testing (should have overlap with qrels)

    evaluate(
        model_path=args.model_path,
        variant=args.variant,
        max_docs=max_docs,
        index_path=args.index_path,
        dense_index_path=args.dense_index_path,
        output_file=args.output_file
    )

