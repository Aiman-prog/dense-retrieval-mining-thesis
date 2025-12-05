"""Evaluation module using PyTerrier's high-level API."""

import argparse
from typing import Dict, Any, Optional, Literal

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


def evaluate(
    model_type: Literal['dense', 'sparse'],
    model_path: str,
    variant: str = 'dev.small',
    max_docs: Optional[int] = None,
    dense_index_path: Optional[str] = None,
    sparse_index_path: str = "./bm25_index",
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate a retrieval model (dense or sparse) against BM25 baseline.

    Args:
        model_type: Type of model ('dense' or 'sparse').
        model_path: Path to model (required). For dense: SentenceTransformer model path or HuggingFace ID.
                    For sparse: ignored (uses BM25).
        variant: Dataset variant to use (default: 'dev.small' for quick testing).
        max_docs: Maximum number of documents to index (None for full corpus).
        dense_index_path: Optional path to save/load dense index.
        sparse_index_path: Path to store sparse/BM25 index.
        output_file: Optional path to save results as JSON.

    Returns:
        Dictionary containing experiment results with metrics.
        
    Raises:
        ValueError: If model_path is not provided or model_type is invalid.
    """
    if not model_path:
        raise ValueError("model_path is required. Please provide a model path.")
    
    if model_type not in ['dense', 'sparse']:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'dense' or 'sparse'.")
    
    # Initialize PyTerrier and load dataset
    init_pyterrier()
    dataset = load_dataset()
    topics = load_topics(dataset, variant=variant)
    qrels = load_qrels(dataset, variant=variant)

    print(f"Loaded {len(topics)} queries and {len(qrels)} relevance judgments")

    # Build and index corpus based on model type
    if model_type == 'dense':
        print(f"\nBuilding dense retrieval model: {model_path}")
        # Handle case where argparse receives string 'None' instead of Python None
        if not dense_index_path or dense_index_path.lower() == 'none':
            dense_index_path = None
        dense_engine = DenseEngine.build_and_index_corpus(
            model_path=model_path,
            dataset=dataset,
            max_docs=max_docs,
            index_path=dense_index_path
        )
        # Create dense retrieval pipeline
        model_pipeline = pt.apply.by_query(
            lambda queries_df: dense_engine.search(queries_df, top_k=10),
            validate='ignore'  # Suppress validation warning
        )
        model_name = "Dense"
    else:  # sparse
        print(f"\nBuilding sparse retrieval model (BM25)")
        model_pipeline = DenseEngine.build_bm25_index(
            dataset=dataset,
            max_docs=max_docs,
            index_path=sparse_index_path,
            top_k=10
        )
        model_name = "Sparse"
    
    # Build BM25 baseline for comparison
    print(f"\nBuilding BM25 baseline index at {sparse_index_path}...")
    bm25_pipeline = DenseEngine.build_bm25_index(
        dataset=dataset,
        max_docs=max_docs,
        index_path=sparse_index_path,
        top_k=10
    )

    # Run experiment
    print(f"\nRunning experiment: {model_name} vs BM25...")
    experiment = pt.Experiment(
        [model_pipeline, bm25_pipeline],
        topics,
        qrels,
        eval_metrics=["ndcg_cut_10", "recip_rank"],
        names=[model_name, "BM25"],
        verbose=True
    )

    # Extract results
    model_row = experiment[experiment['name'] == model_name]
    bm25_row = experiment[experiment['name'] == 'BM25']
    # Access pandas Series values (type checker may not recognize DataFrame column access)
    model_ndcg_series: pd.Series = model_row['ndcg_cut_10']  # type: ignore[assignment]
    model_mrr_series: pd.Series = model_row['recip_rank']  # type: ignore[assignment]
    bm25_ndcg_series: pd.Series = bm25_row['ndcg_cut_10']  # type: ignore[assignment]
    bm25_mrr_series: pd.Series = bm25_row['recip_rank']  # type: ignore[assignment]
    results = {
        f'{model_name.lower()}_ndcg': float(model_ndcg_series.iloc[0]),
        f'{model_name.lower()}_mrr': float(model_mrr_series.iloc[0]),
        'bm25_ndcg': float(bm25_ndcg_series.iloc[0]),
        'bm25_mrr': float(bm25_mrr_series.iloc[0]),
        'experiment_df': experiment
    }

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"{model_name} - nDCG@10: {results[f'{model_name.lower()}_ndcg']:.4f}, MRR@10: {results[f'{model_name.lower()}_mrr']:.4f}")
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
    parser = argparse.ArgumentParser(description='Evaluate retrieval model (dense or sparse) against BM25 baseline')
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['dense', 'sparse'],
        required=True,
        help='Type of model: "dense" for SentenceTransformer models, "sparse" for BM25'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to model (required). For dense: SentenceTransformer model path or HuggingFace ID. For sparse: ignored.'
    )
    parser.add_argument(
        '--variant',
        type=str,
        default='dev.small',
        help='Dataset variant (default: dev.small for local testing, use test-2019 for cluster)'
    )
    parser.add_argument(
        '--max-docs',
        type=int,
        default=100000,
        help='Maximum number of documents to index (default: 100000, use None for full corpus)'
    )
    parser.add_argument(
        '--dense-index-path',
        type=str,
        default=None,
        help='Path to save/load dense index (default: auto-detect scratch space, or ./dense_index)'
    )
    parser.add_argument(
        '--sparse-index-path',
        type=str,
        default='./bm25_index',
        help='Path to store sparse/BM25 index (default: ./bm25_index, use scratch space on cluster)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Path to save results as JSON (optional)'
    )

    args = parser.parse_args()
    
    # Convert max_docs to None if 0 or negative
    max_docs = args.max_docs if args.max_docs and args.max_docs > 0 else None

    evaluate(
        model_type=args.model_type,
        model_path=args.model_path,
        variant=args.variant,
        max_docs=max_docs,
        dense_index_path=args.dense_index_path,
        sparse_index_path=args.sparse_index_path,
        output_file=args.output_file
    )

