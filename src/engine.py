"""Core Search Engine module for Dense Retrieval."""

from typing import Any, Optional

import pyterrier as pt
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


class DenseEngine:
    """Dense retrieval engine using SentenceTransformer models.

    This class can work with both naive (pre-trained) and fine-tuned models.
    It handles document indexing and query retrieval using dense embeddings.

    Attributes:
        model: SentenceTransformer model instance.
        model_name: Name or path of the loaded model.
        _index: Dictionary containing 'docno', 'text', and 'embeddings' arrays.
    """

    def __init__(self, model_name_or_path: str) -> None:
        """Initialize the DenseEngine with a model.

        Args:
            model_name_or_path: HuggingFace model identifier or local path.
                Examples: 'sentence-transformers/all-MiniLM-L6-v2',
                          'distilbert-base-uncased', or './models/my-model'

        Raises:
            ValueError: If model loading fails.
        """
        try:
            self.model = SentenceTransformer(model_name_or_path)
            self.model_name = model_name_or_path
            self._index: Optional[dict] = None
        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name_or_path}': {e}") from e

    def index(self, dataset: Any) -> None:
        """Index documents using dense embeddings.

        Encodes documents using the SentenceTransformer model and stores
        embeddings for efficient retrieval.

        Args:
            dataset: PyTerrier dataset or pandas DataFrame with documents.
                Must have 'docno' and 'text' columns.

        Raises:
            ValueError: If dataset format is invalid.
        """
        from src.utils import init_pyterrier
        init_pyterrier()
        
        # Convert dataset to DataFrame if needed
        if hasattr(dataset, 'get_corpus_iter'):
            # Directly convert iterable to DataFrame (more efficient than manual loop)
            corpus_iter = dataset.get_corpus_iter()
            docs_df = pd.DataFrame(list(corpus_iter))
        elif isinstance(dataset, pd.DataFrame):
            docs_df = dataset
        else:
            raise ValueError("Dataset must be a PyTerrier dataset or pandas DataFrame")

        # Validate required columns
        if 'docno' not in docs_df.columns or 'text' not in docs_df.columns:
            raise ValueError("Dataset must have 'docno' and 'text' columns")

        # Encode documents
        print(f"Encoding {len(docs_df)} documents...")
        doc_texts = docs_df['text'].tolist()
        doc_embeddings = self.model.encode(
            doc_texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        # Store index data (embeddings as numpy array for efficiency)
        self._index = {
            'docno': docs_df['docno'].values,
            'text': docs_df['text'].values,
            'embeddings': doc_embeddings  # Keep as numpy array
        }

        print(f"Index built successfully with {len(docs_df)} documents")

    def search(self, queries: Any, top_k: int = 10) -> pd.DataFrame:
        """Perform dense retrieval for given queries.

        Args:
            queries: PyTerrier dataset, pandas DataFrame, or list of query strings.
                If DataFrame, must have 'qid' and 'query' columns.
            top_k: Number of top documents to retrieve per query.

        Returns:
            DataFrame with columns: ['qid', 'docno', 'score', 'rank'].

        Raises:
            RuntimeError: If index has not been built yet.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call index() first.")

        # Convert queries to DataFrame format
        if isinstance(queries, list):
            queries_df = pd.DataFrame({
                'qid': range(len(queries)),
                'query': queries
            })
        elif hasattr(queries, 'get_topics'):
            queries_df = queries.get_topics()
        elif isinstance(queries, pd.DataFrame):
            queries_df = queries.copy()
        else:
            raise ValueError("Queries must be a list, PyTerrier dataset, or DataFrame")

        # Validate query format
        if 'query' not in queries_df.columns:
            raise ValueError("Queries must have 'query' column")

        # Ensure qid column exists
        if 'qid' not in queries_df.columns:
            queries_df['qid'] = queries_df.index

        # Encode queries
        query_texts = queries_df['query'].tolist()
        query_embeddings = self.model.encode(
            query_texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        # Get document embeddings and docnos from index
        doc_embeddings = self._index['embeddings']
        docnos = self._index['docno']

        # Normalize document embeddings for cosine similarity
        doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        doc_embeddings_norm = doc_embeddings / (doc_norms + 1e-8)

        # Normalize query embeddings
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings_norm = query_embeddings / (query_norms + 1e-8)

        # Vectorized cosine similarity: (n_queries, n_docs)
        similarity_matrix = np.dot(query_embeddings_norm, doc_embeddings_norm.T)

        # Get top-k for each query
        results = []
        for i, qid in enumerate(queries_df['qid']):
            scores = similarity_matrix[i]
            top_indices = np.argsort(scores)[::-1][:top_k]
            for rank, doc_idx in enumerate(top_indices, start=1):
                results.append({
                    'qid': qid,
                    'docno': docnos[doc_idx],
                    'score': float(scores[doc_idx]),
                    'rank': rank
                })

        return pd.DataFrame(results)


if __name__ == "__main__":
    """Test DenseEngine with user query."""
    from src.utils import load_dataset, load_corpus_sample
    
    # Load dataset sample and create indexed engine
    dataset = load_dataset()
    corpus_df = load_corpus_sample(dataset, max_docs=1000)
    engine = DenseEngine('sentence-transformers/all-MiniLM-L6-v2')
    engine.index(corpus_df)
    
    # Get query from user
    query = input("\nEnter your query: ")
    results = engine.search([query], top_k=5)
    
    # Display results
    print(f"\nTop 5 results for '{query}':\n")
    for idx, row in results.iterrows():
        doc_idx = np.where(engine._index['docno'] == row['docno'])[0][0]
        doc_text = engine._index['text'][doc_idx]
        print(f"[{row['rank']}] (score: {row['score']:.4f})")
        print(f"{doc_text[:150]}...\n")
