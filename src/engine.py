"""Core Search Engine module for Dense Retrieval."""

from typing import Any, Optional, Union
import os
import numpy as np
import pandas as pd
import faiss
import pyterrier as pt
from sentence_transformers import SentenceTransformer

from src.utils import get_device, init_pyterrier, load_dataset, load_corpus_sample


class DenseEngine:
    """Dense retrieval engine using SentenceTransformer models.
    
    Uses FAISS for efficient similarity search with cosine similarity
    (via inner product on L2-normalized embeddings).
    """

    def __init__(self, model_name_or_path: str) -> None:
        """Initialize the dense retrieval engine.
        
        Args:
            model_name_or_path: Path to SentenceTransformer model or HuggingFace model ID.
        """
        # Cache location and offline mode are controlled via environment variables set in shell script:
        # HF_HOME, TRANSFORMERS_CACHE, SENTENCE_TRANSFORMERS_HOME, HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE
        # Don't use cache_folder parameter - let SentenceTransformer use environment variables automatically
        self.model = SentenceTransformer(model_name_or_path)
        
        self.device = get_device()
        self.model = self.model.to(self.device)
        self.faiss_index: Optional[faiss.Index] = None
        self.docno_map: Optional[np.ndarray] = None
        self.text_map: Optional[np.ndarray] = None

    def index(self, dataset: Any, index_path: Optional[str] = None) -> None:
        """Index documents using dense embeddings.
        
        Args:
            dataset: Dataset with 'docno' and 'text' columns, or PyTerrier dataset.
            index_path: Optional path to save/load FAISS index. If provided and exists, 
                       index will be loaded instead of rebuilt.
        """
        # Try to load existing index
        if index_path and os.path.exists(f"{index_path}.faiss") and os.path.exists(f"{index_path}.meta.npz"):
            print(f"Loading existing index from {index_path}...")
            self.faiss_index = faiss.read_index(f"{index_path}.faiss")
            meta = np.load(f"{index_path}.meta.npz", allow_pickle=True)
            self.docno_map, self.text_map = meta['docno'], meta['text']
            if self.docno_map is not None:
                print(f"Index loaded ({len(self.docno_map)} documents)")
            return
        
        # Load documents
        from src.utils import init_pyterrier
        init_pyterrier()
        
        if hasattr(dataset, 'get_corpus_iter'):
            docs_df = pd.DataFrame(list(dataset.get_corpus_iter()))
        else:
            docs_df = dataset.copy()
        
        if 'docno' not in docs_df.columns or 'text' not in docs_df.columns:
            raise ValueError("Dataset must have 'docno' and 'text' columns")

        # Encode documents to embeddings
        doc_embeddings = self.model.encode(
            docs_df['text'].tolist(),
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
            device=self.device
        )

        # Create FAISS index for cosine similarity (inner product on normalized vectors)
        # IndexFlatIP = Inner Product (for cosine similarity with normalized vectors)
        embedding_dim = doc_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        
        # Normalize embeddings for cosine similarity
        doc_embeddings = doc_embeddings.astype('float32')
        faiss.normalize_L2(doc_embeddings)  # type: ignore[arg-type]
        
        # Add embeddings to index
        assert self.faiss_index is not None  # Type narrowing for type checker
        self.faiss_index.add(doc_embeddings)  # type: ignore[call-arg]
        
        # Store document metadata (convert to numpy arrays)
        self.docno_map = np.asarray(docs_df['docno'].values)
        self.text_map = np.asarray(docs_df['text'].values)

        # Save index if path provided
        if index_path:
            os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else '.', exist_ok=True)
            assert self.faiss_index is not None
            faiss.write_index(self.faiss_index, f"{index_path}.faiss")
            assert self.docno_map is not None and self.text_map is not None
            np.savez_compressed(f"{index_path}.meta", docno=self.docno_map, text=self.text_map)
            print(f"Index saved to {index_path}.faiss and {index_path}.meta.npz")

    @staticmethod
    def build_and_index_corpus(
        model_path: str,
        dataset: Optional[Union[pt.datasets.Dataset, pd.DataFrame]] = None,
        max_docs: Optional[int] = None,
        index_path: Optional[str] = None
    ) -> 'DenseEngine':
        """Build and index a corpus using a dense model.
        
        Args:
            model_path: Path to SentenceTransformer model or HuggingFace model ID.
            dataset: PyTerrier dataset or DataFrame. If None, loads MS MARCO dataset.
            max_docs: Maximum number of documents to index. If None, indexes full corpus.
            index_path: Optional path to save/load FAISS index.
            
        Returns:
            DenseEngine instance with indexed corpus.
        """
        # Initialize PyTerrier if needed
        init_pyterrier()
        
        # Load dataset if not provided
        if dataset is None:
            dataset = load_dataset()
        
        # Create engine
        engine = DenseEngine(model_path)
        
        # Determine index path if not provided
        if index_path is None:
            scratch_dir = os.environ.get('SCRATCH_DIR')
            if scratch_dir:
                index_path = f"{scratch_dir}/dense_index"
            else:
                index_path = "./dense_index"
        
        # Index corpus (with or without sampling)
        if max_docs is not None:
            print(f"Indexing corpus sample ({max_docs} documents) for DenseEngine...")
            if isinstance(dataset, pd.DataFrame):
                corpus_df = dataset.head(max_docs)
            else:
                corpus_df = load_corpus_sample(dataset, max_docs=max_docs)
            engine.index(corpus_df, index_path=index_path)
        else:
            print("Indexing full corpus (this may take a while) for DenseEngine...")
            engine.index(dataset, index_path=index_path)
        
        return engine

    @staticmethod
    def build_bm25_index(
        dataset: Optional[Union[pt.datasets.Dataset, pd.DataFrame]] = None,
        max_docs: Optional[int] = None,
        index_path: str = "./bm25_index",
        top_k: int = 10
    ) -> pt.Transformer:
        """Build and return a BM25 retrieval pipeline.
        
        Args:
            dataset: PyTerrier dataset or DataFrame. If None, loads MS MARCO dataset.
            max_docs: Maximum number of documents to index. If None, indexes full corpus.
            index_path: Path to store BM25 index.
            top_k: Number of documents to retrieve.
            
        Returns:
            PyTerrier transformer pipeline for BM25 retrieval.
        """
        # Initialize PyTerrier if needed
        init_pyterrier()
        
        # Load dataset if not provided
        if dataset is None:
            dataset = load_dataset()
        
        # Ensure index directory exists
        os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else '.', exist_ok=True)
        
        # Build index with or without sampling
        if max_docs is not None:
            print(f"Building BM25 index with {max_docs} documents...")
            if isinstance(dataset, pd.DataFrame):
                corpus_df = dataset.head(max_docs)
            else:
                corpus_df = load_corpus_sample(dataset, max_docs=max_docs)
            corpus_iter = (row.to_dict() for _, row in corpus_df.iterrows())
            indexer = pt.IterDictIndexer(index_path, overwrite=True, fields=['text'])  # type: ignore[arg-type]
            indexref = indexer.index(corpus_iter)  # type: ignore[assignment]
            assert indexref is not None  # Type narrowing for type checker
            retriever = pt.terrier.Retriever(indexref, wmodel="BM25")  # type: ignore[assignment]
            assert retriever is not None  # Type narrowing for type checker
            return retriever % top_k
        else:
            print("Building BM25 index with full corpus...")
            if isinstance(dataset, pd.DataFrame):
                corpus_iter = (row.to_dict() for _, row in dataset.iterrows())
            else:
                corpus_iter = dataset.get_corpus_iter()
            indexer = pt.IterDictIndexer(index_path, overwrite=True, fields=['text'])  # type: ignore[arg-type]
            indexref = indexer.index(corpus_iter)  # type: ignore[assignment]
            assert indexref is not None  # Type narrowing for type checker
            retriever = pt.terrier.Retriever(indexref, wmodel="BM25")  # type: ignore[assignment]
            assert retriever is not None  # Type narrowing for type checker
            return retriever % top_k

    def search(self, queries: Any, top_k: int = 10) -> pd.DataFrame:
        """Perform dense retrieval for given queries.
        
        Args:
            queries: List of query strings, DataFrame with 'query' column, or PyTerrier topics.
            top_k: Number of top documents to retrieve per query.
            
        Returns:
            DataFrame with columns: 'qid', 'docno', 'score', 'rank'.
        """
        if self.faiss_index is None:
            raise RuntimeError("Index not built. Call index() first.")

        # Normalize query input format
        if isinstance(queries, list):
            queries_df = pd.DataFrame({'qid': range(len(queries)), 'query': queries})
        elif hasattr(queries, 'get_topics'):
            queries_df = queries.get_topics()
        else:
            queries_df = queries.copy()
        
        if 'query' not in queries_df.columns:
            raise ValueError("Queries must have 'query' column")
        if 'qid' not in queries_df.columns:
            queries_df['qid'] = queries_df.index

        # Encode queries to embeddings
        query_embeddings = self.model.encode(
            queries_df['query'].tolist(),
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
            device=self.device
        )

        # Normalize query embeddings for cosine similarity
        query_embeddings = query_embeddings.astype('float32')
        faiss.normalize_L2(query_embeddings)  # type: ignore[arg-type]
        
        # Search in FAISS index
        assert self.faiss_index is not None and self.docno_map is not None
        scores, indices = self.faiss_index.search(query_embeddings, top_k)  # type: ignore[call-arg]

        # Format results
        results = []
        for i, qid in enumerate(queries_df['qid']):
            for rank, (doc_idx, score) in enumerate(zip(indices[i], scores[i]), start=1):
                if doc_idx >= 0:  # Valid result (FAISS returns -1 for invalid results)
                    results.append({
                        'qid': qid,
                        'docno': self.docno_map[doc_idx],
                        'score': float(score),
                        'rank': rank
                    })
        return pd.DataFrame(results)

if __name__ == "__main__":
    from src.utils import load_dataset, load_corpus_sample
    dataset = load_dataset()
    engine = DenseEngine('sentence-transformers/distilbert-base-uncased')
    engine.index(load_corpus_sample(dataset, max_docs=1000))
    query = input("\nEnter your query: ")
    results = engine.search([query], top_k=5)
    print(f"\nTop 5 results for '{query}':\n")
    assert engine.docno_map is not None and engine.text_map is not None
    for _, row in results.iterrows():
        doc_idx = np.where(engine.docno_map == row['docno'])[0][0]
        print(f"[{row['rank']}] (score: {row['score']:.4f}) {engine.text_map[doc_idx][:150]}...")
