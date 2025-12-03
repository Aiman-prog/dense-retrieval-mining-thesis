"""Negative mining strategies for training dual encoders."""

import random
from typing import Dict, List
import pandas as pd
from sentence_transformers import InputExample


def mine_in_batch_negatives(
    topics: pd.DataFrame,
    query_positives: Dict[str, List[str]],
    num_examples: int
) -> List[InputExample]:
    """Mine training examples for in-batch negative sampling.
    
    Creates (query, positive) pairs. Negatives are sampled from other examples
    in the same batch during training.
    
    Args:
        topics: DataFrame with 'qid' and 'query' columns.
        query_positives: Dictionary mapping query ID to list of positive passages.
        num_examples: Maximum number of training examples to create.
        
    Returns:
        List of InputExample objects with (query, positive) pairs.
    """
    train_examples = []
    
    for _, topic_row in topics.iterrows():
        if len(train_examples) >= num_examples:
            break
        qid = topic_row['qid']
        if qid not in query_positives or len(query_positives[qid]) == 0:
            continue
        
        query = topic_row['query']
        # Use first positive passage for this query
        positive = query_positives[qid][0]
        train_examples.append(InputExample(texts=[query, positive]))
    
    return train_examples


def mine_random_negatives(
    topics: pd.DataFrame,
    query_positives: Dict[str, List[str]],
    corpus_dict: Dict[str, str],
    num_examples: int
) -> List[InputExample]:
    """Mine training examples with random negative sampling.
    
    Creates (query, positive, negative) triplets by randomly sampling
    negatives from the corpus.
    
    Args:
        topics: DataFrame with 'qid' and 'query' columns.
        query_positives: Dictionary mapping query ID to list of positive passages.
        corpus_dict: Dictionary mapping document ID to text.
        num_examples: Maximum number of training examples to create.
        
    Returns:
        List of InputExample objects with (query, positive, negative) triplets.
    """
    train_examples = []
    # Build list of all passages for random negative sampling
    all_passages = list(corpus_dict.values())
    
    for _, topic_row in topics.iterrows():
        if len(train_examples) >= num_examples:
            break
        qid = topic_row['qid']
        if qid not in query_positives or len(query_positives[qid]) == 0:
            continue
        
        query = topic_row['query']
        positive = query_positives[qid][0]
        # Sample random negative (ensure it's different from positive)
        negative = random.choice(all_passages)
        while negative == positive and len(all_passages) > 1:
            negative = random.choice(all_passages)
        
        train_examples.append(InputExample(texts=[query, positive, negative]))
    
    return train_examples


def mine_negatives(
    strategy: str,
    topics: pd.DataFrame,
    query_positives: Dict[str, List[str]],
    corpus_dict: Dict[str, str] = None,
    num_examples: int = 1000
) -> List[InputExample]:
    """Mine training examples using the specified negative sampling strategy.
    
    Args:
        strategy: Negative mining strategy ('in-batch' or 'random').
        topics: DataFrame with 'qid' and 'query' columns.
        query_positives: Dictionary mapping query ID to list of positive passages.
        corpus_dict: Dictionary mapping document ID to text (required for 'random' strategy).
        num_examples: Maximum number of training examples to create.
        
    Returns:
        List of InputExample objects.
        
    Raises:
        ValueError: If strategy is unknown or corpus_dict is missing for 'random' strategy.
    """
    if strategy == 'in-batch':
        print(f"Preparing {num_examples} (query, positive) pairs for in-batch negative sampling...")
        print("Using contrastive loss (MultipleNegativesRankingLoss) with in-batch negatives")
        return mine_in_batch_negatives(topics, query_positives, num_examples)
    
    elif strategy == 'random':
        if corpus_dict is None:
            raise ValueError("corpus_dict is required for 'random' negative mining strategy")
        print(f"Preparing {num_examples} (query, positive, negative) triplets...")
        print("Using contrastive loss (MultipleNegativesRankingLoss) with explicit negatives")
        return mine_random_negatives(topics, query_positives, corpus_dict, num_examples)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be 'in-batch' or 'random'")

