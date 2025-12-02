"""Training script for fine-tuning dual encoder on MS MARCO."""

import argparse
import random
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import torch

from src.utils import (
    init_pyterrier,
    load_dataset,
    load_topics,
    load_qrels,
    load_corpus_by_docnos,
    prepare_evaluation_data,
    plot_training_loss
)


def train(
    strategy: str = 'in-batch',
    model_name: str = 'distilbert-base-uncased',
    output_dir: str = './output',
    num_examples: int = 100000,
    batch_size: int = 32,
    epochs: int = 1
) -> None:
    """Train a dual encoder model on MS MARCO.
    
    Args:
        strategy: Training strategy ('in-batch' or 'random').
        model_name: Base model to fine-tune.
        output_dir: Directory to save the fine-tuned model.
        num_examples: Number of training examples to use.
        batch_size: Training batch size.
        epochs: Number of training epochs.
    """
    # Initialize PyTerrier and load dataset
    init_pyterrier()
    dataset = load_dataset()
    
    # Load training data
    print(f"Loading MS MARCO training data...")
    topics = load_topics(dataset, variant='train')
    qrels = load_qrels(dataset, variant='train')
    
    # Build corpus for document lookup (only documents referenced in qrels)
    print(f"Loading corpus for document lookup...")
    unique_docnos = set(qrels['docno'].unique())
    print(f"Found {len(unique_docnos)} unique documents in training qrels")
    corpus_dict = load_corpus_by_docnos(dataset, unique_docnos)
    print(f"Loaded {len(corpus_dict)} documents from corpus")
    
    # Create query -> positive passages mapping
    # Filter qrels to only label=1 (relevant)
    positive_qrels = qrels[qrels['label'] == 1]
    query_positives = {}
    for _, row in positive_qrels.iterrows():
        qid = row['qid']
        docno = row['docno']
        if qid not in query_positives:
            query_positives[qid] = []
        if docno in corpus_dict:
            query_positives[qid].append(corpus_dict[docno])
    
    # Initialize model first (needed for loss functions)
    model = SentenceTransformer(model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Prepare training examples
    train_examples = []
    
    if strategy == 'in-batch':
        print(f"Preparing {num_examples} (query, positive) pairs for in-batch negative sampling...")
        for i, (_, topic_row) in enumerate(topics.iterrows()):
            if len(train_examples) >= num_examples:
                break
            qid = topic_row['qid']
            if qid not in query_positives or len(query_positives[qid]) == 0:
                continue
            
            query = topic_row['query']
            # Use first positive passage for this query
            positive = query_positives[qid][0]
            train_examples.append(InputExample(texts=[query, positive]))
        
        loss = losses.MultipleNegativesRankingLoss(model)
        
    elif strategy == 'random':
        print(f"Preparing {num_examples} (query, positive, negative) triplets...")
        # Build list of all passages for random negative sampling
        all_passages = list(corpus_dict.values())
        
        for i, (_, topic_row) in enumerate(topics.iterrows()):
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
        
        loss = losses.TripletLoss(model)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be 'in-batch' or 'random'")
    
    print(f"Prepared {len(train_examples)} training examples")
    
    # Prepare evaluation data (eval.small)
    print(f"\nPreparing evaluation data (eval.small)...")
    queries_dict, eval_corpus, relevant_docs = prepare_evaluation_data(dataset, variant='eval.small')
    
    # Create evaluator
    evaluator = InformationRetrievalEvaluator(
        queries=queries_dict,
        corpus=eval_corpus,
        relevant_docs=relevant_docs,
        show_progress_bar=True
    )
    
    print(f"Training with {len(train_examples)} examples...")
    print(f"Evaluation set: {len(queries_dict)} queries, {len(eval_corpus)} documents")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model with evaluation
    training_history = model.fit(
        train_objectives=[(train_examples, loss)],
        epochs=epochs,
        batch_size=batch_size,
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=5000,  # Evaluate every 5000 steps
        output_path=output_dir
    )
    
    # Plot training loss if available
    loss_history = []
    if training_history and isinstance(training_history, dict) and 'loss' in training_history:
        loss_data = training_history['loss']
        if isinstance(loss_data, list):
            loss_history = loss_data
    
    if loss_history:
        plot_path = os.path.join(output_dir, 'training_loss.png')
        plot_training_loss(loss_history, plot_path)
    
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune dual encoder on MS MARCO')
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['in-batch', 'random'],
        default='in-batch',
        help='Training strategy: in-batch (pairs) or random (triplets)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='distilbert-base-uncased',
        help='Base model to fine-tune'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Directory to save fine-tuned model'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=100000,
        help='Number of training examples to use'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Number of training epochs'
    )
    
    args = parser.parse_args()
    
    train(
        strategy=args.strategy,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

