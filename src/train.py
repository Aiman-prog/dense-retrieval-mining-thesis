"""Training script for fine-tuning dual encoder on MS MARCO."""

import argparse
import os

from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset
from transformers import TrainerCallback

from src.utils import (
    init_pyterrier,
    load_dataset,
    load_topics,
    load_qrels,
    load_corpus_by_docnos,
    build_query_positives_mapping,
    prepare_evaluation_data,
    plot_training_loss,
    get_and_print_device
)
from src.mine_negatives import mine_negatives


class LossTrackingCallback(TrainerCallback):
    """Callback to track training loss during sentence-transformers training."""
    def __init__(self):
        self.losses = []
        self.steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are created."""
        if logs is not None and 'loss' in logs:
            loss_value = logs['loss']
            self.losses.append(float(loss_value))
            self.steps.append(state.global_step)
            # Print progress every 100 steps
            if state.global_step % 100 == 0:
                print(f"  Step {state.global_step}: Loss = {loss_value:.4f}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called after each epoch."""
        if self.losses:
            avg_loss = sum(self.losses[-100:]) / min(100, len(self.losses))
            print(f"  Epoch {state.epoch} complete. Recent avg loss: {avg_loss:.4f}")


def train(
    strategy: str = 'in-batch',
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    output_dir: str = './output',
    num_examples: int = 1000,
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
    query_positives = build_query_positives_mapping(qrels, corpus_dict)
    
    # Initialize model first (needed for loss functions)
    # Cache location and offline mode are controlled via environment variables set in shell script:
    # HF_HOME, TRANSFORMERS_CACHE, SENTENCE_TRANSFORMERS_HOME, HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE
    # Don't use cache_folder parameter - let SentenceTransformer use environment variables automatically
    model = SentenceTransformer(model_name)
    
    # Use device utility to support CUDA, MPS, or CPU
    device = get_and_print_device()
    model = model.to(device)
    
    # Mine training examples using the specified negative sampling strategy
    train_examples = mine_negatives(
        strategy=strategy,
        topics=topics,
        query_positives=query_positives,
        corpus_dict=corpus_dict if strategy == 'random' else None,
        num_examples=num_examples
    )
    
    # Use MultipleNegativesRankingLoss (contrastive loss) for both strategies
    # This implements the InfoNCE contrastive learning objective
    loss = losses.MultipleNegativesRankingLoss(model)
    
    print(f"Prepared {len(train_examples)} training examples")
    
    # Prepare evaluation data (dev.small - has both topics and qrels)
    print(f"\nPreparing evaluation data (dev.small)...")
    queries_dict, eval_corpus, relevant_docs = prepare_evaluation_data(dataset, variant='dev.small')
    
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
    
    # Convert InputExamples to Dataset format for SentenceTransformerTrainer
    # InputExamples need to be converted to dict format
    if strategy == 'in-batch':
        # For pairs: (text1, text2)
        # InputExample.texts is always a list, but type checker may not know this
        train_data = {
            'text1': [ex.texts[0] for ex in train_examples if ex.texts and len(ex.texts) > 0],  # type: ignore[index]
            'text2': [ex.texts[1] for ex in train_examples if ex.texts and len(ex.texts) > 1]  # type: ignore[index]
        }
    else:  # random strategy
        # For triplets: (anchor, positive, negative)
        train_data = {
            'anchor': [ex.texts[0] for ex in train_examples if ex.texts and len(ex.texts) > 0],  # type: ignore[index]
            'positive': [ex.texts[1] for ex in train_examples if ex.texts and len(ex.texts) > 1],  # type: ignore[index]
            'negative': [ex.texts[2] for ex in train_examples if ex.texts and len(ex.texts) > 2]  # type: ignore[index]
        }
    
    train_dataset = Dataset.from_dict(train_data)
    
    # Create callback to track training loss
    loss_callback = LossTrackingCallback()
    
    # Create training arguments
    # Set logging_steps to a small value to capture loss during training
    # With small datasets, we want to log more frequently
    logging_steps = max(1, min(10, len(train_examples) // (batch_size * 4)))
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=logging_steps,
        save_strategy="epoch",
        # Don't set eval_strategy - InformationRetrievalEvaluator handles evaluation automatically
        # The evaluator will be called by SentenceTransformerTrainer at the end of each epoch
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
        args=training_args,
        callbacks=[loss_callback] 
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save(output_dir)
    
    # Plot training loss from callback
    print(f"\nCaptured {len(loss_callback.losses)} loss values from training")
    # Create list of (step, loss) tuples for plotting
    if loss_callback.steps:
        loss_data = list(zip(loss_callback.steps, loss_callback.losses))
    else:
        # Use indices as steps if steps weren't tracked
        loss_data = list(enumerate(loss_callback.losses))
    
    plot_path = os.path.join(output_dir, 'training_loss.png')
    plot_training_loss(loss_data, plot_path)
    
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
        default='sentence-transformers/all-MiniLM-L6-v2',
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
        default=1000,
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

