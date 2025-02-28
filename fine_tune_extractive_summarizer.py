import argparse
import csv
import logging
import math
import os
import random
from typing import List, Tuple
import torch
from sentence_transformers import (SentenceTransformer, SentencesDataset, InputExample, losses)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> List[List[str]]:
    try:
        data = []
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                data.append(row)
        random.shuffle(data)
        return data
    except FileNotFoundError:
        logger.error(f"Could not find data file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_examples(data: List[List[str]]) -> List[InputExample]:
    examples = []
    for row in data:
        score = int(row[2])
        examples.append(InputExample(texts=[str(row[0]), str(row[1])], label=score))
    return examples

def train_model(
    train_path: str,
    val_path: str,
    model_name: str = 'NeuML/pubmedbert-base-embeddings',
    batch_size: int = 32,
    epochs: int = 5,
    output_path: str,
    device: str = None,
    evaluation_steps: int = None
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading data from {train_path} and {val_path}")
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    
    logger.info(f"Initializing model {model_name} on {device}")
    model = SentenceTransformer(model_name).to(device)
    
    train_examples = create_examples(train_data)
    val_examples = create_examples(val_data)
    
    logger.info(f"Training samples: {len(train_examples)}")
    logger.info(f"Validation samples: {len(val_examples)}")
    
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
    train_loss = losses.ContrastiveLoss(model=model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='dev')

    warmup_steps = math.ceil(len(train_dataset) * epochs / batch_size * 0.1)
    if evaluation_steps is None:
        evaluation_steps = len(train_dataloader)
    
    logger.info(f"Starting training with {epochs} epochs")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Evaluation steps: {evaluation_steps}")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        save_best_model=True
    )
    
    logger.info(f"Model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a PubMedBERT model for extractive summarization"
    )
    parser.add_argument(
        "--train_data",
        required=True,
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--val_data",
        required=True,
        help="Path to validation data CSV file"
    )
    parser.add_argument(
        "--model_name",
        default="NeuML/pubmedbert-base-embeddings"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="training epochs"
    )
    parser.add_argument(
        "--output_path",
        default="./fine_tuned_extractive_summarization_model",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--device",
        help="Device to use for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    
    train_model(
        train_path=args.train_data,
        val_path=args.val_data,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_path=args.output_path,
        device=args.device
    )

if __name__ == "__main__":
    main()
