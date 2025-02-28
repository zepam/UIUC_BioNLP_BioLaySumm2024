import argparse
import logging
import os
import json
import random
from typing import Dict, List, Tuple
import evaluate
import torch
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune LED model for lay summary generation')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--encoder_max_length', type=int, default=8192, help='Maximum encoder sequence length')
    parser.add_argument('--decoder_max_length', type=int, default=512, help='Maximum decoder sequence length')
    parser.add_argument('--output_dir', type=str, default='./output_model', help='Output directory for model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_category_labels(file_path: str) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    categories = {
        'background': [], 'objective': [], 'methods': [],
        'results': [], 'conclusions': []
    }
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                components = line.strip().split('|')
                title, category, _, _ = components
                if category.lower() in categories:
                    categories[category.lower()].append(title.lower())
    except FileNotFoundError:
        logger.error(f"Could not find the file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise
    return (
        categories['background'], categories['objective'],
        categories['methods'], categories['results'],
        categories['conclusions']
    )

def read_jsonl_file(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r') as file:
            return [json.loads(line)['text'] for line in file]
    except FileNotFoundError:
        logger.error(f"Could not find the file: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
        raise

def load_json(filename: str) -> dict:
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"Could not find the file: {filename}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {filename}")
        raise

def process_texts(texts: List[str]) -> List[str]:
    return [s.replace(' . ', '. ').replace(' , ', ', ') for s in texts]

def load_data(dataset: str, datatype: str) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    data_folder = './biolaysumm2024_data'
    data_path = os.path.join(data_folder, f'{dataset}_{datatype}.jsonl')
    
    try:
        with open(data_path, 'r') as file:
            data = [json.loads(line) for line in file.readlines()]
            return (
                [item['article'] for item in data],
                [item['lay_summary'] for item in data],
                [item['keywords'] for item in data],
                [item['headings'] for item in data],
                [item['id'] for item in data]
            )
    except FileNotFoundError:
        logger.error(f"Could not find the file: {data_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {data_path}")
        raise

def main():
    args = parse_args()
    set_seed(args.seed)
    logger.info("Loading tokenizer and metric...")
    tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
    metric = evaluate.load("rouge")

    logger.info("Loading training and validation data...")
    article_train, lay_sum_train, keyword_train, headings_train, id_train = load_data('elife', 'train')
    article_val, lay_sum_val, keyword_val, headings_val, id_val = load_data('elife', 'val')

    logger.info("Loading category labels...")
    background, objective, methods, results, conclusions = load_category_labels(
        './Structured-Abstracts-Labels-102615.txt'
    )
    train_wiki = read_jsonl_file('./elife_train_abstract_wiki_retriever.jsonl')
    val_wiki = read_jsonl_file('./elife_val_abstract_wiki_retriever.jsonl')
    extract_train = load_json('./elife_train_extractive_sum.json')
    extract_val = load_json('./elife_val_extractive_sum.json')
    wiki_definitions_train = load_json('./elife_train_retrieval.json')
    wiki_definitions_val = load_json('./elife_val_retrieval.json')

    article_train = process_texts(article_train)
    article_val = process_texts(article_val)
    lay_sum_train = process_texts(lay_sum_train)
    lay_sum_val = process_texts(lay_sum_val)

    def process_article_with_sections(article, headings, wiki, extract, definitions):
        sections = article.split('\n')
        selected_sections = []
        other_sections = []
        
        for heading, section in zip(headings, sections):
            heading = heading.lower()
            if heading in background or heading in conclusions or heading == 'abstract':
                selected_sections.append(section)
            else:
                other_sections.append(section)

        selected_text = ' '.join(selected_sections)
        return f"{selected_text} {extract} {wiki} {definitions}"

    logger.info("Processing articles with sections...")
    wiki_article_train = [
        process_article_with_sections(art, head, wiki, ext, def_)
        for art, head, wiki, ext, def_ in zip(
            article_train, headings_train, train_wiki,
            extract_train, wiki_definitions_train
        )
    ]
    
    wiki_article_val = [
        process_article_with_sections(art, head, wiki, ext, def_)
        for art, head, wiki, ext, def_ in zip(
            article_val, headings_val, val_wiki,
            extract_val, wiki_definitions_val
        )
    ]

    def process_data_to_model_inputs(batch):
        inputs = tokenizer(
            batch["article"],
            padding="max_length",
            truncation=True,
            max_length=args.encoder_max_length,
        )
        outputs = tokenizer(
            batch["abstract"],
            padding="max_length",
            truncation=True,
            max_length=args.decoder_max_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["global_attention_mask"] = [[0] * len(ids) for ids in inputs.input_ids]
        for mask in batch["global_attention_mask"]:
            mask[0] = 1

        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in outputs.input_ids
        ]
        return batch

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        return metric.compute(predictions=pred_str, references=label_str)

    logger.info("Preparing datasets...")
    train_dataset = Dataset.from_dict({
        'article': wiki_article_train,
        'abstract': lay_sum_train
    }).map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=["article", "abstract"]
    )

    val_dataset = Dataset.from_dict({
        'article': wiki_article_val,
        'abstract': lay_sum_val
    }).map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=["article", "abstract"]
    )

    columns = ["input_ids", "attention_mask", "global_attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns)
    val_dataset.set_format(type="torch", columns=columns)

    logger.info("Initializing model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "patrickvonplaten/led-large-16384-pubmed",
        gradient_checkpointing=True,
        use_cache=False
    )
    
    model.config.num_beams = 2
    model.config.max_length = args.decoder_max_length
    model.config.min_length = 100
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=True,
        output_dir=args.output_dir,
        logging_steps=5,
        eval_steps=250,
        save_steps=250,
        save_total_limit=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,
        load_best_model_at_end=True,
    )

    logger.info("Initializing trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
