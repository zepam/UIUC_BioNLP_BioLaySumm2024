import argparse
import logging
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

from utils import (
    set_seed,
    load_category_labels,
    read_jsonl_file,
    load_json,
    process_texts,
    load_data,
    process_article_with_sections,
    create_data_processor,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune LED model for PLOS')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--encoder_max_length', type=int, default=8192, help='Maximum encoder sequence length')
    parser.add_argument('--decoder_max_length', type=int, default=512, help='Maximum decoder sequence length')
    parser.add_argument('--output_dir', type=str, default='./plos_output_model', help='Output directory for model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    return metric.compute(predictions=pred_str, references=label_str)

def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    
    logger.info("Loading tokenizer and metric...")
    tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
    metric = evaluate.load("rouge")
    

    logger.info("Loading training and validation data...")
    article_train, lay_sum_train, keyword_train, headings_train, id_train = load_data('PLOS', 'train')
    article_val, lay_sum_val, keyword_val, headings_val, id_val = load_data('PLOS', 'val')

    logger.info("Loading category labels...")
    background, objective, methods, results, conclusions = load_category_labels(
        './Structured-Abstracts-Labels-102615.txt'
    )
    logger.info("Loading additional resources...")
    train_wiki = read_jsonl_file('./plos_train_abstract_wiki_retriever.jsonl')
    val_wiki = read_jsonl_file('./plos_val_abstract_wiki_retriever.jsonl')
    extract_train = load_json('./plos_train_extractive_sum.json')
    extract_val = load_json('./plos_val_extractive_sum.json')
    wiki_definitions_train = load_json('./plos_train_retrieval.json')
    wiki_definitions_val = load_json('./plos_val_retrieval.json')

    article_train = process_texts(article_train)
    article_val = process_texts(article_val)
    lay_sum_train = process_texts(lay_sum_train)
    lay_sum_val = process_texts(lay_sum_val)

    logger.info("Processing articles with sections...")
    wiki_article_train = [
        process_article_with_sections(
            art, head, wiki, ext, def_, background, conclusions
        )
        for art, head, wiki, ext, def_ in zip(
            article_train, headings_train, train_wiki,
            extract_train, wiki_definitions_train
        )
    ]
    
    wiki_article_val = [
        process_article_with_sections(
            art, head, wiki, ext, def_, background, conclusions
        )
        for art, head, wiki, ext, def_ in zip(
            article_val, headings_val, val_wiki,
            extract_val, wiki_definitions_val
        )
    ]

    process_data_to_model_inputs = create_data_processor(
        tokenizer, args.encoder_max_length, args.decoder_max_length
    )

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

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
