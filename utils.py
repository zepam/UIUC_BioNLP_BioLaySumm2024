import json
import logging
import os
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def set_seed(seed: int):
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

def process_article_with_sections(article: str, headings: List[str], wiki: str, 
                                extract: str, definitions: str,
                                background: List[str], conclusions: List[str]) -> str:
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

def create_data_processor(tokenizer: AutoTokenizer, encoder_max_length: int, decoder_max_length: int):
    def process_data_to_model_inputs(batch):
        inputs = tokenizer(
            batch["article"],
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
        )
        outputs = tokenizer(
            batch["abstract"],
            padding="max_length",
            truncation=True,
            max_length=decoder_max_length,
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
    
    return process_data_to_model_inputs 
