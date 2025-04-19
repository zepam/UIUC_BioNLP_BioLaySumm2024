import argparse
import json
import logging
import os
import re
import time
from typing import Dict, List, Tuple
import torch

from transformers import AutoConfig, AutoTokenizer, AutoModel
from summarizer import Summarizer
from langchain_text_splitters import NLTKTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SectionCategories:
    BACKGROUND = 'background'
    OBJECTIVE = 'objective'
    METHODS = 'methods'
    RESULTS = 'results'
    CONCLUSIONS = 'conclusions'
    ABSTRACT = 'abstract'

def load_section_labels(labels_file: str) -> Dict[str, List[str]]:
    categories = {
        'background': [],
        'objective': [],
        'methods': [],
        'results': [],
        'conclusions': []
    }
    
    try:
        with open(labels_file, 'r') as file:
            for line in file:
                components = line.strip().split('|')
                title, category = components[0:2]
                category = category.lower()
                if category in categories:
                    categories[category].append(title.lower())
        
        return categories
    except FileNotFoundError:
        logger.error(f"Could not find labels file: {labels_file}")
        raise
    except Exception as e:
        logger.error(f"Error loading labels: {str(e)}")
        raise

def load_dataset(data_folder: str, dataset: str, split: str, is_test: bool = False) -> Tuple:
    data_path = os.path.join(data_folder, f'{dataset}_{split}.jsonl')
    articles = []
    headings = []
    ids = []
    lay_summaries = [] if not is_test else None
    
    try:
        with open(data_path, 'r') as file:
            for line in file.readlines():
                data = json.loads(line)
                articles.append(data['article'])
                headings.append(data['headings'])
                ids.append(data['id'])
                if not is_test:
                    lay_summaries.append(data['lay_summary'])
    
        if is_test:
            return articles, headings, ids
        return articles, lay_summaries, headings, ids
    
    except FileNotFoundError:
        logger.error(f"Could not find dataset file: {data_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in dataset file: {data_path}")
        raise

def preprocess_text(text: str, dataset: str) -> str:
    text = text.replace(' . ', '. ').replace(' , ', ', ')
    if dataset == 'PLOS':
        text = re.sub(r'\s\[.*?\]', '', text)
    elif dataset == 'eLife':
        text = re.sub(r'(\(\s([^()]*\s\,\s)*[^()]*\s\))', '', text)
    
    return text

def process_sections(
    article: str,
    headings: List[str],
    section_categories: Dict[str, List[str]]
) -> Tuple[str, str]:
    sections = article.split('\n')
    selected_sections = []
    non_selected_sections = []
    
    for heading, section in zip(headings, sections):
        heading = heading.lower()
        if (heading in section_categories['background'] or 
            heading in section_categories['conclusions'] or 
            heading == SectionCategories.ABSTRACT):
            selected_sections.append(section)
        elif (heading in section_categories['methods'] or 
              heading in section_categories['results']):
            non_selected_sections.append(section)
        else:
            non_selected_sections.append(section)
    
    return (''.join(non_selected_sections), ''.join(selected_sections))

def setup_summarizer(model_path: str) -> Summarizer:
    config = AutoConfig.from_pretrained(model_path)
    config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, config=config)
    return Summarizer(custom_model=model, custom_tokenizer=tokenizer)

def main():

    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Generate extractive summaries for BioLaySumm datasets"
    )
    parser.add_argument(
        "--data_folder",
        required=True,
        help="Path to the data folder containing dataset files"
    )
    parser.add_argument(
        "--dataset",
        choices=['PLOS', 'eLife'],
        required=True,
        help="Dataset to process (PLOS or eLife)"
    )
    parser.add_argument(
        "--split",
        choices=['train', 'val', 'test'],
        required=True,
        help="Data split to process"
    )
    parser.add_argument(
        "--labels_file",
        required=True,
        help="Path to section labels file"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the custom summarization model"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=600,
        help="Size of text chunks for processing"
    )
    parser.add_argument(
        "--num_sentences",
        type=int,
        default=50,
        help="Number of sentences to include in extractive summary"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum length for each summary sentence"
    )
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Loading section labels...")
    section_categories = load_section_labels(args.labels_file)
  
    text_splitter = NLTKTextSplitter(chunk_size=args.chunk_size)
    summarizer = setup_summarizer(args.model_path)
    
    logger.info(f"Processing {args.dataset} {args.split} split...")
    is_test = args.split == 'test'
    data = load_dataset(args.data_folder, args.dataset, args.split, is_test)
    
    if is_test:
        articles, headings, ids = data
    else:
        articles, lay_summaries, headings, ids = data

    extractive_summaries = []
    
    for i, (article, heading_list) in enumerate(zip(articles, headings)):
        logger.info(f"Processing article {i+1}/{len(articles)}")
      
        article = preprocess_text(article, args.dataset)
        non_selected, selected = process_sections(
            article, heading_list, section_categories
        )
        
        if non_selected:
            chunks = text_splitter.split_text(non_selected)
            chunks = [chunk.replace('\n\n', ' ') for chunk in chunks]
            text = ' '.join(chunks)
            
            summary = summarizer(
                text,
                num_sentences=args.num_sentences,
                max_length=args.max_length
            )
            summary_text = ''.join(summary)

        if 'summary_text' in locals() and summary_text:
            extractive_summaries.append(summary_text)
    
    logger.info("Saving results...")
    base_filename = f"{args.dataset.lower()}_{args.split}"

    with open(os.path.join(args.output_dir, f"{base_filename}_extractive_summaries.json"), 'w') as f:
        json.dump(extractive_summaries, f)

    end_time = time.time()
    logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()  
