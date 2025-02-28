import os
import torch
import json
import re
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    LEDTokenizer,
    LEDForConditionalGeneration,
    AutoTokenizer,
    LongT5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    GenerationConfig
)
from peft import PeftConfig, PeftModelForSeq2SeqLM, PeftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    model_checkpoint: str = "./fine_tuned_model/"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    max_length: int = 8192
    random_seed: int = 42
    data_folder: str = "./biolaysumm2024_data"
    structured_abstracts_file: str = "./Structured-Abstracts-Labels-102615.txt"

class BiomedicalEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        random.seed(config.random_seed)
        self.tokenizer = None
        self.model = None
        self.categories = self._load_categories()
        logger.info(f"Initialized evaluator with device: {self.device}")

    def _load_categories(self) -> Dict[str, List[str]]:
        categories = {
            'background': [], 'objective': [], 
            'methods': [], 'results': [], 'conclusions': []
        }
        
        try:
            with open(self.config.structured_abstracts_file, 'r') as file:
                for line in file:
                    components = line.strip().split('|')
                    title, category, _, _ = components
                    if category.lower() in categories:
                        categories[category.lower()].append(title.lower())
            
            logger.info("Successfully loaded abstract categories")
            return categories
            
        except Exception as e:
            logger.error(f"Error loading categories: {str(e)}")
            raise

    def setup_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint)
            self.model = LEDForConditionalGeneration.from_pretrained(
                self.config.model_checkpoint
            ).to(self.device)
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def load_data(self, dataset: str, datatype: str) -> Tuple:
        data_path = Path(self.config.data_folder) / f'{dataset}_{datatype}.jsonl'
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        try:
            with open(data_path, 'r') as file:
                data = [json.loads(line) for line in file]
            article = [d['article'] for d in data]
            lay_sum = [d.get('lay_summary', '') for d in data]
            keyword = [d['keywords'] for d in data]
            headings = [d['headings'] for d in data]
            id_list = [d['id'] for d in data]
            logger.info(f"Successfully loaded {dataset} {datatype} dataset")
            return article, lay_sum, keyword, headings, id_list
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def read_jsonl_file(self, file_path: str) -> List[str]:
        try:
            with open(file_path, 'r') as file:
                return [json.loads(line)['text'] for line in file]
        except Exception as e:
            logger.error(f"Error reading JSONL file {file_path}: {str(e)}")
            raise

    def load_json(self, filename: str) -> Dict:
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error loading JSON file {filename}: {str(e)}")
            raise

    def process_articles(self, articles: List[str], pattern: str) -> List[str]:
        return [
            re.sub(pattern, '', s.replace(' . ', '. ').replace(' , ', ', '))
            for s in articles
        ]

    def generate_summary(self, batch: Dict) -> Dict:
        inputs_dict = self.tokenizer(
            batch["article"],
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt",
            truncation=True
        )
        
        input_ids = inputs_dict.input_ids.to(self.device)
        attention_mask = inputs_dict.attention_mask.to(self.device)
        
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1
        
        predicted_abstract_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        
        batch["predicted_abstract"] = self.tokenizer.batch_decode(
            predicted_abstract_ids,
            skip_special_tokens=True
        )
        return batch

    def process_dataset_articles(
        self,
        articles: List[str],
        headings: List[List[str]],
        wiki_data: List[str],
        extracts: List[str],
        definitions: List[str]
    ) -> List[str]:
        final_articles = []
        
        for article, article_headings, wiki, extract, defs in tqdm(
            zip(articles, headings, wiki_data, extracts, definitions),
            desc="Processing articles"
        ):
            sections = article.split('\n')
            selected_sections = []
            other_sections = []
            
            for heading, section in zip(article_headings, sections):
                heading = heading.lower()
                if (heading in self.categories['background'] or
                    heading in self.categories['conclusions'] or
                    heading == 'abstract'):
                    selected_sections.append(section)
                else:
                    other_sections.append(section)

            final_selected_string = ' '.join(selected_sections)
            final_selected_string = f"{final_selected_string} {extract} {wiki} {defs}"
            final_articles.append(final_selected_string)
            
        return final_articles

    def evaluate_dataset(self, dataset_name: str):
        logger.info(f"Starting evaluation for {dataset_name}")
        articles, lay_sums, keywords, headings, ids = self.load_data(dataset_name, 'val')
        wiki_data = self.read_jsonl_file(f'./{dataset_name.lower()}_val_abstract_wiki_retriever.jsonl')
        extracts = self.load_json(f'./{dataset_name.lower()}_val_extractive_sum.json')
        definitions = self.load_json(f'./{dataset_name.lower()}_val_retrieval.json')
        
        pattern = r'\s\[.*?\]'
        processed_articles = self.process_articles(articles, pattern)
        final_articles = self.process_dataset_articles(
            processed_articles, headings, wiki_data, extracts, definitions
        )

        dataset = Dataset.from_dict({'article': final_articles})
        results = dataset.map(
            self.generate_summary,
            batched=True,
            batch_size=self.config.batch_size
        )

        output_file = f'./{dataset_name.lower()}_val.txt'
        try:
            with open(output_file, 'w') as file:
                for abstract in results["predicted_abstract"]:
                    file.write(abstract + '\n')
            logger.info(f"Successfully saved predictions to {output_file}")
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            raise

def main():
    config = EvaluationConfig()
    try:
        evaluator = BiomedicalEvaluator(config)
        evaluator.setup_model()
    
        evaluator.evaluate_dataset("PLOS")

        evaluator.evaluate_dataset("eLife")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
