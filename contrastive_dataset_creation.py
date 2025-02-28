import os
import json
import csv
import re
import random
import argparse
from typing import List, Tuple, Dict
from pathlib import Path

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import NLTKTextSplitter



class ContrastiveDatasetCreator:
    def __init__(self, args):
        self.args = args
        self.device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._load_model()
        self.text_splitter = NLTKTextSplitter(chunk_size=args.chunk_size)
        random.seed(args.random_seed)
        
    def _load_model(self) -> SentenceTransformer:
        try:
            model = SentenceTransformer(self.args.model).to(self.device)
            print(f"Model loaded successfully on {self.device}")
            return model
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            raise

    def load_data(self, dataset: str, datatype: str) -> Tuple:
        data_folder = Path('./biolaysumm2024_data')
        data_path = data_folder / f'{dataset}_{datatype}.jsonl'
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        try:
            with open(data_path, 'r') as file:
                data = [json.loads(line) for line in file]
                
            article = [d['article'] for d in data]
            keyword = [d['keywords'] for d in data]
            headings = [d['headings'] for d in data]
            id_list = [d['id'] for d in data]
            lay_sum = [d.get('lay_summary', '') for d in data]
            return article, lay_sum, keyword, headings, id_list
        except Exception as e:
            print(f"Error loading data from {data_path}: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        text = text.replace(' . ', '. ').replace(' , ', ', ')
        text = text.replace('\n\n', ' ')
        # remove citations and references
        text = re.sub(r'\s\[.*?\]', "", text)
        text = re.sub(r'(\(\s([^()]*\s\,\s)*[^()]*\s\))', "", text)
        return text

    def chunk_text(self, texts: List[str]) -> List[List[str]]:
        chunked_texts = []
        for text in tqdm(texts, desc="Chunking texts"):
            text = self.preprocess_text(text)
            chunks = self.text_splitter.split_text(text)
            chunked_texts.append(chunks)
        return chunked_texts

    def create_pairs(self, lay_sums: List[str], docs: List[str]) -> Tuple[List[Tuple], List[Tuple]]:
        positive_pairs = []
        negative_pairs = []
        
        for lay_sum, doc in tqdm(zip(lay_sums, docs), desc="Creating pairs"):
            doc_embeddings = self.model.encode(doc, convert_to_tensor=True)
            lay_embeddings = self.model.encode(lay_sum, convert_to_tensor=True)
            cosine_scores = util.cos_sim(lay_embeddings, doc_embeddings)
            for i in range(cosine_scores.shape[0]):
                for j in range(cosine_scores.shape[1]):
                    if cosine_scores[i][j] >= self.args.pos_threshold:
                        positive_pairs.append((lay_sum[i], doc[j], 1))
                    elif cosine_scores[i][j] <= self.args.neg_threshold:
                        negative_pairs.append((lay_sum[i], doc[j], 0))
        
        return positive_pairs, negative_pairs

    def save_pairs(self, pairs: List[Tuple], filename: str):
        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['lay_sum', 'sentence', 'label'])
                for row in pairs:
                    writer.writerow(row)
            print(f"Successfully saved pairs to {filename}")
        except Exception as e:
            print(f"Error saving pairs to {filename}: {str(e)}")
            raise

    def process_dataset(self, dataset: str, datatype: str):
        logger.info(f"Processing {dataset} {datatype} dataset")
        article, lay_sum, _, _, _ = self.load_data(dataset, datatype)
        chunked_articles = self.chunk_text(article)
        chunked_lay_sums = self.chunk_text(lay_sum)
        positive_pairs, negative_pairs = self.create_pairs(chunked_lay_sums, chunked_articles)
        if len(negative_pairs) > len(positive_pairs):
            negative_pairs = random.sample(negative_pairs, len(positive_pairs))
        all_pairs = positive_pairs + negative_pairs
        output_file = f'./{dataset.lower()}_{datatype}_sentence_level_positive_negative_pairs.csv'
        self.save_pairs(all_pairs, output_file)
        
def main():
    parser = argparse.ArgumentParser(description="Create contrastive datasets from biomedical articles")
    parser.add_argument('--device', type=str, default=None, help="Device to run on (cuda:N or cpu)")
    parser.add_argument('--chunk-size', type=int, default=600, help="Size of text chunks")
    parser.add_argument('--pos-threshold', type=float, default=0.9, help="Threshold for positive pairs")
    parser.add_argument('--neg-threshold', type=float, default=0.01, help="Threshold for negative pairs")
    parser.add_argument('--model', type=str, default="NeuML/pubmedbert-base-embeddings",
                      help="Sentence transformer model to use")
    parser.add_argument('--random-seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    try:
        creator = ContrastiveDatasetCreator(args)
        
        creator.process_dataset("PLOS", "train")
        creator.process_dataset("PLOS", "val")

        creator.process_dataset("eLife", "train")
        creator.process_dataset("eLife", "val")
        
        print("Successfully completed all dataset processing")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
