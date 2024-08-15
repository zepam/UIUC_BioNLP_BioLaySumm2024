import torch
from datasets import Dataset
from transformers import LEDTokenizer, LEDForConditionalGeneration
from utils import *
from transformers import AutoTokenizer, LongT5ForConditionalGeneration, AutoModelForSeq2SeqLM
from peft import PeftConfig, PeftModelForSeq2SeqLM, PeftModel
from transformers import GenerationConfig
import random
import re
import json
import os

print("Evaluation Datasets Generation")
random.seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pattern = r'\s\[.*?\]'

background = []
objective = []
methods = []
results = []
conclusions = []

with open('/ocean/projects/cis230089p/zyou2/Structured-Abstracts-Labels-102615.txt', 'r') as file:
    for line in file:
        components = line.strip().split('|')
        title, category, _, _ = components
        if category == 'BACKGROUND':
            background.append(title)
        elif category == 'OBJECTIVE':
            objective.append(title)
        elif category == 'METHODS':
            methods.append(title)
        elif category == 'RESULTS':
            results.append(title)
        elif category == 'CONCLUSIONS':
            conclusions.append(title)
            
background = [item.lower() for item in background]
objective = [item.lower() for item in objective]
methods = [item.lower() for item in methods]
results = [item.lower() for item in results]
conclusions = [item.lower() for item in conclusions]

def read_jsonl_file(file_path):
    texts = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            texts.append(json_obj['text'])
    return texts

# Load JSON files
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Process articles
def process_articles(articles, pattern):
    return [re.sub(pattern, '', s.replace(' . ', '. ').replace(' , ', ', ')) for s in articles]

### Load Your Fine-Tuned LED Checkpoint
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_led_model/checkpoint-1500/")
model = LEDForConditionalGeneration.from_pretrained("./fine_tuned_led_model/checkpoint-1500/").to(device)

def generate_sum(batch):
    inputs_dict = tokenizer(batch["article"], padding="max_length", max_length=8192, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to(device)
    attention_mask = inputs_dict.attention_mask.to(device)
    global_attention_mask = torch.zeros_like(attention_mask)
    global_attention_mask[:, 0] = 1
    predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
    return batch

def load_data(dataset, datatype):
    data_folder = './biolaysumm2024_data'
    data_path = os.path.join(data_folder, f'{dataset}_{datatype}.jsonl')
    lay_sum = []
    article =[]
    keyword = []
    headings = []
    id = []
    file = open(data_path, 'r')
    for line in (file.readlines()):
        dic = json.loads(line)
        article.append(dic['article'])
        keyword.append(dic['keywords'])
        headings.append(dic['headings'])
        id.append(dic['id'])
        lay_sum.append(dic['lay_summary'])
    
    return article, lay_sum, keyword, headings, id

### PLOS 
## val
plos_article_val, plos_lay_sum_val, plos_keyword_val, plos_headings_val, plos_id_val = load_data('PLOS', 'val')

### eLife 
## val
elife_article_val, elife_lay_sum_val, elife_keyword_val, elife_headings_val, elife_id_val = load_data('eLife', 'val')

### Load RAG DPR Retrieval Val Results
val_path = './plos_val_abstract_wiki_retriever.jsonl'
val_plos_wiki = read_jsonl_file(val_path)
val_path = './elife_val_abstract_wiki_retriever.jsonl'
val_elife_wiki = read_jsonl_file(val_path)

### Load Extractive Summarization and Wiki Definition Retrieval Val Results
plos_extract_val = load_json('./plos_val_extractive_sum.json')
elife_extract_val = load_json('./elife_val_extractive_sum.json')
plos_wiki_definitions_val = load_json('./plos_val_retrieval.json')
elife_wiki_definitions_val = load_json('./elife_val_retrieval.json')

new_plos_article_val = process_articles(plos_article_val, pattern)
new_elife_article_val = process_articles(elife_article_val, pattern)

final_plos_article_val = []
for article, headings, wiki, extract, definitions in zip(new_plos_article_val, plos_headings_val, val_plos_wiki, plos_extract_val, plos_wiki_definitions_val):
    sections = article.split('\n')
    temp_selected_sections = []
    temp_sections = []
    temp_retrieval = []
    for i, (heading, section) in enumerate(zip(headings, sections)):
        heading = heading.lower()
        if heading in background:
            temp_selected_sections.append(section)
        elif heading in methods:
            temp_sections.append(section)
        elif heading in conclusions:
            temp_selected_sections.append(section)
        elif heading in results:
            temp_sections.append(section)
        elif heading in 'abstract':
            temp_selected_sections.append(section)
        else:
            temp_sections.append(section)

    final_string = ''.join(temp_sections)
    final_selected_string = ''.join(temp_selected_sections)
    final_selected_string = final_selected_string + ' ' + extract + ' ' + wiki + ' ' + definitions
    final_plos_article_val.append(final_selected_string)

final_elife_article_val = []
for article, headings, wiki, extract, definitions in zip(new_elife_article_val, elife_headings_val, val_elife_wiki, elife_extract_val, elife_wiki_definitions_val):
    sections = article.split('\n')
    temp_selected_sections = []
    temp_sections = []
    temp_retrieval = []
    for i, (heading, section) in enumerate(zip(headings, sections)):
        heading = heading.lower()
        if heading in background:
            temp_selected_sections.append(section)
        elif heading in methods:
            temp_sections.append(section)
        elif heading in conclusions:
            temp_selected_sections.append(section)
        elif heading in results:
            temp_sections.append(section)
        elif heading in 'abstract':
            temp_selected_sections.append(section)
        else:
            temp_sections.append(section)

    final_string = ''.join(temp_sections)
    final_selected_string = ''.join(temp_selected_sections)
    final_selected_string = final_selected_string + ' ' + extract + ' ' + wiki + ' ' + definitions
    final_elife_article_val.append(final_selected_string)

# # val
plos_val_dataset = {'article': final_plos_article_val}
plos_val_dataset = Dataset.from_dict(plos_val_dataset)
final_plos_val_result = plos_val_dataset.map(generate_sum, batched=True, batch_size=4)
plos_predicted_val_abstract = final_plos_val_result["predicted_abstract"]

elife_val_dataset = {'article': final_elife_article_val}
elife_val_dataset = Dataset.from_dict(elife_val_dataset)
final_elife_val_result = elife_val_dataset.map(generate_sum, batched=True, batch_size=4)
elife_predicted_val_abstract = final_elife_val_result["predicted_abstract"]

### output val
with open('./plos_val.txt', 'w') as file:
    for abstract in plos_predicted_val_abstract:
        file.write(abstract + '\n')

### output test
with open('./elife_val.txt', 'w') as file:
    for abstract in elife_predicted_val_abstract:
        file.write(abstract + '\n')
        
print("finished writing")
