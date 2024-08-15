from utils import *

from transformers import(
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import Dataset
import os
import json
import random
import evaluate
import json
import re

random.seed(42)
metric = evaluate.load("rouge")
print("Fine-tuning on elife dataset")

### Params
tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
encoder_max_length = 8192
decoder_max_length = 512
batch_size = 1 

background = []
objective = []
methods = []
results = []
conclusions = []

with open('./Structured-Abstracts-Labels-102615.txt', 'r') as file:
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

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def process_texts(texts):
    return [s.replace(' . ', '. ').replace(' , ', ', ') for s in texts]

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
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]
    return batch

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    rouge_output = metric.compute(predictions=pred_str, references=label_str)
    return rouge_output

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

def load_test_data(dataset, datatype):
    data_folder = './biolaysumm2024_data'
    data_path = os.path.join(data_folder, f'{dataset}_{datatype}.jsonl')
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
    return article, keyword, headings, id

### elife
# train
elife_article_train, elife_lay_sum_train, elife_keyword_train, elife_headings_train, elife_id_train = load_data('elife', 'train')
# val
elife_article_val, elife_lay_sum_val, elife_keyword_val, elife_headings_val, elife_id_val = load_data('elife', 'val')

# add functional modules' outputs
trian_path = './elife_train_abstract_wiki_retriever.jsonl'
train_elife_wiki = read_jsonl_file(trian_path)
val_path = './elife_val_abstract_wiki_retriever.jsonl'
val_elife_wiki = read_jsonl_file(val_path)

extract_train = load_json('./elife_train_extractive_sum.json')
extract_val = load_json('./elife_val_extractive_sum.json')

wiki_definitions_train = load_json('./elife_train_retrieval.json')
wiki_definitions_val = load_json('./elife_val_retrieval.json')

new_elife_article_train = process_texts(elife_article_train)
new_elife_article_val = process_texts(elife_article_val)
new_elife_lay_sum_train = process_texts(elife_lay_sum_train)
new_elife_lay_sum_val = process_texts(elife_lay_sum_val)

wiki_elife_article_train = []
for article, headings, wiki, extract, definitions in zip(new_elife_article_train, elife_headings_train, train_elife_wiki, extract_train, wiki_definitions_train):
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
    wiki_elife_article_train.append(final_selected_string)
    
wiki_elife_article_val = []
for article, headings, wiki, extract, definitions in zip(new_elife_article_val, elife_headings_val, val_elife_wiki, extract_val, wiki_definitions_val):
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
    wiki_elife_article_val.append(final_selected_string)

# train
elife_train_dataset = {'article': wiki_elife_article_train, 'abstract': new_elife_lay_sum_train}
elife_train_dataset = Dataset.from_dict(elife_train_dataset)
# val
elife_val_dataset = {'article': wiki_elife_article_val, 'abstract': new_elife_lay_sum_val}
elife_val_dataset = Dataset.from_dict(elife_val_dataset)

train_dataset = elife_train_dataset.map(
    process_data_to_model_inputs,
    batched = True,
    batch_size = batch_size,
    remove_columns=["article", "abstract"]
)

val_dataset = elife_val_dataset.map(
    process_data_to_model_inputs,
    batched = True,
    batch_size = batch_size,
    remove_columns=["article", "abstract"]
)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

led_model = AutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/led-large-16384-pubmed", gradient_checkpointing=True, use_cache=False)
led_model.config.num_beams = 2
led_model.config.max_length = 512
led_model.config.min_length = 100
led_model.config.length_penalty = 2.0
led_model.config.early_stopping = True
led_model.config.no_repeat_ngram_size = 3

# Training
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    output_dir="./output_model",
    logging_steps=5,
    eval_steps=250,
    save_steps=250,
    save_total_limit=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    load_best_model_at_end=True,
)

print("start training...")

trainer = Seq2SeqTrainer(
    model=led_model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()