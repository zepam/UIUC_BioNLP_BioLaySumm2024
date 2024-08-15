import os
import json
from sentence_transformers import SentenceTransformer, util
import torch
import csv
import random
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import *
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util
import torch
import csv
import re
from langchain_text_splitters import NLTKTextSplitter
import random

# set random seed
random_seed = 42
random.seed(random_seed)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer("NeuML/pubmedbert-base-embeddings").to(device)

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

background = []
objective = []
methods = []
results = []
conclusions = []

with open('./data/Structured-Abstracts-Labels-102615.txt', 'r') as file:
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

### PLOS
# train
plos_article_train, plos_lay_sum_train, plos_keyword_train, plos_headings_train, plos_id_train = load_data('PLOS', 'train')
# val
plos_article_val, plos_lay_sum_val, plos_keyword_val, plos_headings_val, plos_id_val = load_data('PLOS', 'val')

### eLife
# train
elife_article_train, elife_lay_sum_train, elife_keyword_train, elife_headings_train, elife_id_train = load_data('eLife', 'train')
# val
elife_article_val, elife_lay_sum_val, elife_keyword_val, elife_headings_val, elife_id_val = load_data('eLife', 'val')


### chunk articles and lay summs for PLOS 

pattern = r'\s\[.*?\]'
text_splitter = NLTKTextSplitter(chunk_size=600)

### train
new_plos_article_train = []
for s in plos_article_train:
    new_s = s.replace(' . ', '. ')
    new_s = new_s.replace(' , ', ', ')
    new_plos_article_train.append(new_s)

chunked_plos_article_train = []
for article in new_plos_article_train:
    texts = text_splitter.split_text(article)
    new_texts = []
    for t in texts:
        t = t.replace('\n\n', ' ')
        ### remove the irrelevant citations and references
        result = re.sub(pattern, "", t)
        result = result.replace(' , ', ', ')
        new_texts.append(result)
    chunked_plos_article_train.append(new_texts)
    
### val
new_plos_article_val = []
for s in plos_article_val:
    new_s = s.replace(' . ', '. ')
    new_s = new_s.replace(' , ', ', ')
    new_plos_article_val.append(new_s)

chunked_plos_article_val = []  # 100 tokens per chunk
for article in new_plos_article_val:
    texts = text_splitter.split_text(article)
    new_texts = []
    for t in texts:
        t = t.replace('\n\n', ' ')
        result = re.sub(pattern, "", t)
        result = result.replace(' , ', ', ')
        new_texts.append(result)
    chunked_plos_article_val.append(new_texts)


### train lay sum
new_plos_lay_sum_train = []
for s in plos_lay_sum_train:
    new_s = s.replace(' . ', '. ')
    new_s = new_s.replace(' , ', ', ')
    new_plos_lay_sum_train.append(new_s)

chunked_plos_lay_sum_train = []
for sum in new_plos_lay_sum_train:
    texts = text_splitter.split_text(sum)
    new_texts = []
    for t in texts:
        t = t.replace('\n\n', ' ')
        new_texts.append(t)
    chunked_plos_lay_sum_train.append(new_texts)

### val lay sum
new_plos_lay_sum_val = []
for s in plos_lay_sum_val:
    new_s = s.replace(' . ', '. ')
    new_s = new_s.replace(' , ', ', ')
    new_plos_lay_sum_val.append(new_s)

chunked_plos_lay_sum_val = []
for sum in new_plos_lay_sum_val:
    texts = text_splitter.split_text(sum)
    new_texts = []
    for t in texts:
        t = t.replace('\n\n', ' ')
        new_texts.append(t)
    chunked_plos_lay_sum_val.append(new_texts)

### chunk articles and lay summs for eLife

pattern = r'(\(\s([^()]*\s\,\s)*[^()]*\s\))'
text_splitter = NLTKTextSplitter(chunk_size=600)

### train
new_elife_article_train = []
for s in elife_article_train:
    new_s = s.replace(' . ', '. ')
    new_s = new_s.replace(' , ', ', ')
    new_elife_article_train.append(new_s)

chunked_elife_article_train = []
for article in new_elife_article_train:
    texts = text_splitter.split_text(article)
    new_texts = []
    for t in texts:
        t = t.replace('\n\n', ' ')
        ### remove the irrelevant citations and references
        result = re.sub(pattern, "", t)
        result = result.replace(' , ', ', ')
        new_texts.append(result)
    chunked_elife_article_train.append(new_texts)
    
### val
new_elife_article_val = []
for s in elife_article_val:
    new_s = s.replace(' . ', '. ')
    new_s = new_s.replace(' , ', ', ')
    new_elife_article_val.append(new_s)

chunked_elife_article_val = []
for article in new_elife_article_val:
    texts = text_splitter.split_text(article)
    new_texts = []
    for t in texts:
        t = t.replace('\n\n', ' ')
        ### remove the irrelevant citations and references
        result = re.sub(pattern, "", t)
        result = result.replace(' , ', ', ')
        new_texts.append(result)
    chunked_elife_article_val.append(new_texts)

### train lay sum
new_elife_lay_sum_train = []
for s in elife_lay_sum_train:
    new_s = s.replace(' . ', '. ')
    new_s = new_s.replace(' , ', ', ')
    new_elife_lay_sum_train.append(new_s)

chunked_elife_lay_sum_train = []
for sum in new_elife_lay_sum_train:
    texts = text_splitter.split_text(sum)
    new_texts = []
    for t in texts:
        t = t.replace('\n\n', ' ')
        new_texts.append(t)
    chunked_elife_lay_sum_train.append(new_texts)

### val lay sum
new_elife_lay_sum_val = []
for s in elife_lay_sum_val:
    new_s = s.replace(' . ', '. ')
    new_s = new_s.replace(' , ', ', ')
    new_elife_lay_sum_val.append(new_s)

chunked_elife_lay_sum_val = []
for sum in new_elife_lay_sum_val:
    texts = text_splitter.split_text(sum)
    new_texts = []
    for t in texts:
        t = t.replace('\n\n', ' ')
        new_texts.append(t)
    chunked_elife_lay_sum_val.append(new_texts)



### PLOS train contrastive datasets creation
positive_chunk = []
negative_chunk = []

for lay_sum, doc in zip(chunked_plos_lay_sum_train, chunked_plos_article_train):
    doc_embeddings = model.encode(doc, convert_to_tensor=True)
    lay_embeddings = model.encode(lay_sum, convert_to_tensor=True)
    cosine_scores = util.cos_sim(lay_embeddings, doc_embeddings)
    for i in range(cosine_scores.shape[0]):
        for j in range(cosine_scores.shape[1]):
            if cosine_scores[i][j] >= 0.9:
                positive_chunk.append((lay_sum[i], doc[j], 1))
            elif cosine_scores[i][j] <= 0.01:
                negative_chunk.append((lay_sum[i], doc[j], 0))
                
print("Number of PLOS training positive sentences:", len(positive_chunk))
print("Number of PLOS training negative sentences:", len(negative_chunk))

selected_samples = random.sample(negative_chunk, 17910)

with open('./plos_train_sentence_level_positive_negative_pairs.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['lay_sum', 'sentence', 'label'])
    for row in positive_chunk:
        writer.writerow(row)
    for row in selected_samples:
        writer.writerow(row)
        
### eLife train contrastive datasets creation
positive_chunk = []
negative_chunk = []

for lay_sum, doc in zip(chunked_elife_lay_sum_train, chunked_elife_article_train):
    doc_embeddings = model.encode(doc, convert_to_tensor=True)
    lay_embeddings = model.encode(lay_sum, convert_to_tensor=True)
    cosine_scores = util.cos_sim(lay_embeddings, doc_embeddings)
    for i in range(cosine_scores.shape[0]):
        for j in range(cosine_scores.shape[1]):
            if cosine_scores[i][j] >= 0.9:
                positive_chunk.append((lay_sum[i], doc[j], 1))
            elif cosine_scores[i][j] <= 0.01:
                negative_chunk.append((lay_sum[i], doc[j], 0))
                
print("Number of eLife training positive sentences:", len(positive_chunk))
print("Number of eLife training negative sentences:", len(negative_chunk))

selected_samples = random.sample(negative_chunk, 17910)

with open('./elife_train_sentence_level_positive_negative_pairs.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['lay_sum', 'sentence', 'label'])
    for row in positive_chunk:
        writer.writerow(row)
    for row in selected_samples:
        writer.writerow(row)

### PLOS val contrastive datasets creation
positive_chunk = []
negative_chunk = []
for lay_sum, doc in zip(chunked_plos_lay_sum_val, chunked_plos_article_val):
    doc_embeddings = model.encode(doc, convert_to_tensor=True)
    lay_embeddings = model.encode(lay_sum, convert_to_tensor=True)
    cosine_scores = util.cos_sim(lay_embeddings, doc_embeddings)
    for i in range(cosine_scores.shape[0]):
        for j in range(cosine_scores.shape[1]):
            if cosine_scores[i][j] >= 0.9:
                positive_chunk.append((lay_sum[i], doc[j], 1))
            elif cosine_scores[i][j] <= 0.01:
                negative_chunk.append((lay_sum[i], doc[j], 0))

print("Number of val positive sentences:", len(positive_chunk))
print("Number of val negative sentences:", len(negative_chunk))

selected_samples = random.sample(negative_chunk, 16210)

with open('./plos_val_sentence_level_positive_negative_pairs.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['lay_sum', 'sentence', 'label'])
    for row in positive_chunk:
        writer.writerow(row)
    for row in negative_chunk:
        writer.writerow(row)
        
### eLife val contrastive datasets creation
positive_chunk = []
negative_chunk = []
for lay_sum, doc in zip(chunked_elife_lay_sum_val, chunked_elife_article_val):
    doc_embeddings = model.encode(doc, convert_to_tensor=True)
    lay_embeddings = model.encode(lay_sum, convert_to_tensor=True)
    cosine_scores = util.cos_sim(lay_embeddings, doc_embeddings)
    for i in range(cosine_scores.shape[0]):
        for j in range(cosine_scores.shape[1]):
            if cosine_scores[i][j] >= 0.9:
                positive_chunk.append((lay_sum[i], doc[j], 1))
            elif cosine_scores[i][j] <= 0.01:
                negative_chunk.append((lay_sum[i], doc[j], 0))

print("Number of eLife val positive sentences:", len(positive_chunk))
print("Number of eLife val negative sentences:", len(negative_chunk))

selected_samples = random.sample(negative_chunk, 16210)

with open('./elife_val_sentence_level_positive_negative_pairs.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['lay_sum', 'sentence', 'label'])
    for row in positive_chunk:
        writer.writerow(row)
    for row in negative_chunk:
        writer.writerow(row)