import torch
from transformers import AutoTokenizer , AutoModel
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import numpy as np


# Tokenize text
#Initialize pretrained model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
headers = ['ID', 'Reviewer Name', 'Review Text', 'Date of Collection', 'Annotator_1', 'Annotator_2', 'Ground_Truth']

model.eval()

def tokenize_text_in_batches(texts, batch_size=50):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print("Working on number: " + str(i))
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def label_to_num(labels):
    num_labels = []
    for i in labels:
        if i == 'Negative':
            num_labels.append(0)
        elif i == 'Neutral':
            num_labels.append(1)
        else:
            num_labels.append(2)
    return num_labels

def preprocess_text(data, column_name):
    return data[column_name].fillna("").astype(str)

#Load in data
print("Loading data...")
unprocessed_train_dev = pd.read_csv('data/train_data.csv')
unprocessed_test = pd.read_csv('data/test_data.csv')

#clean data
unprocessed_train_dev['Review Text'] = preprocess_text(unprocessed_train_dev, column_name='Review Text')
unprocessed_test['Review Text'] = preprocess_text(unprocessed_test, column_name='Review Text')

#split train into train and dev
X_train, X_dev, y_train, y_dev= train_test_split(unprocessed_train_dev['Review Text'], unprocessed_train_dev['Ground_Truth'], test_size=0.1, random_state=42)

#process data
print("processing training data...")
processed_train = pd.DataFrame(tokenize_text_in_batches(X_train.tolist()))
processed_train_labels = pd.DataFrame(label_to_num(y_train))
print("processing dev data...")
processed_dev = pd.DataFrame(tokenize_text_in_batches(X_dev.tolist()))
processed_dev_labels = pd.DataFrame(label_to_num(y_dev))
print("processing test data...")
processed_test = pd.DataFrame(tokenize_text_in_batches(unprocessed_test['Review Text'].tolist()))

#save to processed files
print("Saving data...")
processed_train.to_csv('data/text_train.csv', index=False)
processed_train_labels.to_csv('data/text_train_labels.csv', index=False)
processed_dev.to_csv('data/text_dev.csv', index=False)
processed_dev_labels.to_csv('data/text_dev_labels.csv', index=False)
processed_test.to_csv('data/text_test.csv', index=False)
print("Done!")