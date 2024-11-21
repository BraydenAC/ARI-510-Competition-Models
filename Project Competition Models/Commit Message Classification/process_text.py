import torch
from transformers import AutoTokenizer , AutoModel
import pandas as pd
import json
import numpy as np


# Tokenize text
#Initialize pretrained model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

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
    switcher = {
        'bug fix': 0,
        'feature addition': 1,
        'code refactoring': 2,
        'maintenance/other': 3,
        'not enough information': 4,
        'not enough inforamtion': 4
    }
    if pd.isna(labels):
        return [-2]
    return [switcher.get(label.strip().lower(), -1) for label in labels.split(',')]


#Function to remove nulls
def preprocess_text(data, column_name):
    return data[column_name].fillna("").astype(str)

#Load in data
print("Loading data...")
pre_X_train = pd.read_csv('Datasets/training_data.csv')
pre_X_train = pre_X_train.drop(['id', 'Commit Hash', 'Old Contents', 'New Contents'], axis=1)
pre_y_train = pd.read_csv('Datasets/training_label.csv')
pre_y_train = pre_y_train.drop(['id'], axis=1)

pre_X_dev = pd.read_csv('Datasets/validation_data.csv')
pre_X_dev = pre_X_dev.drop(['id', 'Commit Hash', 'Old Contents', 'New Contents'], axis=1)
pre_y_dev = pd.read_csv('Datasets/validation_label.csv')
pre_y_dev = pre_y_dev.drop(['id'], axis=1)

pre_X_test = pd.read_csv('Datasets/testing_data.csv')
pre_X_test = pre_X_test.drop(['id', 'Commit Hash', 'Old Contents', 'New Contents'], axis=1)

#process data
print("processing training data...")
processed_train = pd.concat([pd.DataFrame(tokenize_text_in_batches(pre_X_train['Subject'].tolist())),
                             pd.DataFrame(tokenize_text_in_batches(pre_X_train['Message'].tolist()))], axis=1, ignore_index=True)
processed_train_labels = pd.DataFrame(pre_y_train['Ground truth'].apply(label_to_num))
print("processing dev data...")
processed_dev = pd.concat([pd.DataFrame(tokenize_text_in_batches(pre_X_dev['Subject'].tolist())),
                             pd.DataFrame(tokenize_text_in_batches(pre_X_dev['Message'].tolist()))], axis=1, ignore_index=True)
processed_dev_labels = pd.DataFrame(pre_y_dev['Ground truth'].apply(label_to_num))
print("processing test data...")
processed_test = pd.concat([pd.DataFrame(tokenize_text_in_batches(pre_X_test['Subject'].tolist())),
                             pd.DataFrame(tokenize_text_in_batches(pre_X_test['Message'].tolist()))], axis=1, ignore_index=True)

#JSONL Writer Function(From ChatGPT)
def save_to_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')

#save to processed files
print("Saving data...")
processed_train.to_csv('Datasets/X_train.csv', index=False)
save_to_jsonl(processed_train_labels.values.tolist(), 'Datasets/y_train.jsonl')

processed_dev.to_csv('Datasets/X_dev.csv', index=False)
save_to_jsonl(processed_dev_labels.values.tolist(), 'Datasets/y_dev.jsonl')

processed_test.to_csv('Datasets/X_test.csv', index=False)
print("Done!")