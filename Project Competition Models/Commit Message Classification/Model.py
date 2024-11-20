import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, hamming_loss
import json

#Variable containing count of possible classes(0, 1, 2, 3, 4)
num_classes = 5
class_names = ['bug fix', 'feature addition', 'code refactoring', 'maintenance/other', 'Not enough inforamtion']
#TODO: Check above in codalab

#convert jsonl file into multi-hot encoded dataframe(Large assist from ChatGPT)
def load_labels(file_path):
    # Read the .jsonl file and create a multi-hot encoded DataFrame
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            labels = json.loads(line)
            labels = labels[0]
            multi_hot = [1 if i in labels else 0 for i in range(num_classes)]
            data.append(multi_hot)

    # return as dataframe
    return pd.DataFrame(data, columns=[f"label_{i}" for i in range(num_classes)])

#Import Datasets
X_train = pd.read_csv('Datasets/X_train.csv')
y_train = load_labels('Datasets/y_train.jsonl')

X_dev = pd.read_csv('Datasets/X_dev.csv')
y_dev = load_labels('Datasets/y_dev.jsonl')

X_test = pd.read_csv('Datasets/X_test.csv')
test_output = pd.read_csv('Datasets/testing_data.csv')

#List to store individual models, set threshold
models = []
binary_dev_preds = pd.DataFrame(index=y_dev.index)
binary_test_preds = pd.DataFrame(index=X_test.index)
threshold = 0.3

#Train a model for each individual label
for label in y_train.columns:
    xgb_train = xgb.DMatrix(X_train, label=y_train[label])
    xgb_dev = xgb.DMatrix(X_dev)
    xgb_test = xgb.DMatrix(X_test)

    #Train Model
    xgb_params = {
        'objective':'binary:logistic',
        'max_depth': 5,
        'learning_rate': 0.1
    }
    model1 = xgb.train(dtrain=xgb_train, params=xgb_params, num_boost_round=50)
    models.append(model1)

    #Predict on validation set
    dev_preds = model1.predict(xgb_dev)
    binary_dev_preds[label] = (dev_preds > threshold).astype(int)

    #Predict on test set
    test_preds = model1.predict(xgb_test)
    binary_test_preds[label] = (test_preds > threshold).astype(int)

print(hamming_loss(y_dev, binary_dev_preds))
print(binary_dev_preds)

# Show validation results
print("Validation Results:")
for label in y_dev.columns:
    print(f"Results for {label}:")
    print(classification_report(y_dev[label], binary_dev_preds[label]))

#Function for converting number predictions to original label names
def num_to_label(labels):
    output = []
    for i in range(num_classes):
        if labels[i] == 1:
            output.append(class_names[i])
    output = ','.join(output)
    #fill nulls
    if output == "":
        output = class_names[4]
    return output

#Convert test labels to proper format and store within output dataframe
text_y_test = binary_test_preds.apply(num_to_label, axis=1)
#fill any null cells
print(text_y_test)

#Write text test predictions to output file
final_y_test = pd.DataFrame({
    'id': test_output['id'],
    'Ground truth': text_y_test
})
final_y_test.to_csv('Datasets/y_test.csv', index=False)