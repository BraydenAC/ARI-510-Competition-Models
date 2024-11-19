import pandas as pd
import numpy as np
import xgboost as xgb

#Import Datasets
X_train = pd.read_csv('Datasets/X_train.csv')
y_train = pd.read_json('Datasets/y_train.jsonl', lines=True)

X_dev = pd.read_csv('Datasets/X_dev.csv')
y_dev = pd.read_json('Datasets/y_dev.jsonl', lines=True)

X_test = pd.read_csv('Datasets/X_test.csv')

#Initialize Model
xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=False)
xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=False)

model1 = xgb.
#Train Model

#Predict on validation set

#Show validation results

#Predict on test set

#Write test predicitions to output file