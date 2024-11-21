import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

training_set = pd.read_csv('training_set_labeled.csv')
training_headers = ['id', 'skills', 'exp', 'grades', 'projects', 'extra', 'offer', 'hire', 'pay']

training_set.columns = training_headers


X_train_dev = training_set.drop([training_headers[7], training_headers[8]], axis=1)
y_train_dev = training_set[[training_headers[7], training_headers[8]]]

X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.1, random_state=42)

y1_train = y_train[training_headers[7]]
y2_train = y_train[training_headers[8]]
y1_dev = y_dev[training_headers[7]]
y2_dev = y_dev[training_headers[8]]

X_train.to_csv('X_train.csv', index=False)
y1_train.to_csv('y1_train.csv', index=False)
y2_train.to_csv('y2_train.csv', index=False)

X_dev.to_csv('X_dev.csv', index=False)
y1_dev.to_csv('y1_dev.csv', index=False)
y2_dev.to_csv('y2_dev.csv', index=False)