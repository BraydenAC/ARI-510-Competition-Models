import pandas as pd
import numpy as np

#Import Datasets
X_train = pd.read_csv('Datasets/training_data.csv')
X_train = X_train.drop(['Id', 'Commit Hash', 'Old Contents', 'New Contents'], axis=1).values #Try with and without Old Contents and New Contents
y_train = pd.read_csv('Datasets/training_labels.csv')
y_train = y_train.drop(['Id'], axis=1).values

X_dev = pd.read_csv('Datasets/validation_data.csv.csv')
X_dev = X_dev.drop(['Id', 'Commit Hash', 'Old Contents', 'New Contents'], axis=1).values #Try with and without Old Contents and New Contents
y_dev = pd.read_csv('Datasets/validation_label.csv.csv')
y_dev = y_dev.drop(['Id'], axis=1).values

X_test = pd.read_csv('Datasets/testing_data.csv.csv')
X_test = X_test.drop(['Id', 'Commit Hash', 'Old Contents', 'New Contents'], axis=1).values #Try with and without Old Contents and New Contents

#