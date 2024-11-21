import pandas as pd
import sklearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

#Load in Data
pre_X_train = pd.read_csv('data/text_train.csv')
pre_y_train = pd.read_csv('data/text_train_labels.csv')

X_dev = pd.read_csv('data/text_dev.csv')
y_dev = pd.read_csv('data/text_dev_labels.csv')

X_test = pd.read_csv('data/text_test.csv')



smote = SMOTE(random_state=42)
undersampler = RandomUnderSampler(sampling_strategy={0: 14, 1: 31, 2: 31}, random_state=42)
mid_X_train, mid_y_train = undersampler.fit_resample(pre_X_train, pre_y_train)
X_train, y_train = smote.fit_resample(mid_X_train, mid_y_train)

print("Class distribution before SMOTE:")
print(pre_y_train.value_counts())
print("Class distribution after SMOTE:")
print(y_train.value_counts())

y_train = y_train.values.ravel()
y_dev = y_dev.values.ravel()


#Random Forest Classifier
model2 = RandomForestClassifier(class_weight='balanced_subsample', max_depth=10, n_estimators=200)
model2.fit(X_train, y_train)

model2_pred = model2.predict(X_dev)
print('Predictions')
print(model2_pred)
print("Ground Truth")
print(y_dev)
print(f"Model 2 accuracy: {accuracy_score(y_dev, model2_pred)}")

print(classification_report(y_dev, model2_pred))

#Make test predictions and store them in preds.csv

test_predictions = model2.predict(X_test)
print("Test Predictions")
print(test_predictions)
processed_test_predictions = []

for pred in test_predictions:
    if pred == 0:
        processed_test_predictions.append('Negative')
    elif pred == 1:
        processed_test_predictions.append('Neutral')
    else:
        processed_test_predictions.append('Positive')

pd.DataFrame(processed_test_predictions).to_csv('data/preds.csv', index=False)