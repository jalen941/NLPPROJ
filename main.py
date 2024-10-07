import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

train_data = pd.read_csv('train_amazon.csv', header=None, names=['label', 'text'])
test_data = pd.read_csv('test_amazon.csv', header=None, names=['label', 'text'])

train_data.dropna(subset=['text'], inplace=True)
test_data.dropna(subset=['text'], inplace=True)
train_data = train_data[train_data['text'].str.strip() != '']
test_data = test_data[test_data['text'].str.strip() != '']


X_train = train_data['text']
print(X_train)
y_train = train_data['label']

X_test = test_data['text']
y_test = test_data['label']

vectorizer = CountVectorizer(stop_words='english', max_features=8000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("Training Model 1 (default setup)...")
start_time = time.time()
model_1 = LogisticRegression(max_iter=200, solver='lbfgs', C=1.0)
model_1.fit(X_train_vectorized, y_train)
train_time_1 = time.time() - start_time

start_time = time.time()
y_pred_1 = model_1.predict(X_test_vectorized)
inference_time_1 = time.time() - start_time

accuracy_1 = accuracy_score(y_test, y_pred_1)
print(f"Model 1 Accuracy: {accuracy_1}")
print(f"Model 1 Training Time: {train_time_1} seconds")
print(f"Model 1 Inference Time: {inference_time_1} seconds")
print("Model 1 Classification Report:\n", classification_report(y_test, y_pred_1, zero_division=0))
print("Model 1 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_1))


