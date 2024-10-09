import pandas as pd
import re
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns

lemmatizer = WordNetLemmatizer()


train_data = pd.read_csv('train_amazon.csv', header=0, names=['label', 'text'])
test_data = pd.read_csv('test_amazon.csv', header=0, names=['label', 'text'])


train_data.dropna(subset=['text'], inplace=True)
test_data.dropna(subset=['text'], inplace=True)
train_data = train_data[train_data['text'].str.strip() != '']
test_data = test_data[test_data['text'].str.strip() != '']


def preprocess_text(text):
    text = text.lower()

    # remove all non-word characters
    text = re.sub(r'[^a-z\s]', '', text)

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text



X_train_full = train_data['text'].apply(preprocess_text)
y_train_full = train_data['label']
X_test = test_data['text'].apply(preprocess_text)
y_test = test_data['label']


vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=8000)
X_test_vectorized = vectorizer.fit_transform(X_test)


def train_with_sample_size(sample_size):
    print(f"\nTraining with {sample_size} samples...")

    sample_indices = random.sample(range(len(X_train_full)), sample_size)
    X_train_sample = X_train_full.iloc[sample_indices]
    y_train_sample = y_train_full.iloc[sample_indices]


    X_train_vectorized = vectorizer.fit_transform(X_train_sample)


    start_time = time.time()
    model = LogisticRegression(max_iter=500, solver='lbfgs', C=0.01)
    model.fit(X_train_vectorized, y_train_sample)
    train_time = time.time() - start_time


    start_time = time.time()
    y_pred = model.predict(X_test_vectorized)
    inference_time = time.time() - start_time


    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {sample_size} samples: {accuracy}")
    print(f"Training Time: {train_time} seconds")
    print(f"Inference Time: {inference_time} seconds")
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))


    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Model 1 Confusion Matrix:\n", conf_matrix)


    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()



sample_sizes = [50000, 60000, 70000, 80000]
for sample_size in sample_sizes:
    train_with_sample_size(sample_size)
