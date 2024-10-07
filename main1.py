import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


train_data = pd.read_csv('train_amazon.csv', header=None, names=['label', 'text'])
test_data = pd.read_csv('test_amazon.csv', header=None, names=['label', 'text'])


train_data.dropna(subset=['text'], inplace=True)
test_data.dropna(subset=['text'], inplace=True)
train_data = train_data[train_data['text'].str.strip() != '']
test_data = test_data[test_data['text'].str.strip() != '']


def preprocess_text(text):

    text = text.lower()

    # remove all non-word characters
    text = re.sub(r'[^a-z\s]', '', text)

    # rmove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


    return text


X_train = train_data['text'].apply(preprocess_text)

y_train = train_data['label']

X_test = test_data['text'].apply(preprocess_text)
y_test = test_data['label']


vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=8000)  # Use bigrams
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


print("\nTraining Model 2 (with preprocessing and tweaked parameters)...")
start_time = time.time()
model_2 = LogisticRegression(max_iter=500, solver='lbfgs', C=.01)
model_2.fit(X_train_vectorized, y_train)

train_time_2 = time.time() - start_time


start_time = time.time()
y_pred_2 = model_2.predict(X_test_vectorized)
inference_time_2 = time.time() - start_time


accuracy_2 = accuracy_score(y_test, y_pred_2)
print(f"Model 2 Accuracy: {accuracy_2}")
print(f"Model 2 Training Time: {train_time_2} seconds")
print(f"Model 2 Inference Time: {inference_time_2} seconds")
print("Model 2 Classification Report:\n", classification_report(y_test, y_pred_2, zero_division=0))
print("Model 2 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_2))
