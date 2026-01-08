import pandas as pd
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("spam.csv", sep="\t", header=None)
df.columns = ["label", "text"]

emails = df["text"].tolist()
labels = df["label"].map({"spam": 1, "ham": 0}).tolist()
X_train, X_test, y_train, y_test = train_test_split(
    emails,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)
import numpy as np

def extract_features(texts):
    return np.array([
        [len(t), sum(c.isdigit() for c in t)]
        for t in texts
    ])


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

X_train_struct = extract_features(X_train)
X_test_struct = extract_features(X_test)

X_train_final = hstack([X_train_vec, X_train_struct])
X_test_final = hstack([X_test_vec, X_test_struct])

model = MultinomialNB()
model.fit(X_train_final, y_train)
y_probs = model.predict_proba(X_test_final)[:, 1]
y_pred = (y_probs >= 0.3).astype(int)

y_test_arr = np.array(y_test)
X_test_arr = np.array(X_test)

false_negatives = X_test_arr[(y_test_arr == 1) & (y_pred == 0)]

print("Number of false negatives:", len(false_negatives))
print("\nSample false negatives:\n")

for msg in false_negatives[:10]:
    print("-", msg)


print(len(emails))
print(len(X_train), len(X_test))
print(X_train_vec.shape)
print(X_test_vec.shape)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib

THRESHOLD = 0.3

joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(model, "spam_classifier.joblib")
joblib.dump(THRESHOLD, "decision_threshold.joblib")

print("Model, vectorizer, and threshold saved successfully.")
