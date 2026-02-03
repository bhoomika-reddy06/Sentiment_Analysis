
# Sentiment Analysis - Flipkart Reviews


import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report


# Download required NLTK data

nltk.download('stopwords')
nltk.download('wordnet')


# Load Dataset

df = pd.read_csv("badmiton_review_data.csv")

print("Dataset Shape:", df.shape)


# Remove Neutral Reviews (Rating = 3)

df = df[df['Ratings'] != 3]


# Create Sentiment Label
# 1 -> Positive, 0 -> Negative

df['sentiment'] = df['Ratings'].apply(lambda x: 1 if x >= 4 else 0)


# Text Cleaning Function

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


# Apply Text Cleaning

df['clean_review'] = df['Review text'].apply(clean_text)


# Feature Extraction (TF-IDF)

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment']


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Model Training

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Model Evaluation

y_pred = model.predict(X_test)

print("\nF1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# Save Model & Vectorizer

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\n✅ Model and Vectorizer saved successfully!")
