# ====================================================
# training_file.py
# ====================================================

import pandas as pd
import re
import nltk
import mlflow
import mlflow.sklearn

from prefect import flow, task

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')


# ----------------------------
# Load Data
# ----------------------------
@task
def load_data():
    df = pd.read_csv("badmiton_review_data.csv")
    df = df[df['Ratings'] != 3]
    df['sentiment'] = df['Ratings'].apply(lambda x: 1 if x >= 4 else 0)
    return df


# ----------------------------
# Preprocess Data
# ----------------------------
@task
def preprocess_data(df):

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [
            lemmatizer.lemmatize(word)
            for word in words
            if word not in stop_words
        ]
        return " ".join(words)

    df['clean_review'] = df['Review text'].apply(clean_text)
    return df


# ----------------------------
# Train + Log Model
# ----------------------------
@task
def train_and_log_model(df):

    mlflow.set_experiment("Flipkart_Sentiment_Analysis")

    with mlflow.start_run():

        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_review'],
            df['sentiment'],
            test_size=0.2,
            random_state=42,
            stratify=df['sentiment']
        )

        # 🔥 PIPELINE (VERY IMPORTANT)
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("model", LogisticRegression(max_iter=1000))
        ])

        pipeline.fit(X_train, y_train)
        import pickle
        pickle.dump(pipeline, open("sentiment_pipeline.pkl", "wb"))

        y_pred = pipeline.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # ✅ Log FULL Pipeline
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="Flipkart_Sentiment_Model"
        )

    return f1, acc


# ----------------------------
# Flow
# ----------------------------
@flow
def sentiment_training_pipeline():
    df = load_data()
    df = preprocess_data(df)
    train_and_log_model(df)


if __name__ == "__main__":
    sentiment_training_pipeline()
