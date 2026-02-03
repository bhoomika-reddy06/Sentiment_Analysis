from flask import Flask, render_template, request
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask App

app = Flask(__name__)


# Load Trained Model & Vectorizer

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))


# NLTK Setup (same as training)

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Text Cleaning Function
# (MUST MATCH TRAINING SCRIPT)

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


# Routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review_text')

    if not review or not review.strip():
        return render_template(
            'index.html',
            prediction="⚠️ Please enter a review!"
        )

    cleaned_review = clean_text(review)
    vector = vectorizer.transform([cleaned_review])
    prediction = model.predict(vector)[0]

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

    return render_template(
        'index.html',
        prediction=sentiment,
        user_review=review
    )



# Run Flask App

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)
