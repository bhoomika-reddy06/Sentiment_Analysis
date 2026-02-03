import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Page Config

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)


# Custom CSS Styling

st.markdown("""
<style>

/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Main background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

/* Heading style */
.custom-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #1f1f1f;   /* Dark color */
    margin-bottom: 10px;
}

/* Subtitle style */
.custom-subtitle {
    text-align: center;
    font-size: 18px;
    font-weight: 400;
    color: #2d2d2d;   /* Slight dark grey */
    margin-bottom: 30px;
}

/* Button styling */
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #e63946;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Header Section

st.markdown('<div class="custom-title">💬 Sentiment Analysis App</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle">Analyze your review instantly using Machine Learning 🚀</div>', unsafe_allow_html=True)


st.write("")


# Load Model & Vectorizer

@st.cache_resource
def load_models():
    model = pickle.load(open("sentiment_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_models()


# NLTK Setup

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Text Cleaning Function

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


# Input Section

review = st.text_area("📝 Enter your review here:", height=150)

st.write("")


# Prediction

if st.button("✨ Predict Sentiment"):

    if not review.strip():
        st.warning("⚠️ Please enter a review!")
    else:
        cleaned_review = clean_text(review)
        vector = vectorizer.transform([cleaned_review])
        prediction = model.predict(vector)[0]

        st.markdown("---")

        if prediction == 1:
            st.markdown(
                "<h2 style='color: green; text-align: center;'>😊 Positive Review</h2>",
                unsafe_allow_html=True
            )
            st.balloons()
        else:
            st.markdown(
                "<h2 style='color: red; text-align: center;'>😞 Negative Review</h2>",
                unsafe_allow_html=True
            )

        st.markdown("### 📌 Your Review:")
        st.info(review)
