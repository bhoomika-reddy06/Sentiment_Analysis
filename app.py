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

/* Light Gradient Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f3f7ff, #e6ecff);
}

/* Glass Container Effect */
.block-container {
    max-width: 700px;
    margin: auto;
    background-color: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(12px);
    padding: 2.5rem;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.08);
}


/* Heading */
.custom-title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    color: #2c3e50;
    margin-bottom: 5px;
}

/* Subtitle */
.custom-subtitle {
    text-align: center;
    font-size: 18px;
    color: #5f6c7b;
    margin-bottom: 30px;
}

/* Text Area Styling */
textarea {
    border-radius: 12px !important;
    border: 1px solid #dfe6e9 !important;
    padding: 10px !important;
    font-size: 16px !important;
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(135deg, #89f7fe, #66a6ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #66a6ff, #89f7fe);
    color: white;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: #e0e0e0;
}

</style>
""", unsafe_allow_html=True)



# Header Section

st.markdown('<div class="custom-title">💬 Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle">Analyze your review instantly using Machine Learning 🚀</div>', unsafe_allow_html=True)

st.write("")



# Load Model & Vectorizer

@st.cache_resource
def load_models():
    return pickle.load(open("sentiment_pipeline.pkl", "rb"))

model=load_models()



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



# Prediction Section

if st.button("✨ Predict Sentiment"):

    if not review.strip():
        st.warning("⚠️ Please enter a review!")
    else:
        cleaned_review = clean_text(review)
        prediction = model.predict([cleaned_review])[0]


        st.markdown("---")

        if prediction == 1:
            st.markdown("""
                <div style="
                    background-color:#e8f9f1;
                    padding:20px;
                    border-radius:15px;
                    text-align:center;
                    font-size:24px;
                    font-weight:600;
                    color:#2ecc71;">
                    😊 Positive Review
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown("""
                <div style="
                    background-color:#fdecea;
                    padding:20px;
                    border-radius:15px;
                    text-align:center;
                    font-size:24px;
                    font-weight:600;
                    color:#e74c3c;">
                    😞 Negative Review
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📌 Your Review:")
        st.info(review)
