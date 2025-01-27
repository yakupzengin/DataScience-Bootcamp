import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import re

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to analyze its sentiment.")

model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))
review = st.text_input('Enter your review:')   

def preprocess_text(text):
    """Preprocess the input text: lowercase, remove punctuation, and extra spaces."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

if st.button("Predict"):
    if not review.strip():
        st.write("Please enter a review.")
        st.stop()
    else:
        review = preprocess_text(review)
        tfidf = TfidfVectorizer(max_features=5000)
        review = scaler.transform([review]).toarray()
        prediction = model.predict(review)
        if prediction[0] == 0:
            st.write("Negative Review")
        else:
            st.write("Positive Review") 