import streamlit as st
import pickle
from preprocessing import clean_text

# Load model
model = pickle.load(open("models/model.pkl", "rb"))
tfidf = pickle.load(open("models/vectorizer.pkl", "rb"))

st.title("Resume Screening System")

text = st.text_area("Paste Resume Text")

if st.button("Predict"):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    result = model.predict(vector)[0]
    
    st.success(f"Predicted Role: {result}")