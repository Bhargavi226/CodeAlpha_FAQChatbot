import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Title
st.title("🤖 FAQ Chatbot")
st.write("Ask a question related to the FAQs.")

# Load FAQ data
faq = pd.read_csv("faq.csv")

# Convert questions into vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(faq["Question"])

# User input
user_question = st.text_input("Enter your question:")

# Find the best matching answer
if user_question:
    user_vector = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, question_vectors)

    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score >= 0.8:
        st.success(faq.iloc[best_match]["Answer"])
    else:
        st.error("Sorry! I don't have an answer for that question.")