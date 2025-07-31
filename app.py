import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.write("Enter a movie review and get a prediction whether it's **Positive** or **Negative**.")

# Text input
review = st.text_area("✍️ Your Review:", height=200)

# Predict button
if st.button("🔍 Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        vectorized = vectorizer.transform([review])
        prediction = model.predict(vectorized)[0]
        sentiment = "🌟 Positive" if prediction == 1 else "💔 Negative"
        st.success(f"Predicted Sentiment: {sentiment}")
