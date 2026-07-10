import streamlit as st
import joblib

# Load model
model = joblib.load("trained_model.joblib")

# Load vectorizer
vectorizer = joblib.load("tfidf_vectorizer.joblib")

st.set_page_config(page_title="Customer Support Ticket Classifier")

st.title("🎫 Customer Support Ticket Classification")

st.write("Enter the customer support ticket description below.")

ticket = st.text_area("Ticket Description")

if st.button("Predict"):

    if ticket.strip() == "":
        st.warning("Please enter a ticket description.")
    else:

        # Same preprocessing used during training
        ticket = ticket.lower()

        vector = vectorizer.transform([ticket])

        prediction = model.predict(vector)

        st.success(f"Predicted Ticket Type: {prediction[0]}")
