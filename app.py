import streamlit as st
import pickle

# Load model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

st.title("Customer Support Ticket Classification")

ticket = st.text_area("Enter Ticket Description")

if st.button("Predict"):
    if ticket.strip():
        vector = vectorizer.transform([ticket.lower()])
        prediction = model.predict(vector)
        st.success(f"Predicted Ticket Type: {prediction[0]}")
    else:
        st.warning("Please enter a ticket description.")
    
