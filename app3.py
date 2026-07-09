import streamlit as st
import pickle

# Load model
with open("best_model.pkl","wb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("tfidf_vectorizer.pkl","wb") as file:
    vectorizer = pickle.load(file)

st.title("Customer Support Ticket Classification")

st.write("Enter a customer support ticket below.")

ticket = st.text_area("Ticket Description")

if st.button("Predict"):

    ticket = ticket.lower()

    vector = vectorizer.transform([ticket])

    prediction = model.predict(vector)

    st.success("Predicted Ticket Type")

    st.write(prediction[0])
