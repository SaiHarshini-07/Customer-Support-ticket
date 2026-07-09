import gradio as gr
import pickle

# Load model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


def predict_ticket(text):

    text = text.lower()

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)

    return prediction[0]


demo = gr.Interface(
    fn=predict_ticket,
    inputs=gr.Textbox(
        lines=5,
        label="Enter Ticket Description"
    ),
    outputs=gr.Textbox(label="Predicted Ticket Type"),
    title="Customer Support Ticket Classification",
    description="Predict customer support ticket categories using Machine Learning."
)

demo.launch()
