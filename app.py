import gradio as gr
import pickle

# Load vectorizer + model
with open("spam_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

def predict_spam(email_text):
    if not email_text.strip():
        return "⚠️ Please enter some text"
    features = vectorizer.transform([email_text])
    prediction = model.predict(features)[0]
    return "🚨 Spam" if prediction == 1 else "✅ Not Spam"

demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=5, placeholder="Paste email text here...", label="Email Text"),
    outputs=gr.Label(label="Prediction"),
    title="📧 Spam Email Classifier",
    description="Classify emails as Spam or Not Spam using a trained ML model."
)

if __name__ == "__main__":
    demo.launch()
