# app.py
import gradio as gr
from transformers import pipeline
import re

# Load Hugging Face model (fallback)
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Keywords per category
KEYWORDS = {
    "Urgent": ["urgent", "immediate", "critical", "failure", "problem", "attention", "right away", "asap"],
    "Important but not urgent": ["important", "review", "feedback", "meeting", "document", "contract"],
    "Spam/Promotion": [
        "offer", "promotion", "subscribe", "buy", "discount", "earn money", "sign up", "limited offer",
        "exclusive", "deal", "save big", "buy now"
    ],
    "Informative": ["informative", "reminder", "notification", "notice", "summary"]
}

NEGATION_WORDS = ["not", "without", "never"]

def has_negation(text, keyword, window=5):
    text_clean = re.sub(r'[.,;!?]', ' ', text.lower())
    words = text_clean.split()
    for i, w in enumerate(words):
        if w == keyword.lower():
            start = max(0, i - window)
            context = words[start:i]
            if any(neg in context for neg in NEGATION_WORDS):
                return True
    return False

def classify_email(email_text):
    text_lower = email_text.lower()

    # Rule-based classification takes priority
    for category, words in KEYWORDS.items():
        for word in words:
            if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
                if has_negation(email_text, word):
                    continue
                explanation = f"Detected the keyword '{word}', which indicates category '{category}'."
                return category, explanation

    # Fallback to Hugging Face model
    pred = classifier(email_text)[0]
    label = pred['label']
    score = pred['score']

    if label == 'NEGATIVE' and score > 0.7:
        return "Urgent", "The content suggests urgency according to the sentiment model."
    elif label == 'POSITIVE' and score > 0.7:
        return "Important but not urgent", "The content seems relevant but not urgent according to the sentiment model."
    else:
        return "Informative", "The email seems to be informational according to the model."

# Gradio interface
iface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=10, placeholder="Paste the email text here..."),
    outputs=[gr.Label(num_top_classes=1), gr.Textbox()],
    title="Email Classifier (with Priority Rules)",
    description="Classifies emails into: Urgent, Important, Informative, Spam. Keyword rules have priority over the model."
)

if __name__ == "__main__":
    iface.launch()
