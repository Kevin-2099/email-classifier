# app_en.py (English version)
import gradio as gr
from transformers import pipeline
import re

# -----------------------------
# Pipelines
# -----------------------------
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")

# -----------------------------
# Keywords
# -----------------------------
KEYWORDS = {
    "Urgent": ["urgent", "immediate", "critical", "failure", "problem", "attention"],
    "Important but not urgent": ["important", "review", "feedback", "meeting", "document", "contract"],
    "Spam/Promotion": ["offer", "promotion", "subscribe", "buy", "discount", "make money", "now"],
    "Informative": ["informative", "reminder", "notification", "notice", "summary"]
}

CONTEXT_KEYWORDS = {
    "Urgent": [["respond", "urgent"], ["attention", "immediate"]],
}

NEGATION_WORDS = ["no", "not", "never", "without"]

# -----------------------------
# Helper functions
# -----------------------------
def has_negation_robust(text, keyword, window=5):
    text_clean = re.sub(r'[.,;!?]', ' ', text.lower())
    words = text_clean.split()
    for i, w in enumerate(words):
        if w == keyword.lower():
            start = max(0, i - window)
            context = words[start:i]
            if any(neg in context for neg in NEGATION_WORDS):
                return True
    return False

def check_context(text, keyword):
    text_clean = re.sub(r'[.,;!?]', ' ', text.lower())
    words = text_clean.split()
    for ctx_pair in CONTEXT_KEYWORDS.get("Urgent", []):
        if keyword in ctx_pair:
            if all(word in words for word in ctx_pair):
                return True
            else:
                return False
    return True

# -----------------------------
# Main function
# -----------------------------
def classify_email(email_text):
    text_lower = email_text.lower()
    
    # Keyword-based classification
    for category, words in KEYWORDS.items():
        for word in words:
            if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
                if has_negation_robust(email_text, word):
                    continue
                if not check_context(email_text, word):
                    continue
                explanation = f"The keyword '{word}' was detected, indicating the category '{category}'."
                break
        else:
            continue
        break
    else:
        # Fallback model
        pred = classifier(email_text)[0]
        label = pred['label']
        score = pred['score']

        if label == 'NEGATIVE' and score > 0.7:
            category = "Urgent"
            explanation = "The email content suggests urgency according to the model."
        elif label == 'POSITIVE' and score > 0.7:
            category = "Important but not urgent"
            explanation = "The email is relevant but not urgent according to the model."
        else:
            category = "Informative"
            explanation = "The email appears to be informational according to the model."

    # Sentiment
    sentiment = sentiment_analyzer(email_text)[0]
    tone = f"{sentiment['label']} ({sentiment['score']:.2f})"

    # Summary
    try:
        summary = summarizer(email_text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    except:
        summary = email_text[:100] + "..."

    # HTML card output
    html_output = f"""
    <div style='border:2px solid #4CAF50; padding:15px; border-radius:10px; max-width:700px; background-color:#f9f9f9;'>
        <h3>📧 Email Classification</h3>
        <p><b>Category:</b> {category}</p>
        <p><b>Tone:</b> {tone}</p>
        <p><b>Summary:</b> {summary}</p>
        <p><b>Explanation:</b> {explanation}</p>
    </div>
    """
    return html_output

# -----------------------------
# Gradio interface
# -----------------------------
iface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=10, placeholder="Paste your email here..."),
    outputs=gr.HTML(),
    title="Enhanced Email Classifier",
    description="Classifies emails as Urgent, Important, Informative, or Spam. Also detects tone and generates an automatic summary in a visual card."
)

if __name__ == "__main__":
    iface.launch()
