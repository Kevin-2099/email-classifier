# Email classifier

This project is an intelligent email classification tool built with Python, Hugging Face, and Gradio.
It analyzes email text and classifies it into one of the following categories, while also providing tone detection and an automatic summary:

- 🟥 **Urgent** (requires immediate attention)  
- 🟨 **Important but not urgent**  
- 🟦 **Informative**  
- 🟩 **Spam / Promotion**  

---

## 🚀 Technologies Used
- Transformers (Hugging Face) → models:

- distilbert-base-uncased-finetuned-sst-2-english (text classification)

- summarization pipeline (automatic summary)

- sentiment-analysis pipeline (tone detection)

- Gradio → interactive web interface with visual card output

- scikit-learn → optional, for preprocessing or future training
---

## 🧠 How It Works
- Rule-based classification

The system checks for keywords typical for each category (e.g., "urgent", "immediate", "offer", "reminder").
Context rules and negation detection help reduce false positives.

- Fallback to Hugging Face model

If no clear keywords are found, the DistilBERT model predicts the category based on email content.
This ensures the system always provides a reasonable classification.

- Tone detection

The system analyzes the sentiment of the email (positive, negative, neutral) to give an additional tone indicator.

- Automatic summary

A short summary of the email content is generated automatically using the summarization pipeline.

- Visual card output

The results are displayed in a single visual card with category, tone, summary, and explanation, so everything is easy to read without scrolling.
