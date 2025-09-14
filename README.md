# Email classifier

This project is an **intelligent email classification tool** built with **Python, Hugging Face, and Gradio**.  
It analyzes email text and classifies it into one of the following categories:

- 🟥 **Urgent** (requires immediate attention)  
- 🟨 **Important but not urgent**  
- 🟦 **Informative**  
- 🟩 **Spam / Promotion**  

---

## 🚀 Technologies Used
- [Transformers (Hugging Face)](https://huggingface.co/transformers/) → model: `distilbert-base-uncased-finetuned-sst-2-english`  
- [Gradio](https://gradio.app/) → interactive web interface  
- [scikit-learn](https://scikit-learn.org/) → optional, for preprocessing or future training  

---

## ⚙️ Installation

Clone the repository and install dependencies:

git clone https://github.com/yourusername/email-classifier.git

cd email-classifier

pip install -r requirements.txt

Run the application:

python app.py

This will launch a Gradio web interface at http://127.0.0.1:7860/.

## 🧠 How It Works
1.Rule-based classification

The system checks for keywords typical for each category (e.g., "urgent", "immediate", "offer", "reminder").

Context rules and negation detection reduce false positives.

2.Fallback to Hugging Face model

If no clear keywords are found, the distilBERT model predicts the category based on email content.

This ensures the system always provides a reasonable classification.
