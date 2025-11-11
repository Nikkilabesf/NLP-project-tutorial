<div align="center">

# ✨ URL Spam Detection System  
### 🚨 Detect malicious or spammy links using Machine Learning  

*Built with love, data, and code by **Tenika Powell** 💻🩵*  

</div>

---

## 🌐 Overview  

The **URL Spam Detection System** is a Natural Language Processing (NLP) and Machine Learning project that automatically classifies web links as **SPAM** or **SAFE**.  
It demonstrates an end-to-end workflow from raw text cleaning to model training, evaluation, and deployment — all inside a Jupyter notebook.

---

## ⚙️ Tech Stack  

| Category | Tools |
|-----------|--------|
| **Language** | Python 🐍 |
| **Libraries** | `pandas`, `nltk`, `scikit-learn`, `joblib` |
| **Model** | Support Vector Machine (SVM) |
| **Feature Engineering** | TF-IDF Vectorization |
| **Environment** | Jupyter Notebook → `url_spam_detector.ipynb` |

---

## 🧩 Project Workflow  

1. **Load Dataset** → `url_spam.csv`  
2. **Clean URLs** → remove protocols, symbols, normalize case  
3. **Vectorize Text** → TF-IDF converts text into numeric features  
4. **Train Model** → linear SVM learns spam vs. safe patterns  
5. **Evaluate Model** → accuracy, precision, recall, F1-score  
6. **Save Artifacts** → export trained model & vectorizer  

---

## 🧼 Data Cleaning  

Each URL is processed with regex & tokenization to keep meaningful words like  
`paypal`, `login`, `secure`, `update`, `net`, etc.

Example: Original: http://paypal-login-secure-update.com
Cleaned: paypal-login-secure-update.com




---

## 📊 Model Performance  

| Metric | Score |
|--------|-------|
| **Accuracy** | ~0.92 |
| **Precision** | 0.93 |
| **Recall** | 0.88 |
| **F1-Score** | 0.90 |

*(Exact results may vary per training run.)*

---

## 💾 Saved Files  

models/
├── url_spam_svm.pkl ← trained SVM model
└── tfidf_vectorizer.pkl ← TF-IDF vectorizer



These files allow you to reload the trained model for real-time predictions.

---

## 🔍 Example Usage  

```python
from joblib import load
import re

# Load model and vectorizer
model = load("models/url_spam_svm.pkl")
vectorizer = load("models/tfidf_vectorizer.pkl")

def clean_url(text):
    text = re.sub(r'https?://|www\.', '', str(text))
    text = re.sub(r'[-_/]', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\. ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def predict_spam(url_text):
    cleaned = clean_url(url_text)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    return "🚨 SPAM DETECTED" if pred else "✅ SAFE LINK"

# Example
print(predict_spam("http://paypal-login-secure-update.com"))

Output:

🚨 SPAM DETECTED


url_spam_detector/
│
├── url_spam_detector.ipynb
├── models/
│   ├── url_spam_svm.pkl
│   └── tfidf_vectorizer.pkl
└── README.md



💫 About the Author

👩🏽‍💻 Tenika Powell
Machine Learning Engineer | Data Science & AI Student
📍 Benton Harbor, MI
🌐 GitHub – Nikkilabesf

“Turning curiosity into code and data into power.”

<div align="center">

✨ Built with passion, patience, and teal energy 💎
Made for the journey from Data Science Student → Machine Learning Engineer 🚀

</div> ```

