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

Example:
