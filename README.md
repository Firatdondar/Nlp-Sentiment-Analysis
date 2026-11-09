# 🧠 NLP Sentiment Analysis & Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-154f3c)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end **Natural Language Processing (NLP)** pipeline in Python that performs **text preprocessing (tokenization, stop-word removal, Porter stemming)**, **unigram/bigram vectorization (`CountVectorizer`)**, and **Multinomial Naive Bayes (`MultinomialNB`)** sentiment classification on e-commerce product reviews, paired with a content-based fallback recommendation trigger.

---

## ✨ Key Features

- **NLP Text Preprocessing:** Tokenizes raw review text, filters English stop-words, and extracts word stems using `NLTK` (`PorterStemmer`).
- **Unigram & Bigram Feature Extraction:** Captures both single words and two-word phrases via `CountVectorizer(ngram_range=(1, 2))`.
- **Probabilistic Sentiment Classification:** Trains a `MultinomialNB` classifier and outputs accuracy, precision, recall, and F1-score metrics.
- **Macro & Micro Sentiment Dashboard:** Computes overall product satisfaction ratios, ranks the top 5 most positive and most negative reviews by AI confidence score, and triggers an alternative product recommendation if satisfaction drops below the 50% threshold.
- **Interactive CLI Testing:** Allows testing custom product reviews in real time from the terminal.

---

## 🗂️ Project Structure

```text
Nlp-Sentiment-Analysis/
├── main.py                  # NLP pipeline, Naive Bayes classifier & interactive CLI
├── reviews.txt              # Labeled product review corpus (100 reviews)
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Firatdondar/Nlp-Sentiment-Analysis.git
   cd Nlp-Sentiment-Analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the sentiment analyzer:**
   ```bash
   python main.py
   ```

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
