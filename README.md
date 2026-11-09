# 🧠 NLP Sentiment Analysis & Recommendation Engine

An end-to-end **Natural Language Processing (NLP)** pipeline in Python that performs **text preprocessing (tokenization, stop-word removal, Porter stemming)**, **unigram/bigram vectorization (`CountVectorizer`)**, and **Multinomial Naive Bayes (`MultinomialNB`)** sentiment classification on e-commerce product reviews, paired with a content-based fallback recommendation trigger.

## ✨ Key Features

- **NLP Text Preprocessing:** Tokenizes raw review text, filters English stop-words, and extracts word stems using `NLTK` (`PorterStemmer`).
- **Unigram & Bigram Feature Extraction:** Captures both single words and two-word phrases via `CountVectorizer(ngram_range=(1, 2))`.
- **Probabilistic Sentiment Classification:** Trains a `MultinomialNB` classifier and outputs accuracy, precision, recall, and F1-score metrics.
- **Macro & Micro Sentiment Dashboard:** Computes overall product satisfaction ratios, ranks the top 5 most positive and most negative reviews by AI confidence score, and triggers an alternative product recommendation if satisfaction drops below the 50% threshold.
- **Interactive CLI Testing:** Allows testing custom product reviews in real time from the terminal.

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
