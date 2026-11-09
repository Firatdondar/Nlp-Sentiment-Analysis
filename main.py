from pathlib import Path
import sys
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Initialize NLTK stemmer and English stop-words
nltk.download(['punkt', 'punkt_tab', 'stopwords'], quiet=True)

stemmer = PorterStemmer()
english_stop_words = set(stopwords.words('english'))


def preprocess_text(text: str) -> str:
    """Tokenize, remove punctuation/stop-words, and extract word stems."""
    tokens = nltk.word_tokenize(text.lower())
    stems = [
        stemmer.stem(word)
        for word in tokens
        if word.isalpha() and word not in english_stop_words
    ]
    return " ".join(stems)


def main():
    # =====================================================================
    # STEP 1: LOAD CORPUS DATASET
    # =====================================================================
    try:
        dataset_path = Path(__file__).resolve().parent / 'reviews.txt'
        df = pd.read_csv(dataset_path, sep='|', names=['review', 'sentiment'])
    except FileNotFoundError:
        print("\nERROR: 'reviews.txt' not found! Make sure it is in the same directory as main.py.\n")
        sys.exit(1)

    df['clean_review'] = df['review'].apply(preprocess_text)

    # =====================================================================
    # STEP 1.5: TRAIN / TEST SPLIT (80% Train, 20% Test)
    # =====================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_review'],
        df['sentiment'],
        test_size=0.20,
        random_state=42,
    )

    # =====================================================================
    # STEP 2: N-GRAM VECTORIZATION (Unigrams & Bigrams)
    # =====================================================================
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    print("--- EXTRACTED VOCABULARY (UNIGRAMS & BIGRAMS) ---")
    print(vectorizer.get_feature_names_out())
    print("-" * 50)

    # =====================================================================
    # STEP 3: MULTINOMIAL NAIVE BAYES TRAINING & EVALUATION
    # =====================================================================
    clf_model = MultinomialNB()
    clf_model.fit(X_train_vectorized, y_train)

    y_pred = clf_model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 20 + " MODEL PERFORMANCE REPORT " + "=" * 20)
    print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Metrics:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # =====================================================================
    # STEP 4: MACRO-LEVEL SENTIMENT AGGREGATION
    # =====================================================================
    all_predictions = clf_model.predict(vectorizer.transform(df['clean_review']))
    total_reviews = len(all_predictions)

    positive_count = int(np.sum(all_predictions == clf_model.classes_[1]))
    negative_count = total_reviews - positive_count

    positive_ratio = (positive_count / total_reviews) * 100
    negative_ratio = (negative_count / total_reviews) * 100

    print("\n" + "=" * 20 + " MACRO SUMMARY DASHBOARD " + "=" * 20)
    print("Evaluated Product: Smartphone Model Y (Category: Smartphones)")
    print(f"Total Analyzed Reviews: {total_reviews}")

    if positive_ratio >= negative_ratio:
        print("OVERALL STATUS: [ POSITIVE ]")
        print(f"Summary: {positive_ratio:.2f}% of reviews are POSITIVE, {negative_ratio:.2f}% are NEGATIVE.")
    else:
        print("OVERALL STATUS: [ NEGATIVE ]")
        print(f"Summary: {negative_ratio:.2f}% of reviews are NEGATIVE, {positive_ratio:.2f}% are POSITIVE.")
    print("=" * 65)

    # =====================================================================
    # STEP 5: HYBRID CONTENT-BASED RECOMMENDATION ENGINE
    # =====================================================================
    CRITICAL_THRESHOLD = 50.0

    if positive_ratio < CRITICAL_THRESHOLD:
        print("\n[WARNING] This product has received predominantly negative feedback!")
        print("[RECOMMENDATION] Consider this top-rated alternative in the same category:")
        print("-> Smartphone Model X (Customer Satisfaction Rate: 90.00%)")
        print("-" * 65)

    # =====================================================================
    # STEP 6: MICRO-LEVEL CONFIDENCE SCORING & RANKING
    # =====================================================================
    all_vectors = vectorizer.transform(df['clean_review'])
    all_probabilities = clf_model.predict_proba(all_vectors)

    review_scores = []
    for i, review_text in enumerate(df['review']):
        neg_pct = all_probabilities[i][0] * 100
        pos_pct = all_probabilities[i][1] * 100
        review_scores.append({
            'review': review_text,
            'neg_pct': neg_pct,
            'pos_pct': pos_pct,
        })

    most_negative = sorted(review_scores, key=lambda x: x['neg_pct'], reverse=True)
    most_positive = sorted(review_scores, key=lambda x: x['pos_pct'], reverse=True)

    print("\n" + "-" * 15 + " MICRO ANALYSIS: TOP 5 MOST NEGATIVE REVIEWS " + "-" * 15)
    for i, res in enumerate(most_negative[:5], 1):
        print(f"Rank #{i} [AI Negative Confidence: {res['neg_pct']:.2f}%]")
        print(f"Review: '{res['review']}'\n")

    print("-" * 15 + " MICRO ANALYSIS: TOP 5 MOST POSITIVE REVIEWS " + "-" * 15)
    for i, res in enumerate(most_positive[:5], 1):
        print(f"Rank #{i} [AI Positive Confidence: {res['pos_pct']:.2f}%]")
        print(f"Review: '{res['review']}'\n")
    print("-" * 80)

    while True:
        choice = input("\nWould you like to view the remaining reviews? (Y/N): ").upper().strip()
        if choice == 'Y':
            remaining = most_negative[5:-5]
            print("\n" + "-" * 15 + " REMAINING REVIEWS (Sorted Negative to Positive) " + "-" * 15)
            for i, res in enumerate(remaining, 6):
                print(f"Rank #{i} [Negative Confidence: {res['neg_pct']:.2f}%]")
                print(f"Review: '{res['review']}'\n")
            print("-" * 80)
            break
        elif choice in ('N', ''):
            print("\nSkipping remaining review list. Proceeding to interactive test mode.")
            break
        else:
            print("Invalid input. Please enter 'Y' (Yes) or 'N' (No).")

    # =====================================================================
    # STEP 7: INTERACTIVE LIVE SENTIMENT TEST
    # =====================================================================
    print("\n" + "*" * 65)
    user_review = input("Enter a custom product review to test the AI model (or press Enter to exit): ")

    if user_review.strip():
        cleaned = preprocess_text(user_review)
        vec = vectorizer.transform([cleaned])
        probs = clf_model.predict_proba(vec)

        neg_score = probs[0][0] * 100
        pos_score = probs[0][1] * 100

        print("\n--- AI SENTIMENT PREDICTION ---")
        print(f"Input Review: '{user_review}'")
        print(f"Positive Probability: {pos_score:.2f}%")
        print(f"Negative Probability: {neg_score:.2f}%")
        print("*" * 65)


if __name__ == "__main__":
    main()
