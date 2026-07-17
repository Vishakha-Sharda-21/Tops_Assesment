"""
Data Science • Module 11 – Natural Language Processing (NLP) • M11-A1
Section C — Mini Capstone Project: Food Delivery Review Intelligence System
"""

import collections
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from Task1 import clean_review

# Pre-seeded training data to stabilize the models out of the box
SEED_TRAINING_CORPUS = [
    ("The delivery arrived extremely quickly and the food was hot.", "Positive", "Delivery"),
    ("Super fast courier service, highly recommend this platform driver.", "Positive", "Delivery"),
    ("The chicken burger was delicious, exceptionally fresh, and amazing.", "Positive", "Food Quality"),
    ("Incredible hot pizza meal, excellent quality ingredients used.", "Positive", "Food Quality"),
    ("The mobile system interface design functions flawlessly without bugs.", "Positive", "App"),
    ("Updating my account profile address details was very easy.", "Positive", "App"),
    ("Nice service overall, satisfied with my order transactions.", "Positive", "General"),
    ("Everything was fine, acceptable experience.", "Positive", "General"),
    ("The driver took three hours to arrive and food was freezing.", "Negative", "Delivery"),
    ("Rider spilled all the soup containers inside the delivery bag.", "Negative", "Delivery"),
    ("The steak meat was completely raw, smelly, and rotten.", "Negative", "Food Quality"),
    ("Found long black hair inside my noodle bowl, disgusting.", "Negative", "Food Quality"),
    ("The application keeps crashing during the payment checkout step.", "Negative", "App"),
    ("Promo voucher coupon field returns a network exception bug error.", "Negative", "App"),
    ("Horrible treatment, completely terrible experience from this place.", "Negative", "General"),
    ("Disappointed with everything, will never order again.", "Negative", "General")
]

class ReviewIntelligenceSystem:
    def __init__(self):
        self.user_reviews = []
        self.train_texts = [clean_review(item[0]) for item in SEED_TRAINING_CORPUS]
        self.train_sentiments = [item[1] for item in SEED_TRAINING_CORPUS]
        self.train_categories = [item[2] for item in SEED_TRAINING_CORPUS]
        
        self.sentiment_vectorizer = TfidfVectorizer()
        self.category_vectorizer = TfidfVectorizer()
        self.sentiment_classifier = LogisticRegression(random_state=42)
        self.category_classifier = LogisticRegression(random_state=42)
        self._retrain_models()

    def _retrain_models(self):
        """Retrains models on the running internal database text arrays."""
        X_sent = self.sentiment_vectorizer.fit_transform(self.train_texts)
        self.sentiment_classifier.fit(X_sent, self.train_sentiments)
        X_cat = self.category_vectorizer.fit_transform(self.train_texts)
        self.category_classifier.fit(X_cat, self.train_categories)

    def add_review(self, raw_text: str):
        """Cleans, extracts single string labels via index 0, and indexes entries."""
        cleaned_text = clean_review(raw_text)
        if not cleaned_text.strip():
            return None, None
            
        # [FIXED] Added [0] to extract raw string value instead of bracket string outputs
        pred_sentiment = str(self.sentiment_classifier.predict(self.sentiment_vectorizer.transform([cleaned_text]))[0])
        pred_category = str(self.category_classifier.predict(self.category_vectorizer.transform([cleaned_text]))[0])
        
        self.user_reviews.append({
            'raw': raw_text, 
            'cleaned': cleaned_text, 
            'sentiment': pred_sentiment, 
            'category': pred_category
        })
        
        self.train_texts.append(cleaned_text)
        self.train_sentiments.append(pred_sentiment)
        self.train_categories.append(pred_category)
        self._retrain_models()
        return pred_sentiment, pred_category

    def evaluate_single_text(self, raw_text: str):
        """Evaluates live single entries on-demand without writing to database logs."""
        cleaned_text = clean_review(raw_text)
        if not cleaned_text.strip():
            return "Invalid input", ""
        pred_sentiment = str(self.sentiment_classifier.predict(self.sentiment_vectorizer.transform([cleaned_text]))[0])
        pred_category = str(self.category_classifier.predict(self.category_vectorizer.transform([cleaned_text]))[0])
        return pred_sentiment, pred_category

    def generate_summary_report(self):
        """Aggregates execution scores and compiles running key term counters perfectly."""
        total_user_count = len(self.user_reviews)
        sentiment_counts = {'Positive': 0, 'Negative': 0}
        category_counts = {'Delivery': 0, 'Food Quality': 0, 'App': 0, 'General': 0}
        word_frequency_counter = collections.Counter()
        
        for item in self.user_reviews:
            s_label = item['sentiment']
            c_label = item['category']
            
            # Increment dictionary tracking nodes safely
            sentiment_counts[s_label] = sentiment_counts.get(s_label, 0) + 1
            category_counts[c_label] = category_counts.get(c_label, 0) + 1
            word_frequency_counter.update(item['cleaned'].split())
            
        return total_user_count, sentiment_counts, category_counts, word_frequency_counter.most_common(5)

def main():
    system = ReviewIntelligenceSystem()
    while True:
        print("\n" + "="*60)
        print("         FOOD DELIVERY REVIEW INTELLIGENCE SYSTEM       ")
        print("="*60)
        print("1. Add a New Review to Database Store")
        print("2. Classify a Review (On-Demand Live Analytics)")
        print("3. View System Executive Summary Report")
        print("4. Exit Application Console Engine")
        print("-"*60)
        choice = input("Select an option (1-4): ").strip()
        
        if choice == '1':
            print("\n>>> OPTION 1: ADD NEW CUSTOMER REVIEW")
            review_input = input("Enter the raw customer review text:\n").strip()
            if not review_input:
                print("[ERROR] Input text cannot be completely empty.")
                continue
            sentiment, category = system.add_review(review_input)
            if sentiment is None:
                print("[ERROR] Preprocessing reduced input text to zero features.")
            else:
                print(f"\n[SUCCESS] Review Indexed. \n -> Sentiment: {sentiment} \n -> Category: {category}")
        elif choice == '2':
            print("\n>>> OPTION 2: LIVE ON-DEMAND CLASSIFICATION")
            test_input = input("Enter raw review text:\n").strip()
            if not test_input:
                print("[ERROR] Empty string.")
                continue
            sentiment, category = system.evaluate_single_text(test_input)
            print(f"\n -> Live Sentiment: {sentiment}\n -> Live Category: {category}")
        elif choice == '3':
            print("\n>>> OPTION 3: EXECUTIVE REPORT")
            total, s_counts, c_counts, common_words = system.generate_summary_report()
            print(f" TOTAL USER REVIEWS ADDED : {total}\n" + "-"*50)
            print(" SENTIMENT COUNTS:")
            for sent, count in s_counts.items(): 
                print(f"  * {sent.ljust(15)}: {count}")
            print("\n CATEGORY COUNTS:")
            for cat, count in c_counts.items(): 
                print(f"  * {cat.ljust(15)}: {count}")
            print("\n TOP 5 FREQUENT CONTENT WORDS:")
            if not common_words: 
                print("  (No unique tokens processed yet. Add items in Option 1 first.)")
            else:
                for idx, (word, freq) in enumerate(common_words, 1):
                    print(f"  {idx}. '{word.ljust(12)}' | Count: {freq}")
            print("-" * 50)
        elif choice == '4':
            print("\nShutting down Engine. Goodbye!")
            break
        else:
            print("[INVALID CHOICE] Select 1 to 4.")

if __name__ == "__main__":
    main()
