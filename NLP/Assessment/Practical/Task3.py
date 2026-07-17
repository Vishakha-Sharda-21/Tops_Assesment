"""
Data Science • Module 11 – Natural Language Processing (NLP) • M11-A1
Task 3: Complaint Category Classifier – Naive Bayes
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Import cleaning module pipeline logic from Task 1 file
from Task1 import clean_review

if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING TASK 3: COMPLAINT CATEGORY CLASSIFIER – NAIVE BAYES")
    print("=" * 80)

    # Balanced dataset of 24 complaint strings mapped across three classes
    complaint_data = [
        # Delivery Issue Class
        ("The courier took two hours to deliver my breakfast package.", "Delivery"),
        ("Driver dropped my food bag on the floor and ran away.", "Delivery"),
        ("My delivery tracker shows arrived but no rider is here.", "Delivery"),
        ("The delivery boy was incredibly rude and delivered to the wrong house.", "Delivery"),
        ("Order is missing half of the items during transit delivery.", "Delivery"),
        ("Rider is stuck in traffic and my dinner is delayed by hours.", "Delivery"),
        ("The courier tracking app shows he went the wrong direction.", "Delivery"),
        ("Package was completely smashed by the bike rider during delivery.", "Delivery"),
        
        # Food Quality Class
        ("The burger meat was entirely raw, pink, and cold inside.", "Food Quality"),
        ("Soup spilled everywhere and the noodles were completely soggy.", "Food Quality"),
        ("This pizza tastes stale, old, and completely lacking cheese.", "Food Quality"),
        ("Found a hair strand inside my chicken noodle soup bowl.", "Food Quality"),
        ("The rice is severely overcooked and tastes like plastic mush.", "Food Quality"),
        ("Sushi does not smell fresh at all, it looks rotten.", "Food Quality"),
        ("The beverage was warm and the French fries were totally oily.", "Food Quality"),
        ("Extremely salty curry, it is completely inedible and burned.", "Food Quality"),
        
        # App Issue Class
        ("The checkout screen freezes every single time I try to add a card.", "App"),
        ("Promo code is valid but the application says it expired.", "App"),
        ("I cannot update my GPS location address inside the user profile.", "App"),
        ("The mobile application crashed during processing payment transaction.", "App"),
        ("Cannot login via Google validation link, screen stays white.", "App"),
        ("Notification settings do not save on this buggy application build.", "App"),
        ("My digital wallet balance shows zero after a successful top up.", "App"),
        ("The checkout page displays a network timeout processing exception.", "App")
    ]

    complaints, complaint_labels = zip(*complaint_data)

    # Map preprocessor transforms over raw string values
    preprocessed_complaints = [clean_review(comp) for comp in complaints]

    # Initialize model structures to prepare validation vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_complaints)
    y = np.array(complaint_labels)

    # Perform structural split configurations using random seeds for consistency
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Implement Multinomial Naive Bayes classification logic
    nb_classifier = MultinomialNB(alpha=1.0)
    nb_classifier.fit(X_train, y_train)

    # Generate model evaluations metrics over held out rows
    y_pred = nb_classifier.predict(X_test)

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=['App', 'Delivery', 'Food Quality']))

    print("--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_test, y_pred, labels=['App', 'Delivery', 'Food Quality']))
