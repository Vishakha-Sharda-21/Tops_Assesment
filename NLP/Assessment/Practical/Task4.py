"""
Data Science • Module 11 – Natural Language Processing (NLP) • M11-A1
Task 4: Sentiment Analysis Pipeline – Model Comparison
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import cleaning module pipeline logic from Task 1 file
from Task1 import clean_review

if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING TASK 4: SENTIMENT ANALYSIS PIPELINE – MODEL COMPARISON")
    print("=" * 80)

    # Dataset consisting of 30 food delivery reviews with Positive and Negative metrics
    sentiment_data = [
        ("The hot meals arrived extremely fast and tasted absolutely wonderful.", "Positive"),
        ("Incredible seasoning, best chicken tacos I have ever eaten.", "Positive"),
        ("Super quick delivery service, pizza was steaming hot and delicious.", "Positive"),
        ("The desserts were spectacular and packaged beautifully to avoid leaks.", "Positive"),
        ("Very polite rider, food arrived ten minutes earlier than expected.", "Positive"),
        ("Amazing food quality, fresh ingredients used for the green salad.", "Positive"),
        ("Perfect lunch experience, everything was exceptionally clean and tasty.", "Positive"),
        ("Highly recommend this restaurant, large portion sizes and great flavor.", "Positive"),
        ("Loved the rich creamy sauce on the pasta, truly brilliant preparation.", "Positive"),
        ("The steak was cooked perfectly to medium rare, very juicy.", "Positive"),
        ("The iced tea was perfectly sweetened and chilled nicely.", "Positive"),
        ("Wow, fastest delivery app on the market, absolutely flawless.", "Positive"),
        ("The burgers were thick, savory, and perfectly grilled.", "Positive"),
        ("Outstanding service, they even included a free complimentary cookie.", "Positive"),
        ("Crispy fries and excellent packaging kept everything clean.", "Positive"),
        ("The chicken was completely dry, completely tasteless, and arrived cold.", "Negative"),
        ("Terrible service, waited over two hours for cold soggy pizza.", "Negative"),
        ("The noodle soup spilled everywhere inside the delivery container bag.", "Negative"),
        ("Horrible stale bread, it was completely hard and impossible to chew.", "Negative"),
        ("Rider was extremely rude and refused to bring the food upstairs.", "Negative"),
        ("Found small pieces of hair inside my white rice portion bowl.", "Negative"),
        ("The food tasted like old oil, completely disgusted with this meal.", "Negative"),
        ("Application glitched out and charged my bank card twice for one order.", "Negative"),
        ("The beverage was missing from my bag and the fries were old.", "Negative"),
        ("Completely wrong items delivered, ordered sushi but received hot dog.", "Negative"),
        ("The meat smelled entirely rotten and went straight into the trash.", "Negative"),
        ("Extremely small portion sizes for such a highly expensive menu item.", "Negative"),
        ("The curry was overwhelmingly salty and caused immediate stomach ache.", "Negative"),
        ("Delayed for ninety minutes without any live status tracking map updates.", "Negative"),
        ("Soggy dynamic tacos, everything fell apart completely inside the wrapper.", "Negative")
    ]

    reviews, labels = zip(*sentiment_data)

    # Process raw reviews corpus through structural Task 1 preprocessor pipelines
    preprocessed_reviews = [clean_review(r) for r in reviews]
    X = preprocessed_reviews
    y = np.array(labels)

    # Stratified validation splits configurations initialization setup logic
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Build individual scikit-learn pipeline workflows objects structures patterns
    pipeline_nb = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])

    pipeline_lr = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(random_state=42))
    ])

    # Fit data configurations parameters down pipeline funnels safely
    pipeline_nb.fit(X_train, y_train)
    pipeline_lr.fit(X_train, y_train)

    # Pull out predicted evaluation matrices over test labels datasets arrays
    y_pred_nb = pipeline_nb.predict(X_test)
    y_pred_lr = pipeline_lr.predict(X_test)

    # Collate performance scores parameters side by side inside dictionaries structures
    metrics_summary = {
        'Model': ['Naive Bayes', 'Logistic Regression'],
        'Accuracy': [accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_lr)],
        'Precision': [precision_score(y_test, y_pred_nb, pos_label='Positive'), precision_score(y_test, y_pred_lr, pos_label='Positive')],
        'Recall': [recall_score(y_test, y_pred_nb, pos_label='Positive'), recall_score(y_test, y_pred_lr, pos_label='Positive')],
        'F1-Score': [f1_score(y_test, y_pred_nb, pos_label='Positive'), f1_score(y_test, y_pred_lr, pos_label='Positive')]
    }

    df_metrics = pd.DataFrame(metrics_summary)
    print("\n--- SIDE-BY-SIDE MODEL COMPARISON TABLE ---")
    print(df_metrics.to_string(index=False))

    # DEPLOYMENT DECISION STATEMENT COMMENT FLAG RULE COMPLIANCE REQUIREMENT
    # I would deploy the Logistic Regression pipeline because it achieves optimal or balanced F1-scores on text classification tasks without making naive feature-independence assumptions, scaling effectively as sample parameters expand.
