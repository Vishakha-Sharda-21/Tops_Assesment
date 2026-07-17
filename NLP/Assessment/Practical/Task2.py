"""
Data Science • Module 11 – Natural Language Processing (NLP) • M11-A1
Task 2: TF-IDF Vectoriser for Menu Reviews
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Import cleaning module pipeline logic from Task 1 file
from Task1 import clean_review

if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING TASK 2: TF-IDF VECTORISER FOR MENU REVIEWS")
    print("=" * 80)

    # Corpus of 8 short food delivery review strings (mixed tones)
    menu_corpus = [
        "The spicy chicken wings were extremely delicious and hot.",
        "Cold oily burgers and soggy fries arrived an hour late.",
        "The food was average, nothing special but delivery was fine.",
        "Incredible sweet dessert, the chocolate cake melted in my mouth.",
        "The sushi was completely raw, smelly, and absolutely terrible.",
        "Mild taste pepper pasta but the cheese portions were small.",
        "Fast delivery of warm crispy tacos, highly recommended food.",
        "Horrible stale bread and overcooked dry chicken meat."
    ]

    # Process individual lines using baseline clean module pipelines
    preprocessed_menu_corpus = [clean_review(rev) for rev in menu_corpus]

    # Instantiate and fit scikit-learn TfidfVectorizer configuration parameters
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_menu_corpus)
    feature_names = vectorizer.get_feature_names_out()

    # Build labeled Pandas DataFrame structural display output format matrices
    row_labels = [f"Review_{i}" for i in range(len(menu_corpus))]
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=row_labels)
    
    print("\n--- FULL TF-IDF MATRIX DATAFRAME ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_tfidf)

    print("\n--- TOP 3 INFORMATIVE WORDS PER DOCUMENT ---")
    for doc_idx, row in enumerate(tfidf_matrix.toarray()):
        # Map out array sequences to display metrics summaries side by side
        word_score_pairs = list(zip(feature_names, row))
        sorted_pairs = sorted(word_score_pairs, key=lambda x: x[1], reverse=True)
        
        # Display top tokens with structural matrix scores higher than 0.00
        top_3 = [f"{w} ({s:.4f})" for w, s in sorted_pairs if s > 0][:3]
        print(f"Review {doc_idx}: {', '.join(top_3)}")
