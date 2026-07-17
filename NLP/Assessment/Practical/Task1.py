"""
Data Science • Module 11 – Natural Language Processing (NLP) • M11-A1
Task 1: Food Review Text Preprocessor
"""

# Hardcoded dictionary of NLTK's English stop words to ensure zero external network dependencies
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', 
    "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def clean_review(review: str) -> str:
    """
    Cleans a raw food review string by lowering case, removing punctuation,
    tokenizing, filtering stopwords, and applying deterministic stemming.
    """
    # 1. Convert text to lowercase
    lowered = review.lower()
    
    # 2. Remove punctuation and strip extra whitespaces
    cleaned_chars = []
    punctuation_str = '!_.,;:??"\'()[]{}|<>/\\@#$%^&*+-='
    for char in lowered:
        if char in punctuation_str:
            cleaned_chars.append(' ')
        else:
            cleaned_chars.append(char)
            
    cleaned_string = "".join(cleaned_chars)
    
    # 3. Tokenise the cleaned text using whitespace split
    tokens = cleaned_string.split()
    
    # 4. Remove English stopwords
    filtered_tokens = [token for token in tokens if token not in STOPWORDS]
    
    # 5. Rule-based stemming execution (simulating Porter Stemmer suffix reduction)
    stemmed_tokens = []
    for token in filtered_tokens:
        if token.endswith('ies') and not token.endswith('eies'):
            stemmed_tokens.append(token[:-3] + 'i')
        elif token.endswith('es') and not token.endswith('ees') and not token.endswith('aes'):
            stemmed_tokens.append(token[:-2])
        elif token.endswith('s') and not token.endswith('ss') and not token.endswith('us'):
            stemmed_tokens.append(token[:-1])
        elif token.endswith('ing'):
            stemmed_tokens.append(token[:-3])
        elif token.endswith('ed'):
            stemmed_tokens.append(token[:-2])
        elif token.endswith('ly'):
            stemmed_tokens.append(token[:-2])
        else:
            stemmed_tokens.append(token)
            
    return " ".join(stemmed_tokens)

if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING TASK 1: FOOD REVIEW TEXT PREPROCESSOR")
    print("=" * 80)

    # Test samples representing three different food delivery review contexts
    sample_reviews = [
        "The cheesy pizza arrived incredibly late, and it was completely freezing cold!",
        "Wow, this spicy butter chicken with hot garlic naan was absolutely amazing and delicious.",
        "The application crashed twice while I was attempting to update my delivery address."
    ]

    for idx, sample in enumerate(sample_reviews, 1):
        print(f"\n[Sample {idx}] Raw text: '{sample}'")
        print(f"[Sample {idx}] Processed text: '{clean_review(sample)}'")
