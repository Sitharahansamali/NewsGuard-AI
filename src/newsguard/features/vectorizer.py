from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer():
    """
    Create the TF-IDF vectorizer used by NewsGuard AI.
    """

    return TfidfVectorizer(
        max_features=10000
    )