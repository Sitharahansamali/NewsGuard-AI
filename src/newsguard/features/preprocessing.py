import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def setup_nltk():
    """Download required NLTK resources."""

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)


setup_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean news article text before feature extraction.

    Parameters
    ----------
    text : str
        Raw news article text.

    Returns
    -------
    str
        Cleaned text.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Keep alphabetic characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)