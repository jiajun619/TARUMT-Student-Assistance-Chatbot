import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)


keep_words = {
    "not",
    "no",
    "nor",
    "can",
    "cannot",
    "won't",
    "wasn't",
    "isn't",
    "don't",
    "neither",
    "nah",
}

_stopwords = set(stopwords.words("english")) - keep_words
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text)

    return [
        token
        for token in tokens
        if token not in _stopwords
        and token not in string.punctuation
    ]


def lemmatize(tokens: list[str]) -> list[str]:
    return [
        _lemmatizer.lemmatize(token)
        for token in tokens
    ]


def rejoin(text: str) -> str:
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    lemmas = lemmatize(tokens)

    return " ".join(lemmas)