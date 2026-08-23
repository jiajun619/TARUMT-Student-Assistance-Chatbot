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


# Common English contractions
contractions = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "shouldn't": "should not",
    "mustn't": "must not",
    "needn't": "need not",
}


keep_words = {
    "not",
    "no",
    "nor",
    "can",
    "cannot",
    "neither",
    "nah",
}


_stopwords = set(stopwords.words("english")) - keep_words
_lemmatizer = WordNetLemmatizer()


def expand_contractions(text: str) -> str:
    for contraction, replacement in contractions.items():
        text = re.sub(
            r"\b" + re.escape(contraction) + r"\b",
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    return text


def clean_text(text: str) -> str:
    text = text.lower()
    text = expand_contractions(text)
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