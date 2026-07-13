import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

keep_words = {"not", "no", "nor", "can", "cannot", "won't", "wasn't", "isn't", "don't", "neither", "nah"}

_stopwords = set(stopwords.words("english")) - keep_words
_lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s", " ", text).strip()

    return text

def tokenize(text: str) -> str:
    token = word_tokenize(text)

    return [t for t in token if t not in _stopwords and t not in string.punctuation]

def lemmatize(t):
    lemma = [_lemmatizer.lemmatize(t)]

    return lemma

def rejoin(text: str) -> str:
    cleaned_text = clean_text(text)
    token = tokenize(cleaned_text)
    lemma = lemmatize(token)

    return " ".join(lemma)

