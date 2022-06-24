"""
Student emails project

Skel at the moment

"""
import pandas as pd
from typing import Tuple
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from libEmails import read


def main() -> None:
    # Read email subjects and demand categories into pandas series
    subjects, categories = read.read_email_subjects()

    # Read the email bodies into a bag of words
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=3,
        norm="l2",
        encoding="latin-1",
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(subjects)

    # Classifier
    clf = MultinomialNB()
    clf.fit(X, categories)

    print(metrics.classification_report(categories, clf.predict(X)))


if __name__ == "__main__":
    main()
