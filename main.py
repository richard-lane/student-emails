"""
Student emails project

Skel at the moment

"""
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn import metrics
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from libEmails import read


def main() -> None:
    # Read email subjects and demand categories into pandas series
    subjects, categories = read.read_email_subjects()

    # Mask for training
    rng = np.random.default_rng(seed=0)
    train_fraction = 0.75
    train = rng.random(len(subjects)) < train_fraction

    # Read the email bodies into a bag of words
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=3,
        norm="l2",
        encoding="latin-1",
        ngram_range=(1, 2),
        stop_words="english",
    )
    X: sparse.csr_matrix = vectorizer.fit_transform(subjects)

    # Classifier
    clf = MultinomialNB()
    clf.fit(X[train], categories[train])

    print(metrics.classification_report(categories[train], clf.predict(X[train])))
    print(metrics.classification_report(categories[~train], clf.predict(X[~train])))


if __name__ == "__main__":
    main()
