""" Student emails project

Skel at the moment

"""
import sys
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn import metrics
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from libEmails import read


def main() -> None:
    interactive = "-i" in sys.argv or "--interactive" in sys.argv

    # Kwargs for parsing the data
    kw = {"min_support": 75, "sampling": "undersample"}

    # We need access to the TfidfVectorizer if we want to later run on arbitrary input
    if interactive:
        kw["return_vectorizer"] = True
        X, categories, vectorizer = read.read_email_subjects(**kw)

    else:
        X, categories = read.read_email_subjects(**kw)

    # Mask for training
    rng = np.random.default_rng(seed=0)
    train_fraction = 0.75
    train = rng.random(X.shape[0]) < train_fraction

    # Classifier
    clf = MultinomialNB()
    clf.fit(X[train], categories[train])

    print(metrics.classification_report(categories[train], clf.predict(X[train])))
    print(metrics.classification_report(categories[~train], clf.predict(X[~train])))

    # Do some interactive stuff maybe
    if interactive:
        s = input()
        while s:
            X: sparse.csr_matrix = vectorizer.transform([s])
            (predicted_class,) = clf.predict(X)
            probabilities = clf.predict_proba(X)
            print(f"{predicted_class}\n{probabilities}")
            s = input()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
