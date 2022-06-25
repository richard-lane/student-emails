"""
Student emails project

Skel at the moment

"""
import sys
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn import metrics
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB

from libEmails import read


def main() -> None:
    # Read email subjects and demand categories into pandas series
    X, categories = read.read_email_subjects(min_support=75, sampling="undersample")

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
    if "-i" in sys.argv or "--interactive" in sys.argv:
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
