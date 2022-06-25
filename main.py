""" Student emails project

Skel at the moment

"""
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn import metrics
from scipy import sparse
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from libEmails import read


def main(args: argparse.Namespace) -> None:
    rv = read.read_email_subjects(
        args.min,
        args.verbose,
        args.sampling,
        args.interactive,
    )

    # We need access to the TfidfVectorizer if we want to later run on arbitrary input
    if args.interactive:
        (X_train, categories_train), (X_test, categories_test), vectorizer = rv
    else:
        (X_train, categories_train), (X_test, categories_test) = rv

    # Classifier
    clf = MultinomialNB()
    clf.fit(X_train, categories_train)

    print(metrics.classification_report(categories_train, clf.predict(X_train)))
    print(metrics.classification_report(categories_test, clf.predict(X_test)))

    cv_train = cross_validate(
        clf,
        X_train,
        categories_train,
        scoring="balanced_accuracy",
        return_train_score=True,
        return_estimator=True,
    )
    cv_test = cross_validate(
        clf,
        X_test,
        categories_test,
        scoring="balanced_accuracy",
        return_train_score=True,
        return_estimator=True,
    )
    print(
        f"Train: balanced accuracy {cv_train['test_score'].mean():.4f}+-{cv_train['test_score'].std():.4f}"
    )
    print(
        f"Test: balanced accuracy {cv_test['test_score'].mean():.4f}+-{cv_test['test_score'].std():.4f}"
    )

    # Do some interactive stuff maybe
    if args.interactive:
        s: str = input()
        while s:
            X_user: sparse.csr_matrix = vectorizer.transform([s])
            (predicted_class,) = clf.predict(X_user)
            probabilities = clf.predict_proba(X_user)
            print(f"{predicted_class}\n{probabilities}")
            s = input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP for student email project")

    # Kwargs for parsing the data
    parser.add_argument(
        "-m",
        "--min",
        help="classifier will only be trained on labels with more counts than this minimum",
        type=int,
        default=75,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Extra print output when reading stuff",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-i",
        "--interactive",
        help="whether to do wait for user input at the end for testing arbitrary strings",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--sampling",
        help="whether to undersample, oversample or use naive sampling",
        default="naive",
        choices={"undersample", "oversample", "naive"},
    )
    try:
        main(parser.parse_args())
    except (KeyboardInterrupt, EOFError):
        pass
