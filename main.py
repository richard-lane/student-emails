"""
Student emails project

Skel at the moment

"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB

from libEmails import read


def main() -> None:
    # Read email bodies into a pandas dataframe
    df: pd.DataFrame = read.read_email_body()

    # Read the email bodies into a bag of words
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=3,
        norm="l2",
        encoding="latin-1",
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(df["EmailBody"].to_numpy())

    # Find what subject categories there are
    subject_cats = set(df["Subject Categorisation"])

    # Map these (deterministically) onto ints
    # Convert to a list so we can sort it
    # Sort it so that the indices map onto alphabetical order
    subject_cats = {v: k for k, v in enumerate(sorted(list(subject_cats)))}
    print(subject_cats)

    # Replace str repr of subject cat with int
    df["Subject Categorisation"].replace(subject_cats, inplace=True)

    categories = df["Subject Categorisation"].to_numpy()

    # Classifier
    clf = MultinomialNB()
    clf.fit(X, categories)

    print(metrics.classification_report(categories, clf.predict(X)))


if __name__ == "__main__":
    main()
