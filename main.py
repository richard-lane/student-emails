"""
Student emails project

Skel at the moment

"""
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier

from libEmails import read


def main() -> None:
    # Read email bodies into a pandas dataframe
    email_bodies: pd.DataFrame = read.read_email_body()

    # Read the email bodies into a bag of words
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(email_bodies["EmailBody"])

    # Find what subject categories there are
    subject_cats = set(email_bodies["Subject Categorisation"])

    # Map these (deterministically) onto ints
    # Convert to a list so we can sort it
    # Sort it so that the indices map onto alphabetical order
    subject_cats = {v: k for k, v in enumerate(sorted(list(subject_cats)))}

    # Replace str repr of subject cat with int
    email_bodies["Subject Categorisation"].replace(subject_cats, inplace=True)

    categories = email_bodies["Subject Categorisation"].to_numpy()

    # Classifier
    clf = RidgeClassifier()
    clf.fit(X, categories)

    print(clf.predict(X))

    score = metrics.f1_score(categories, clf.predict(X))
    print(f"F1 score: {score}")


if __name__ == "__main__":
    main()
