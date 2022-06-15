"""
Student emails project

Skel at the moment

"""
import pandas as pd
from libEmails import read
from sklearn.feature_extraction.text import CountVectorizer


def main() -> None:
    # Read email bodies into a pandas dataframe
    email_bodies: pd.DataFrame = read.read_email_body()

    # Read the email bodies into a bag of words
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(email_bodies.EmailBody)


if __name__ == "__main__":
    main()
