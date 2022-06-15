"""
Student emails project

Skel at the moment

"""
import pandas as pd
from libEmails import read
from sklearn.feature_extraction.text import TfidfVectorizer


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


if __name__ == "__main__":
    main()
