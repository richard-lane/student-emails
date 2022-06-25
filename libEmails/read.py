"""
Read emails

"""
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


def read_email_body() -> pd.DataFrame:
    """
    Read the Excel spreadsheet of emails with email bodies

    The path to this file is hard coded
    Drops rows containing NaN email bodies

    Reads the first sheet
    Uses the first row as column labels
    Parses all columns

    :returns: dataframe holding all the emails + metadata

    """
    rv = pd.read_excel(
        "./data/Anonymised Electrical & Electronic Engineering Email Body Sample 200.xlsx"
    )

    rv.dropna(inplace=True, subset="EmailBody")

    return rv


def read_email_subjects(
    min_support: int = 15, verbose: bool = True, sampling: str = "naive"
) -> Tuple[pd.Series, pd.Series]:
    """
    Read the Excel spreadsheet of emails with email subject lines only

    The path to this file is hard coded
    Drops rows containing NaN subject lines

    Reads the first sheet; uses the first row as column labels; parses all columns

    :param min_support: minimum number of emails with in a given category. Defaults to 15.
    :param verbose: whether to print information about what cuts are being performed.
    :param sampling: sampling strategy; "naive", "undersample" or "oversample".
                     Naive sampling makes no attempt to balance the data.
                     Undersampling resamples all demand types except the minority class.
                     Oversampling uses SMOTE to oversample all classes but the majority.

    :returns: pandas series of email subject lines
    :returns: pandas series of email demand categories

    """
    assert sampling in {"naive", "undersample", "oversample"}

    df: pd.DataFrame = pd.read_excel(
        "./data/Anonymised_UG_Economics_Email_Sample_7500_2022-05-26 Strip.xlsx"
    )

    # Drop any where the subject line is NaN
    subject_heading = "AnonSubject"
    if verbose:
        print(f"Dropping {df[subject_heading].isna().sum()} NaN values of {len(df)}")
    df.dropna(inplace=True, subset=subject_heading)

    # Get rid of any demand types with too few occurrences
    category_heading = "Subject category"
    unique_categories = set(df[category_heading])
    category_counts = {k: np.sum(df[category_heading] == k) for k in unique_categories}

    too_few = {k: v for k, v in category_counts.items() if v < min_support}
    if verbose:
        print(
            f"Following categories will be removed; fewer than the minimum ({min_support}) found:"
        )
        for k, v in too_few.items():
            n_spaces = 30 - len(k)
            print(f"\t{k}:{' ' * n_spaces}{v}")
    mask = ~np.logical_or.reduce([df[category_heading] == s for s in too_few.keys()])
    df = df[mask]

    # Series for our class labels
    labels = df[category_heading]

    # Convert our subject lines into a bag of words
    subjects = df[subject_heading]
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=3,
        norm="l2",
        encoding="latin-1",
        ngram_range=(1, 2),
        stop_words="english",
    )
    bag_of_words: sparse.csr_matrix = vectorizer.fit_transform(subjects)

    return bag_of_words, labels
