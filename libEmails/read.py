"""
Read emails

"""
import pandas as pd
import numpy as np
from typing import Tuple, Union
from pprint import pprint
from scipy.sparse import csr_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer


def _count_unique(series: pd.Series) -> dict:
    """
    Count unique occurrences in a pandas Series

    :param series: the series to find unique counts
    :returns: dict of {value: count} for each unique value in series

    """
    return dict(zip(*np.unique(series, return_counts=True)))


def _undersample(
    bag_of_words: csr_matrix, class_labels: pd.Series
) -> Tuple[csr_matrix, pd.Series]:
    """
    Undersample all classes but the minority in order to have a balanced classification set

    :param bag_of_words: bag of words to undersample.
    :param class_labels: class labels used for classification
    :returns: bag of words with non-minority subject categories undersampled to have a balanced set

    """
    sampler = RandomUnderSampler(random_state=0)
    return sampler.fit_resample(bag_of_words, class_labels)


def _oversample(
    bag_of_words: csr_matrix, class_labels: pd.Series
) -> Tuple[csr_matrix, pd.Series]:
    """
    Oversample all classes but the majority in order to have a balanced classification set
    Uses SMOTE to oversample all classes but the majority.

    :param df: pandas dataframe to oversample.
               Must have a column titled "Subject category"; this is the class used in the classification
    :returns: dataframe with non-majority subject categories oversampled to have a balanced set

    """
    sampler = RandomOverSampler(random_state=0)
    return sampler.fit_resample(bag_of_words, class_labels)


def _prune(
    df: pd.DataFrame, header: Union[Tuple[str], str], min_support: int, verbose: bool
) -> pd.DataFrame:
    """
    Prune the dataframe, removing rows where the specified column heading(s) are NaN and removing rows where the
    demand type is too rare.

    :param df: DataFrame to prune values from
    :param headers: column header(s) to look for NaN values in when removing rows
    :param min_support: minimum number of occurrences in the demand type column
    :param verbose: whether to print info about what is going on
    :returns: dataframe with bad values removed

    """
    # Drop any where the subject line is NaN
    if verbose:
        if not isinstance(header, str):
            for s in header:
                print(f"{s}:\tDropping {df[s].isna().sum()} NaN values of {len(df)}")
        else:
            print(f"Dropping {df[header].isna().sum()} NaN values of {len(df)}")
    df.dropna(inplace=True, subset=header)

    # Get rid of any demand types with too few occurrences
    category_heading = "Subject category"
    category_counts = _count_unique(df[category_heading])

    too_few = {k: v for k, v in category_counts.items() if v < min_support}
    if verbose:
        print(
            f"Following categories will be removed; fewer than the minimum ({min_support}) found:"
        )
        pprint(too_few)
    mask = ~np.logical_or.reduce([df[category_heading] == s for s in too_few.keys()])

    return df[mask]


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
    min_support: int = 15,
    verbose: bool = True,
    sampling: str = "naive",
    return_vectorizer=False,
) -> Tuple:
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
                     Oversampling oversamples all classes but the majority.

    :returns: pandas series of email subject lines
    :returns: pandas series of email demand categories
    :returns: if `return_vectorizer is True`, returns also the vectorizer used for fitting/transforming the subjects

    """
    assert sampling in {"naive", "undersample", "oversample"}

    df: pd.DataFrame = pd.read_excel(
        "./data/Anonymised_UG_Economics_Email_Sample_7500_2022-05-26 Strip.xlsx"
    )

    # Remove NaN values and rows where the demand type is too rare
    subject_heading = "AnonSubject"
    df = _prune(df, subject_heading, min_support, verbose)

    # Series for our class labels
    category_heading = "Subject category"
    labels = df[category_heading]

    # Convert our subject lines into a bag of words
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=3,
        norm="l2",
        encoding="latin-1",
        ngram_range=(1, 2),
        stop_words="english",
    )
    bag_of_words: csr_matrix = vectorizer.fit_transform(df[subject_heading])

    # Resample the data if we need to
    if sampling == "undersample":
        bag_of_words, labels = _undersample(bag_of_words, labels)
        if verbose:
            print("Label counts after under sampling:")
            pprint(_count_unique(labels))

    elif sampling == "oversample":
        bag_of_words, labels = _oversample(bag_of_words, labels)
        if verbose:
            print("Label counts after under sampling:")
            pprint(_count_unique(labels))

    if not return_vectorizer:
        return bag_of_words, labels
    return bag_of_words, labels, vectorizer
