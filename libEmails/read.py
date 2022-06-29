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
    df: pd.DataFrame,
    header: Union[Tuple[str], str],
    label_header: str,
    min_support: int,
    verbose: bool,
) -> pd.DataFrame:
    """
    Prune the dataframe, removing rows where the specified column heading(s) are NaN and removing rows where the
    demand type is too rare.

    :param df: DataFrame to prune values from
    :param headers: column header(s) to look for NaN values in when removing rows
    :param label_header: column header for class labels
    :param min_support: minimum number of occurrences in the demand type column
    :param verbose: whether to print info about what is going on
    :returns: dataframe with bad values removed

    """
    # Drop any where the provided header line(s) are NaN
    if verbose:
        if not isinstance(header, str):
            for s in header:
                print(f"{s}:\tDropping {df[s].isna().sum()} NaN values of {len(df)}")
        else:
            print(f"Dropping {df[header].isna().sum()} NaN values of {len(df)}")
    df.dropna(inplace=True, subset=header)

    # Get rid of any demand types with too few occurrences
    category_counts = _count_unique(df[label_header])
    pprint(category_counts)

    too_few = {k: v for k, v in category_counts.items() if v < min_support}
    if verbose:
        print(
            f"Following categories will be removed; fewer than the minimum ({min_support}) found:"
        )
        pprint(too_few)
    mask = ~np.logical_or.reduce([df[label_header] == s for s in too_few.keys()])

    return df[mask]


def _vectorise(text: pd.Series) -> Tuple[TfidfVectorizer, csr_matrix]:
    """
    Transform a Series of strings to a sparse matrix of tokens

    :param text: Series of strings to parse
    :returns: the vectorizer object
    :returns: sparse matrix representing a bag of words

    """
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=3,
        norm="l2",
        encoding="latin-1",
        ngram_range=(1, 2),
        stop_words="english",
    )
    bag_of_words = vectorizer.fit_transform(text)

    return vectorizer, bag_of_words


def _train_mask(rng: np.random.Generator, n: int, train_fraction: float) -> np.ndarray:
    """
    Boolean mask of which events to train on

    :param rng: random number generator
    :param n: length of the mask
    :param train_fraction: approximate fraction of the mask that should be True
    :returns: boolean mask telling us which events to train/test on

    """
    return rng.random(n) < train_fraction


def _resample(sampling_strategy: str, X: csr_matrix, y: pd.Series, verbose: bool):
    """
    Resample a data sample

    :param sampling_strategy: which sampling strategy to use: "undersample", "oversample" or "naive"
    :param X: sparse matrix representing our feature space
    :param y: class labels
    :param verbose: whether to print other stuff out too
    """
    if sampling_strategy == "naive":
        if verbose:
            print("No over or undersampling")
        return X, y

    sampling_fcn = _undersample if sampling_strategy == "undersample" else _oversample

    X_resampled, y_resampled = sampling_fcn(X, y)
    if verbose:
        print(f"Label counts in training set after resampling ({sampling_strategy}):")
        pprint(_count_unique(y_resampled))

    return X_resampled, y_resampled


def read_email_body(
    min_support: int = 15,
    verbose: bool = True,
    sampling: str = "naive",
    return_vectorizer=False,
) -> Tuple:
    """
    Read the Excel spreadsheet of emails with email subject lines only

    The path to this file is hard coded
    Drops rows containing NaN email bodies

    Reads the first sheet; uses the first row as column labels; parses all columns

    :param min_support: minimum number of emails with in a given category. Defaults to 15.
    :param verbose: whether to print information about what cuts are being performed.
    :param sampling: sampling strategy; "naive", "undersample" or "oversample".
                     Naive sampling makes no attempt to balance the data.
                     Undersampling resamples all demand types except the minority class.
                     Oversampling oversamples all classes but the majority.

    :returns: tuple: (pandas series of email subject lines, series of demand categories) - training
    :returns: tuple: (pandas series of email subject lines, series of demand categories) - testing
    :returns: if `return_vectorizer is True`, returns also the vectorizer used for fitting/transforming the subjects

    """
    assert sampling in {"naive", "undersample", "oversample"}
    df: pd.DataFrame = pd.read_excel(
        "./data/Anonymised Electrical & Electronic Engineering Email Body Sample 200.xlsx"
    )

    # Replace empty subject lines with empty string
    subject_heading = "Subject"
    df[subject_heading] = df[subject_heading].fillna("")

    # Remove NaN values and rows where the demand type is too rare
    body_heading = "EmailBody"
    category_heading = "Subject Categorisation"
    df = _prune(
        df, [body_heading, subject_heading], category_heading, min_support, verbose
    )

    # Series for our class labels
    labels = df[category_heading]

    # Concatenate the email body and the subject line
    email_bodies = df[body_heading]
    email_subjects = df[subject_heading]
    email_documents = email_subjects + "\n\n" + email_bodies

    # Convert to a bag of words
    vectorizer, bag_of_words = _vectorise(email_subjects)

    # Mask for training
    rng = np.random.default_rng(seed=0)
    train = _train_mask(rng, bag_of_words.shape[0], 0.75)

    # Resample the training data if we need to
    bag_of_words_train, labels_train = _resample(
        sampling, bag_of_words[train], labels[train], verbose
    )

    if not return_vectorizer:
        return (bag_of_words_train, labels_train), (
            bag_of_words[~train],
            labels[~train],
        )
    return (
        (bag_of_words_train, labels_train),
        (bag_of_words[~train], labels[~train]),
        vectorizer,
    )


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

    :returns: tuple: (pandas series of email subject lines, series of demand categories) - training
    :returns: tuple: (pandas series of email subject lines, series of demand categories) - testing
    :returns: if `return_vectorizer is True`, returns also the vectorizer used for fitting/transforming the subjects

    """
    assert sampling in {"naive", "undersample", "oversample"}

    df: pd.DataFrame = pd.read_excel(
        "./data/Anonymised_UG_Economics_Email_Sample_7500_2022-05-26 Strip.xlsx"
    )

    # Remove NaN values and rows where the demand type is too rare
    subject_heading = "AnonSubject"
    category_heading = "Subject category"
    df = _prune(df, subject_heading, category_heading, min_support, verbose)

    # Series for our class labels
    labels = df[category_heading]

    # Convert our subject lines into a bag of words
    subject_lines = df[subject_heading]
    vectorizer, bag_of_words = _vectorise(subject_lines)

    # Mask for training
    rng = np.random.default_rng(seed=0)
    train = _train_mask(rng, bag_of_words.shape[0], 0.75)

    # Resample the training data if we need to
    bag_of_words_train, labels_train = _resample(
        sampling, bag_of_words[train], labels[train], verbose
    )

    if not return_vectorizer:
        return (bag_of_words_train, labels_train), (
            bag_of_words[~train],
            labels[~train],
        )
    return (
        (bag_of_words_train, labels_train),
        (bag_of_words[~train], labels[~train]),
        vectorizer,
    )
