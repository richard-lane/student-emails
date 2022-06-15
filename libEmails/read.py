"""
Read emails

"""
import pandas as pd


def read_email_body() -> pd.DataFrame:
    """
    Read the Excel spreadsheet of emails with email bodies

    Replaces NaN email bodies with empty str

    Reads the first sheet
    Uses the first row as column labels
    Parses all columns

    :param path: location of the student emails Excel file
    :returns: dataframe holding all the emails + metadata

    """
    rv = pd.read_excel(
        "./data/Anonymised Electrical & Electronic Engineering Email Body Sample 200.xlsx"
    )
    rv["EmailBody"] = rv["EmailBody"].fillna("")
    return rv
