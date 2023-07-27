"""Module for listing down additional custom functions required for production."""

import pandas as pd
from datetime import datetime


def binned_selling_price(df):
    """Bin the selling price column using quantiles."""
    return pd.qcut(df["unit_price"], q=10)


# to standardize the date format in every dataset
def standardize_date(date_str):
    # List of possible date formats
    date_formats = ["%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%Y-%m-%d"]

    # Iterate through the formats and try to parse the date
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%d-%m-%Y")
        except ValueError:
            pass

    # If no valid format is found, return None or the original string, depending on your preference
    return None
