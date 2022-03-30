"""Module for listing down additional custom functions required for the notebooks."""

import pandas as pd
import numpy as np


def binned_median_income(df):
    """Bin the median income column"""
    # return pd.qcut(df["unit_price"], q=10)

    return pd.cut(
        df["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )
