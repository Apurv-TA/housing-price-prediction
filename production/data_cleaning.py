"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
)
from scripts import binned_median_income


@register_processor("data-cleaning", "housing")
def clean_housing_table(context, params):
    """Clean the ``HOUSING`` data table.

    """

    input_dataset = "raw/housing"
    output_dataset = "cleaned/housing"

    # load dataset
    housing_df = load_dataset(context, input_dataset)

    housing_df_clean = (
        housing_df
        # generating a copy of the original DataFrame as a backup
        .copy()

        .replace({'': np.NaN})

        # clean column names
        .clean_names(case_type="snake")
    )

    # save the dataset
    save_dataset(context, housing_df_clean, output_dataset)

    return housing_df_clean


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``housing`` table into ``train`` and ``test`` datasets."""

    input_dataset = "cleaned/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"

    # load dataset
    housing_df_processed = load_dataset(context, input_dataset)

    # creating additional features that are not affected by train test split. These are features that are processed globally

    # split the data

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=params["test_size"],
        random_state=context.random_seed
    )
    housing_df_train, housing_df_test = custom_train_test_split(
        housing_df_processed, splitter, by=binned_median_income
    )
    # split train dataset into features and target
    target_col = params["target"]

    train_X, train_y = (
        housing_df_train
        .get_features_targets(target_column_names=target_col)
    )

    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    test_X, test_y = (
        housing_df_test
        .get_features_targets(target_column_names=target_col)
    )

    # save the datasets
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)
