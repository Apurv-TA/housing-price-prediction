"""Processors for the model training step of the worklow."""
import logging
import os.path as op

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)
from util import _custom_data_transform

logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    # load training datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # load pre-trained feature pipelines and other artifacts
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))

    # sample data if needed. Useful for debugging/profiling purposes.
    sample_frac = params.get("sampling_fraction", None)
    if sample_frac is not None:
        logger.warn(f"The data has been sample by fraction: {sample_frac}")
        sample_X = train_X.sample(frac=sample_frac, random_state=context.random_seed)
    else:
        sample_X = train_X
    sample_y = train_y.loc[sample_X.index]

    # transform the training data
    train_X = get_dataframe(
        features_transformer.fit_transform(train_X, train_y),
        get_feature_names_from_column_transformer(features_transformer),
    )
    train_X = train_X[curated_columns]
    train_X.rename(columns={"ocean_proximity_<1H OCEAN": "ocean_proximity_1H OCEAN"}, inplace=True)

    imp_features = [
        "housing_median_age", "longitude", "latitude",
        "ocean_proximity_NEAR OCEAN", "median_income", "ocean_proximity_INLAND"
    ]

    xgb_training_pipe = Pipeline([
        (
            "", FunctionTransformer(
                _custom_data_transform,
                kw_args={"cols2keep": imp_features}
            )
        ),
        (
            "XGBoost", XGBRegressor(
                gamma=params["xgboost"]["gamma"],
                learning_rate=params["xgboost"]["learning_rate"],
                max_depth=params["xgboost"]["max_depth"],
                min_child_weight=params["xgboost"]["min_child_weight"],
                n_estimators=params["xgboost"]["n_estimators"]
            ))
    ])

    xgb_training_pipe.fit(train_X, train_y)
    save_pipeline(
        xgb_training_pipe, op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
    )
