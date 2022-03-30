"""Processors for the model scoring/evaluation step of the worklow."""
import os.path as op

from ta_lib.core.api import (
    get_dataframe, get_feature_names_from_column_transformer, load_dataset,
    load_pipeline, register_processor, save_dataset, DEFAULT_ARTIFACTS_PATH
)


@register_processor("model-eval", "score-model")
def score_model(context, params):
    """Score a pre-trained model."""

    input_features_ds = "test/housing/features"
    input_target_ds = "test/housing/target"
    output_ds = "score/housing/output"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load test datasets
    test_X = load_dataset(context, input_features_ds)
    test_y = load_dataset(context, input_target_ds)

    # load the feature pipeline and training pipelines
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))
    model_pipeline = load_pipeline(op.join(artifacts_folder, "train_pipeline.joblib"))

    # transform the test dataset
    test_X = get_dataframe(
        features_transformer.transform(test_X),
        get_feature_names_from_column_transformer(features_transformer),
    )
    test_X = test_X[curated_columns]
    test_X.rename(columns={"ocean_proximity_<1H OCEAN": "ocean_proximity_1H OCEAN"}, inplace=True)

    # make a prediction
    test_X["yhat"] = model_pipeline.predict(test_X)

    # store the predictions for any further processing.
    save_dataset(context, test_X, output_ds)
