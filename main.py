import os

import numpy as np
import pandas as pd
import seaborn as sn

import tensorflow as tf

from ydata_profiling import ProfileReport
import ydf

from src.utils.feature_synthesis import feature_synthesis
from src.utils.imputation import age_imputation
from src.utils.preprocessing import preprocess_pipeline

pd.set_option('future.no_silent_downcasting', True)


def prediction_to_kaggle_format(model, df, threshold=0.5):
    proba_survive = model.predict(df)
    return pd.DataFrame({
        "PassengerId": df["PassengerId"],
        "Survived": (proba_survive >= threshold).astype(int)
    })


def make_submission(kaggle_predictions):
    path = "./data/submission.csv"
    kaggle_predictions.to_csv(path, index=False)
    print(f"Submission exported to {path}")


if __name__ == "__main__":
    train_df = pd.read_csv("./data/train.csv")
    train_df = preprocess_pipeline(train_df)
    train_df, _ = age_imputation(train_df)
    train_df = feature_synthesis(train_df)

    print(train_df.head(10))

    test_df = pd.read_csv("./data/test.csv")
    test_df = preprocess_pipeline(test_df)
    test_df, _ = age_imputation(test_df, True)
    test_df = feature_synthesis(test_df)

    print(test_df.head(10))

    input_features = list(train_df.columns)
    input_features.remove("Survived")
    input_features.remove("Ticket")
    input_features.remove("PassengerId")

    print(f"Input features: {input_features}")

    learner = ydf.GradientBoostedTreesLearner(
        label="Survived",
        features=input_features,
        num_trees=10000,
        split_axis="SPARSE_OBLIQUE",
        sparse_oblique_normalization="MIN_MAX",
        sparse_oblique_num_projections_exponent=1.0,
        shrinkage=0.1,
        growing_strategy="BEST_FIRST_GLOBAL",
        categorical_algorithm="RANDOM",
    )

    model = learner.train(train_df)

    evaluation = learner.cross_validation(train_df, folds=10)
    print(evaluation)

    kaggle_predictions = prediction_to_kaggle_format(model, test_df)
    make_submission(kaggle_predictions)

    # evaluation = model.evaluate(test_df)
    # # Query individual evaluation metrics
    # print(f"test accuracy: {evaluation.accuracy}")
    # # Show the full evaluation report
    # print("Full evaluation report:")
    # print(evaluation)
