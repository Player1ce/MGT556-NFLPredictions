import itertools
import math

import polars as pl
import numpy as np
from sklearn.linear_model import (LinearRegression)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from functools import partial


def adjusted_r2(r2, n, p):
    """
    Compute adjusted R^2
    n = number of samples
    p = number of predictors
    """
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def exhaustive_best_subset_selection(
        df: pl.DataFrame,
        target_col: str = "log1p_future_class_ppr_score",
        max_features: int | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
):
    """
    Perform exhaustive best subset selection using linear regression.
    Returns best feature subset and associated metrics.
    """

    # Convert to pandas for sklearn compatibility
    df_pd = df.to_pandas()
    df_pd.dropna(subset=[target_col], inplace=True)

    terms_to_remove = ["score", 'points']
    score_cols = [col for col in df_pd.columns for term in terms_to_remove if term in col]
    score_cols = list({*score_cols, target_col})

    # Separate X and y
    y = df_pd[target_col]
    X = df_pd.drop(columns=score_cols).select_dtypes(include=['number'])

    feature_names = list(X.columns)

    if max_features is None:
        max_features = len(feature_names)

    best_score = -np.inf
    best_subset = None
    best_metrics = None

    print("total tests:", math.comb(len(feature_names), max_features))

    for k in range(1, max_features + 1):
        for subset in itertools.combinations(feature_names, k):

            X_subset = X[list(subset)]

            X_train, X_test, y_train, y_test = train_test_split(
                X_subset, y,
                test_size=test_size,
                random_state=random_state
            )

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            adj_r2 = adjusted_r2(r2, len(y_test), k)

            if adj_r2 > best_score:
                best_score = adj_r2
                best_subset = subset
                best_metrics = {
                    "r2": r2,
                    "adjusted_r2": adj_r2,
                    "num_features": k
                }
        print("k:", k, "| best_score:", best_score, "| best_subset:", best_subset)

    return {
        "best_features": best_subset,
        "metrics": best_metrics
    }


def forward_stepwise_best_subset(df: pl.DataFrame, target_col: str, max_features: int = 10):
    df_pd = df.to_pandas().dropna(subset=[target_col])

    terms_to_remove = ["score", "points"]
    score_cols = [col for col in df_pd.columns if any(term in col for term in terms_to_remove)]
    score_cols.append(target_col)

    y = df_pd[target_col]
    X = df_pd.drop(columns=score_cols).select_dtypes(include=['number'])
    features = list(X.columns)

    current_features = []
    best_score = -np.inf

    for k in range(max_features):
        best_new_feature = None
        best_score_new = -np.inf
        print("k:", k)

        for feature in features:
            if feature not in current_features:
                test_features = current_features + [feature]
                X_test = X[test_features]

                X_train, X_test, y_train, y_test = train_test_split(X_test, y, random_state=42)
                model = LinearRegression().fit(X_train, y_train)
                score = adjusted_r2(r2_score(y_test, model.predict(X_test)), len(y_test), len(test_features))

                if score > best_score_new:
                    best_score_new = score
                    best_new_feature = feature

        if best_new_feature:
            print("new feature:", best_new_feature)
            current_features.append(best_new_feature)
            best_score = best_score_new
        else:
            print("no new feature, exiting early at k:", k)
            break

    return current_features, best_score


def beam_search_best_subset(df, target_col, max_features=10, beam_width=50):
    """Keep top-N subsets at each step, explore their neighbors"""
    df_pd = df.to_pandas().dropna(subset=[target_col])

    terms_to_remove = ["score", "points"]
    score_cols = [col for col in df_pd.columns if any(term in col for term in terms_to_remove)]
    score_cols.append(target_col)

    # Pre-split ONCE
    y = df_pd[target_col]
    X = df_pd.drop(columns=score_cols).select_dtypes(include=['number'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    features = list(X.columns)

    # Initialize with single best features
    beams = []
    for feature in features:
        score = adjusted_r2(
            r2_score(y_test[[feature]], LinearRegression().fit(X_train[[feature]], y_train)
                     .predict(X_test[[feature]])),
            len(y_test), 1
        )
        beams.append((frozenset([feature]), score))

    # Sort by score (descending order)
    beams = sorted(beams, key=lambda x: x[1], reverse=True)[:beam_width]

    for k in range(2, max_features + 1):
        new_beams = []
        for feature_set, score in beams:
            # Add one new feature at a time (stepwise-like)
            for new_feature in features:
                if new_feature not in feature_set:
                    test_set = feature_set | {new_feature}
                    X_sub = X_train[list(test_set)]
                    score_new = adjusted_r2(
                        r2_score(y_test[list(test_set)], LinearRegression().fit(X_sub, y_train).predict(X_sub)),
                        len(y_test), k
                    )
                    new_beams.append((test_set, score_new))

        # Keep top beam_width (descending order)
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        print(f"k={k}, best_score={beams[0][1]:.4f}")

    return max(beams, key=lambda x: x[1])
