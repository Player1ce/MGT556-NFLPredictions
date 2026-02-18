import itertools
import polars as pl
import numpy as np
from sklearn.linear_model import (LinearRegression)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def adjusted_r2(r2, n, p):
    """
    Compute adjusted R^2
    n = number of samples
    p = number of predictors
    """
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def best_subset_selection(
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

    # Separate X and y
    y = df_pd[target_col]
    X = df_pd.drop(columns=[target_col])

    feature_names = list(X.columns)

    if max_features is None:
        max_features = len(feature_names)

    best_score = -np.inf
    best_subset = None
    best_metrics = None

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

    return {
        "best_features": best_subset,
        "metrics": best_metrics
    }
