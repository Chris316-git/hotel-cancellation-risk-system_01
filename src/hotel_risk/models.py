"""Model pipelines: a linear baseline, LightGBM (the main model) and scikit-learn gradient boosting."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def make_baseline() -> Pipeline:
    """Standardised numerics + one-hot categoricals -> logistic regression."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, C=1.0))])


def _ordinal_pre() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1, min_frequency=50), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])


def make_gbm(random_state: int = 0) -> Pipeline:
    """Main model: LightGBM on ordinal-encoded categoricals."""
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                         random_state=random_state, verbose=-1)
    return Pipeline([("pre", _ordinal_pre()), ("clf", clf)])


def make_hgb(random_state: int = 0) -> Pipeline:
    """scikit-learn HistGradientBoosting, kept for the model comparison (no extra dependency)."""
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1, min_frequency=50), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    n_cat = len(CATEGORICAL)
    clf = HistGradientBoostingClassifier(
        categorical_features=list(range(n_cat)), learning_rate=0.08, max_iter=400,
        max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, random_state=random_state)
    return Pipeline([("pre", pre), ("clf", clf)])
