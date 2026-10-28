"""Model pipelines. Scalers and encoders live inside the Pipeline, so they are fitted on training rows only."""
from __future__ import annotations

from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def make_pipeline(model: RegressorMixin, features: list[str], linear: bool,
                  categorical: list[str] | None = None) -> Pipeline:
    """Linear models: one-hot encode categorical columns and standardise the rest.
    Tree models are scale-invariant: numeric columns pass through and categorical (text) columns are
    ordinal-encoded inside the pipeline (unseen categories -> -1), so encoders are fitted on training rows only."""
    categorical = [c for c in (categorical or []) if c in features]
    numeric = [c for c in features if c not in categorical]
    if not linear:
        if not categorical:
            return Pipeline([("model", model)])
        pre = ColumnTransformer([
            ("num", "passthrough", numeric),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
        ])
        return Pipeline([("prep", pre), ("model", model)])
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline([("prep", pre), ("model", model)])
