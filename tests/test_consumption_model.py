"""Phase 4A tests: relative-target wrapper, categorical encoding inside pipelines, clipping and intervals."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from solvia.features.preprocess import make_pipeline
from solvia.models.consumption import ConsumptionModel, RelativeTarget, conformal_adjustment


def test_relative_target_learns_ratio_and_scales_back():
    X = pd.DataFrame({"roll_mean_30": [1.0, 2.0, 4.0, 8.0], "x": [1.0, 1.0, 1.0, 1.0]})
    y = 1.5 * X["roll_mean_30"]                                     # every home uses 1.5 x its own 30-day mean
    m = RelativeTarget(LinearRegression()).fit(X, y)
    assert m.predict(pd.DataFrame({"roll_mean_30": [10.0], "x": [1.0]}))[0] == pytest.approx(15.0)


def test_relative_target_floor_avoids_division_by_zero():
    X = pd.DataFrame({"roll_mean_30": [0.0, 1.0], "x": [0.0, 1.0]})
    m = RelativeTarget(LinearRegression(), floor=0.1).fit(X, [0.1, 1.0])
    assert np.isfinite(m.predict(X)).all()


def test_tree_pipeline_encodes_categories_and_handles_unseen():
    X = pd.DataFrame({"a": np.arange(40, dtype=float), "household_type": ["Basic", "With AC"] * 20})
    y = X["a"] + np.where(X["household_type"] == "With AC", 10, 0)
    pipe = make_pipeline(HistGradientBoostingRegressor(max_iter=50, random_state=42), ["a", "household_type"],
                         linear=False, categorical=["household_type"]).fit(X, y)
    new = pd.DataFrame({"a": [5.0], "household_type": ["Never seen"]})
    assert np.isfinite(pipe.predict(new)).all()                     # unseen category -> -1, no crash


def test_consumption_model_clips_and_orders_interval():
    class Const:
        def __init__(self, v): self.v = v
        def predict(self, X): return np.full(len(X), self.v)
    X = pd.DataFrame({"f": [1.0, 2.0]})
    m = ConsumptionModel("t", "x", ["f"], Const(-1.0), p10=Const(5.0), p90=Const(3.0))
    assert (m.predict(X) == 0).all()
    lo, hi = m.predict_interval(X)
    assert (lo <= hi).all() and lo[0] == 3.0 and hi[0] == 5.0


def test_conformal_adjustment_reaches_target_coverage_and_widens_interval():
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, 1000)
    lo, hi = np.full(1000, -0.5), np.full(1000, 0.5)                   # far too narrow: ~38 % coverage
    q = conformal_adjustment(y, lo, hi, 0.8)
    assert q > 0 and ((y >= lo - q) & (y <= hi + q)).mean() >= 0.8
    assert conformal_adjustment(y, np.full(1000, -5.0), np.full(1000, 5.0), 0.8) < 0   # too wide -> narrowed


def test_calibrated_interval_uses_adjustment():
    class Const:
        def __init__(self, v): self.v = v
        def predict(self, X): return np.full(len(X), self.v)
    X = pd.DataFrame({"f": [1.0]})
    m = ConsumptionModel("t", "x", ["f"], Const(4.0), p10=Const(3.0), p90=Const(5.0), interval_adjust_kwh=1.0)
    assert m.predict_interval(X)[0][0] == 2.0 and m.predict_interval(X)[1][0] == 6.0
    assert m.predict_interval(X, calibrated=False)[0][0] == 3.0
