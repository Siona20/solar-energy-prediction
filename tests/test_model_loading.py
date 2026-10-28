"""Phase 7 tests: the saved models load with the installed libraries, carry complete metadata and give
sane predictions (night = 0, never negative, ordered intervals) on rows they were not trained on."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from solvia.config import get_path
from solvia.models import consumption as cons
from solvia.models.evaluation import MODEL_FILES

needs_models = pytest.mark.skipif(not all((get_path("models") / f"{n}.joblib").exists() for n in MODEL_FILES),
                                  reason="models not trained (run the pipeline first)")
REQUIRED = {"name", "saved_utc", "python", "scikit_learn", "objective", "model", "features", "target", "hyperparameters",
            "selection", "validation_metrics", "test_metrics"}


@needs_models
@pytest.mark.parametrize("name", MODEL_FILES)
def test_metadata_complete_and_versions_match(name):
    meta = json.loads((get_path("models") / f"{name}.json").read_text(encoding="utf-8"))
    assert REQUIRED <= set(meta), REQUIRED - set(meta)
    assert meta["scikit_learn"] == sklearn.__version__       # pickles are only safe with the same version
    assert isinstance(meta["test_metrics"], dict)             # filled by the Phase 5 run, not the placeholder text


@needs_models
def test_solar_model_zero_at_night_and_never_negative():
    model = joblib.load(get_path("models") / "solar_expected_output.joblib")
    f = pd.read_parquet(get_path("processed") / "solar_features_15min.parquet")
    f = f[f["split"] == "test"]
    pred = model.predict_kw(f)
    assert len(pred) == len(f) and np.isfinite(pred).all() and (pred >= 0).all()
    assert (pred[(f["irradiation_kw_m2"] <= 0).to_numpy()] == 0).all()
    assert pred[(f["irradiation_kw_m2"] > 0.5).to_numpy()].min() > 0


@needs_models
@pytest.mark.parametrize("track", ["india", "france"])
def test_consumption_model_predicts_with_ordered_interval(track):
    cm = joblib.load(get_path("models") / f"consumption_{track}.joblib")
    df = cons.load_track(cons.TRACKS[track])
    t = df[df["split"] == "test"].head(200)
    p = cm.predict(t)
    lo, hi = cm.predict_interval(t)
    assert np.isfinite(p).all() and (p >= 0).all()
    assert (lo <= hi).all() and (lo >= 0).all()
    assert cm.features == list(cons.TRACKS[track].features)  # model and feature code agree
