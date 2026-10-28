"""Phase 6 tests: every dashboard page runs without an exception on the real pipeline outputs, and the data
layer reproduces the Phase 5 numbers and labels synthetic data."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from solvia.config import get_path
from solvia.dashboard import data as D

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["Home.py", *sorted(p.relative_to(APP).as_posix() for p in (APP / "pages").glob("*.py"))]
needs_outputs = pytest.mark.skipif(not (get_path("metrics") / "test_run_log.json").exists(),
                                   reason="pipeline outputs not built (run the pipeline first)")


@needs_outputs
@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_exception(page, monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    at = AppTest.from_file(str(APP / page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


@needs_outputs
def test_dashboard_replay_matches_phase5_test_metrics():
    r = D.consumption_replay("india")
    ev = pd.read_csv(get_path("metrics") / "evaluation_consumption_india.csv")
    phase5 = ev[ev["selected"] & (ev["subset"] == "test")]["mae"].iloc[0]
    assert D.error_summary(r[r["split"] == "test"])["mae_model"] == pytest.approx(phase5)
    assert set(r["split"]) == {"valid", "test"}                 # never shows in-sample training days


@needs_outputs
def test_inverter_labels_unique_and_synthetic_labelled():
    labels = D.inverter_labels()
    assert len(labels) == 44 and len(set(labels.values())) == 44
    assert D.pv360_daily().attrs["kind"] == "SYNTHETIC"
