"""Phase 5 tests: the test-set guard refuses changed models; re-running with the same models is allowed."""
import json

import pytest

from solvia.models import evaluation as ev


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    models, metrics = tmp_path / "models", tmp_path / "metrics"
    models.mkdir(), metrics.mkdir()
    for n in ev.MODEL_FILES:
        (models / f"{n}.joblib").write_bytes(n.encode())
    monkeypatch.setattr(ev, "get_path", lambda key: {"models": models, "metrics": metrics}[key])
    return models, metrics


def test_first_run_then_same_models_allowed(dirs):
    _, metrics = dirs
    hashes = ev.check_test_guard()
    ev._write_test_log(hashes)
    assert ev.check_test_guard() == hashes                  # identical files: allowed
    ev._write_test_log(hashes)
    log = json.loads((metrics / ev.LOG_FILE).read_text())
    assert log["runs"] == 2 and log["first_run_utc"] <= log["last_run_utc"]


def test_changed_model_after_test_run_is_refused(dirs):
    models, _ = dirs
    ev._write_test_log(ev.check_test_guard())
    (models / "consumption_india.joblib").write_bytes(b"retrained after seeing the test")
    with pytest.raises(RuntimeError, match="consumption_india"):
        ev.check_test_guard()
