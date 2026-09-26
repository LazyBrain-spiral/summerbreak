"""End-to-end run on the synth3d flight with the oracle depth predictor.

Needs COLMAP (CLI or pycolmap) and the rendered flight:
    python tools/synth3d.py --out test_data/synth3d_pass
    python -m pytest tests/v3 -m slow
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "test_data" / "synth3d_pass"


def _have_colmap():
    if shutil.which("colmap"):
        return True
    try:
        import pycolmap  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not (SYNTH / "gt.json").exists(), reason="render test_data/synth3d_pass first")
@pytest.mark.skipif(not _have_colmap(), reason="COLMAP not available")
def test_synth_oracle_end_to_end(tmp_path):
    from eval.synth_eval import evaluate
    from src.pipeline import main

    run_root = tmp_path / "runs"
    code = main(["--config", str(ROOT / "configs" / "synth_oracle.yaml"),
                 "--video", str(SYNTH / "flight.mp4"), "--telemetry", str(SYNTH / "flight.srt"),
                 "--gt", str(SYNTH), "--run-id", "e2e", "--skip-env-check",
                 "--set", f"run_root={run_root}"])
    run = run_root / "e2e"
    assert code == 0, (run / "report.json").read_text()
    res = evaluate(str(run), str(SYNTH))
    assert res["camera_position"]["rmse_m"] < 3.0
    assert res["dsm"]["median_abs_err_m"] < 0.6
    b = res["buildings"]
    assert b["matched"] >= 3
    assert b["median_abs_ridge_err_m"] < 1.0
