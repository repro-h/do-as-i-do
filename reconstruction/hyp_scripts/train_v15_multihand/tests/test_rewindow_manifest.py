import importlib.util
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "rewindow_manifest.py"
SPEC = importlib.util.spec_from_file_location("rewindow_manifest", MODULE)
rewindow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rewindow)


def test_long_run_anchors_final_window():
    windows = rewindow.window_positions(list(range(100)), 64, 32)
    assert [positions[0] for positions in windows] == [0, 32, 36]


def test_short_runs_are_rejected():
    assert rewindow.window_positions(list(range(63)), 64, 32) == []
