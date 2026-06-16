from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from habituation_analysis.data import SessionSummary
from habituation_analysis.stats import (
    build_extra_large_mask,
    classify_zscores,
    learn_extra_large_calibration,
    learn_extra_large_reference_band,
)


def _summary() -> SessionSummary:
    return SessionSummary(
        animal_id="M1",
        exp_id="2026-01-01_01_M1",
        date="2026-01-01",
        session_number=1,
        exp_dir="/tmp",
        right_video=None,
        right_pickle=None,
        locomotion_csv=None,
        has_right_video=True,
        has_right_pickle=False,
        has_locomotion_csv=False,
        video_fps=30.0,
        video_frame_count=6,
        video_duration_sec=0.2,
        source_signature={},
    )


def test_learned_reference_band_and_cutoff_from_calibration_interval():
    times = np.arange(6, dtype=float)
    pupil_brightness = np.array([20.0, 22.0, 21.0, 80.0, 82.0, 81.0], dtype=float)
    similarity = np.array([0.15, 0.20, 0.18, 0.92, 0.91, 0.93], dtype=float)

    reference = learn_extra_large_reference_band(times, pupil_brightness, (0.0, 3.0))
    assert reference is not None
    assert reference["reference_band"][0] <= reference["reference_band"][1]

    calibration = learn_extra_large_calibration(times, pupil_brightness, similarity, (0.0, 3.0))
    assert calibration is not None
    assert calibration["reference_band"] == reference["reference_band"]
    assert 0.0 <= calibration["similarity_cutoff"] <= 1.0


def test_build_extra_large_mask_only_marks_missing_pupil_frames():
    bundle = SimpleNamespace(
        summary=_summary(),
        t=np.arange(6, dtype=float),
        radius=np.array([5.0, np.nan, np.nan, 5.0, np.nan, 5.0], dtype=float),
    )

    calibration = {
        "interval": (0.0, 3.0),
        "reference_band": (20.0, 25.0),
        "similarity_cutoff": 0.6,
        "confirmed": True,
    }

    class FakeStore:
        def load_eye_similarity(self, exp_id: str, band, *, force_recompute: bool = False):
            assert exp_id == bundle.summary.exp_id
            assert tuple(band) == calibration["reference_band"]
            return bundle.t, np.array([0.95, 0.10, 0.80, 0.99, 0.75, 0.50], dtype=float)

    mask = build_extra_large_mask(FakeStore(), bundle, calibration, manual_masks=[(4.0, 5.0)], manual_buffer_sec=0.0)
    assert mask.tolist() == [False, False, True, False, False, False]


def test_manual_not_visible_overrides_extra_large_state():
    z = np.array([0.0, 1.0, 3.0], dtype=float)
    extra_large_mask = np.array([False, True, False], dtype=bool)
    not_visible_mask = np.array([False, True, False], dtype=bool)

    state = classify_zscores(z, [0.5, 1.5, 2.5], None, extra_large_mask, not_visible_mask)
    assert state.tolist() == [0, 4, 3]
