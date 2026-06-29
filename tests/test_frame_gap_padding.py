from __future__ import annotations

import numpy as np
from pathlib import Path

from habituation_analysis.data import SessionBundle, SessionSummary, _expand_frame_gaps
from habituation_analysis.stats import (
    STATE_LABELS,
    build_extra_large_mask,
    classify_zscores,
    compute_statistics,
    load_cached_statistics_outputs,
    save_statistics_outputs,
    save_statistics_summary_figure,
    statistics_panel_paths,
    statistics_summary_panel_specs,
    _build_summary_figure,
    _state_labels_for_combined_plot,
)


def _summary(
    *,
    animal_id: str = "M1",
    exp_id: str = "2026-01-01_01_M1",
    date: str = "2026-01-01",
    session_number: int = 1,
    exp_dir: str = "/tmp",
    video_frame_count: int = 6,
    video_duration_sec: float = 0.2,
) -> SessionSummary:
    return SessionSummary(
        animal_id=animal_id,
        exp_id=exp_id,
        date=date,
        session_number=session_number,
        exp_dir=exp_dir,
        right_video=None,
        right_pickle=None,
        locomotion_csv=None,
        has_right_video=True,
        has_right_pickle=True,
        has_locomotion_csv=False,
        video_fps=30.0,
        video_frame_count=video_frame_count,
        video_duration_sec=video_duration_sec,
        source_signature={},
    )


class FakeStore:
    def __init__(self, bundle: SessionBundle):
        self._bundle = bundle
        self.source_root = Path("/tmp")
        self._settings: dict[str, dict] = {}

    def get_animal_settings(self, animal_id: str):
        return self._settings.setdefault(animal_id, {})

    def set_animal_settings(self, animal_id: str, settings: dict):
        self._settings[animal_id] = dict(settings)

    def statistics_cache_signature(self, scope: str, animal_id: str, thresholds: dict):
        return f"{scope}:{animal_id}:{sorted(thresholds.items())}"

    def dataset_sessions(self):
        return [self._bundle.summary]

    def sessions_for_animal(self, animal_id: str):
        return [self._bundle.summary] if animal_id == self._bundle.summary.animal_id else []

    def is_deeplabcut_reference_session(self, exp_id: str) -> bool:
        return False

    def is_session_do_not_use(self, exp_id: str) -> bool:
        return False

    def effective_session_duration_sec(self, exp_id: str) -> float:
        return 1800.0

    def load_session_bundle(self, exp_id: str) -> SessionBundle:
        assert exp_id == self._bundle.summary.exp_id
        return self._bundle

    def load_manual_masks(self, exp_id: str):
        return []

    def session_analysis_cutoff_sec(self, exp_id: str):
        return None

    def session_extra_large_calibration(self, exp_id: str):
        return None

    def load_face_motion(self, exp_id: str):
        return np.array([], dtype=float), np.array([], dtype=float)


class MultiFakeStore:
    def __init__(self, bundles: list[SessionBundle]):
        self._bundles = {bundle.summary.exp_id: bundle for bundle in bundles}
        self.source_root = Path("/tmp")
        self._settings: dict[str, dict] = {}

    def get_animal_settings(self, animal_id: str):
        return self._settings.setdefault(animal_id, {})

    def set_animal_settings(self, animal_id: str, settings: dict):
        self._settings[animal_id] = dict(settings)

    def statistics_cache_signature(self, scope: str, animal_id: str, thresholds: dict):
        return f"{scope}:{animal_id}:{sorted(thresholds.items())}"

    def dataset_sessions(self):
        return [bundle.summary for bundle in self._bundles.values()]

    def sessions_for_animal(self, animal_id: str):
        return [bundle.summary for bundle in self._bundles.values() if bundle.summary.animal_id == animal_id]

    def is_deeplabcut_reference_session(self, exp_id: str) -> bool:
        return False

    def is_session_do_not_use(self, exp_id: str) -> bool:
        return False

    def effective_session_duration_sec(self, exp_id: str) -> float:
        return float(self._bundles[exp_id].summary.video_duration_sec)

    def load_session_bundle(self, exp_id: str) -> SessionBundle:
        return self._bundles[exp_id]

    def load_manual_masks(self, exp_id: str):
        return []

    def session_analysis_cutoff_sec(self, exp_id: str):
        return None

    def session_extra_large_calibration(self, exp_id: str):
        return None

    def load_face_motion(self, exp_id: str):
        return np.array([], dtype=float), np.array([], dtype=float)


def _make_bundle(
    times: np.ndarray,
    radius: np.ndarray,
    frame_observed: np.ndarray | None = None,
    *,
    summary: SessionSummary | None = None,
) -> SessionBundle:
    times = np.asarray(times, dtype=float).reshape(-1)
    radius = np.asarray(radius, dtype=float).reshape(-1)
    n = int(times.size)
    if frame_observed is None:
        frame_observed = np.ones(n, dtype=bool)
    frame_observed = np.asarray(frame_observed, dtype=bool).reshape(-1)
    summary = summary or _summary(video_frame_count=n, video_duration_sec=float(times[-1] + 1.0) if n else 0.0)
    return SessionBundle(
        summary=summary,
        t=times,
        frame_observed=frame_observed,
        radius=radius,
        x=np.linspace(0.0, 1.0, n, dtype=float),
        y=np.linspace(1.0, 2.0, n, dtype=float),
        velocity=np.linspace(2.0, 3.0, n, dtype=float),
        qc=np.zeros(n, dtype=float),
        in_eye=np.ones((n, 1), dtype=bool),
        brake_raw=np.zeros(n, dtype=float),
        brake_t=times.copy(),
        wheel_pos=np.linspace(0.0, 10.0, n, dtype=float),
        locomotion=np.linspace(0.0, 0.4, n, dtype=float),
        locomotion_t=times.copy(),
        eye_lid_x=np.zeros((n, 2), dtype=float),
        eye_lid_y=np.zeros((n, 2), dtype=float),
        eyeX=np.zeros((n, 2), dtype=float),
        eyeY=np.zeros((n, 2), dtype=float),
        pupilX=np.zeros((n, 2), dtype=float),
        pupilY=np.zeros((n, 2), dtype=float),
        source_signature={},
    )


def test_expand_frame_gaps_inserts_missing_frame_and_marks_it_unobserved():
    summary = _summary()
    raw_t = np.array([0.0, 1.0 / 30.0, 2.0 / 30.0, 4.0 / 30.0, 5.0 / 30.0], dtype=float)
    radius = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)
    x = radius + 1.0
    y = radius + 2.0
    velocity = radius + 3.0
    qc = np.zeros(raw_t.size, dtype=float)
    in_eye = np.ones((raw_t.size, 1), dtype=bool)

    t, arrays, observed = _expand_frame_gaps(summary, raw_t, radius, x, y, velocity, qc, in_eye)

    assert t.size == 6
    np.testing.assert_allclose(t, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float) / 30.0)
    assert observed.tolist() == [True, True, True, False, True, True]
    np.testing.assert_allclose(arrays[0][[0, 1, 2, 4, 5]], radius)
    assert np.isnan(arrays[0][3])
    assert not bool(arrays[5][3, 0])


def test_gap_rows_are_ignored_by_classification_and_statistics():
    raw_t = np.array([0.0, 1.0 / 30.0, 2.0 / 30.0, 4.0 / 30.0, 5.0 / 30.0], dtype=float)
    radius = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)
    dense_bundle = _make_bundle(raw_t, radius)

    t, arrays, observed = _expand_frame_gaps(
        dense_bundle.summary,
        raw_t,
        radius,
        dense_bundle.x,
        dense_bundle.y,
        dense_bundle.velocity,
        dense_bundle.qc,
        dense_bundle.in_eye,
        dense_bundle.brake_raw,
        dense_bundle.eye_lid_x,
        dense_bundle.eye_lid_y,
        dense_bundle.eyeX,
        dense_bundle.eyeY,
        dense_bundle.pupilX,
        dense_bundle.pupilY,
    )
    gap_bundle = SessionBundle(
        summary=dense_bundle.summary,
        t=t,
        frame_observed=observed,
        radius=arrays[0],
        x=arrays[1],
        y=arrays[2],
        velocity=arrays[3],
        qc=arrays[4],
        in_eye=arrays[5],
        brake_raw=dense_bundle.brake_raw,
        brake_t=dense_bundle.brake_t,
        wheel_pos=dense_bundle.wheel_pos,
        locomotion=dense_bundle.locomotion,
        locomotion_t=dense_bundle.locomotion_t,
        eye_lid_x=arrays[6],
        eye_lid_y=arrays[7],
        eyeX=arrays[8],
        eyeY=arrays[9],
        pupilX=arrays[10],
        pupilY=arrays[11],
        source_signature={},
    )

    visible = gap_bundle.visible_mask_base
    extra_large_mask = build_extra_large_mask(FakeStore(gap_bundle), gap_bundle, None)
    z = (gap_bundle.radius.astype(float) - np.nanmean(dense_bundle.radius)) / np.nanstd(dense_bundle.radius)
    z[~np.isfinite(z)] = np.nan
    state = classify_zscores(z, [-0.5, 0.5, 1.0], visible, extra_large_mask, None)
    assert state.tolist()[3] == -1
    assert extra_large_mask.tolist() == [False, False, False, False, False, False]

    dense_result = compute_statistics(
        FakeStore(dense_bundle),
        scope="All",
        animal_id="M1",
        percentiles=[25.0, 50.0, 75.0],
        threshold_values=[-0.5, 0.5, 1.0],
        locomotion_threshold=0.1,
    )
    gap_result = compute_statistics(
        FakeStore(gap_bundle),
        scope="All",
        animal_id="M1",
        percentiles=[25.0, 50.0, 75.0],
        threshold_values=[-0.5, 0.5, 1.0],
        locomotion_threshold=0.1,
    )

    np.testing.assert_allclose(gap_result.state_probability, dense_result.state_probability, equal_nan=True)
    np.testing.assert_allclose(gap_result.state_probability_std, dense_result.state_probability_std, equal_nan=True)
    np.testing.assert_allclose(
        np.asarray(gap_result.locomotion_pct_by_session_values["1"], dtype=float),
        np.asarray(dense_result.locomotion_pct_by_session_values["1"], dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray([gap_result.pupil_pct_by_session["1"][label] for label in STATE_LABELS], dtype=float),
        np.asarray([dense_result.pupil_pct_by_session["1"][label] for label in STATE_LABELS], dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray([gap_result.lag_by_session["1"][label] for label in STATE_LABELS], dtype=float),
        np.asarray([dense_result.lag_by_session["1"][label] for label in STATE_LABELS], dtype=float),
        equal_nan=True,
    )
    assert all(label in gap_result.pupil_zscore_mean_by_state_values for label in STATE_LABELS)
    assert any(gap_result.pupil_zscore_mean_by_state_values[label] for label in STATE_LABELS)
    np.testing.assert_allclose(
        np.asarray([gap_result.pupil_pct_by_session_values["1"][label] for label in STATE_LABELS], dtype=float),
        np.asarray([dense_result.pupil_pct_by_session_values["1"][label] for label in STATE_LABELS], dtype=float),
        equal_nan=True,
    )


def test_statistics_outputs_generate_boxplot_panels(tmp_path):
    raw_t = np.array([0.0, 1.0 / 30.0, 2.0 / 30.0, 4.0 / 30.0, 5.0 / 30.0], dtype=float)
    radius = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)
    bundle = _make_bundle(raw_t, radius)
    store = FakeStore(bundle)
    store.source_root = tmp_path

    result = compute_statistics(
        store,
        scope="All",
        animal_id="M1",
        percentiles=[25.0, 50.0, 75.0],
        threshold_values=[-0.5, 0.5, 1.0],
        locomotion_threshold=0.1,
    )

    result_path, svg_path, png_path = save_statistics_outputs(store, result)
    assert result_path.exists()
    assert svg_path.exists()
    assert png_path.exists()
    for _, panel_path in statistics_panel_paths(result_path.parent):
        assert panel_path.exists()



def test_statistics_cache_rejects_incomplete_panel_exports(tmp_path):
    raw_t = np.array([0.0, 1.0 / 30.0, 2.0 / 30.0, 4.0 / 30.0, 5.0 / 30.0], dtype=float)
    radius = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)
    bundle = _make_bundle(raw_t, radius)
    store = FakeStore(bundle)
    store.source_root = tmp_path

    result = compute_statistics(
        store,
        scope="All",
        animal_id="M1",
        percentiles=[25.0, 50.0, 75.0],
        threshold_values=[-0.5, 0.5, 1.0],
        locomotion_threshold=0.1,
    )

    result_path, svg_path, png_path = save_statistics_outputs(store, result)
    output_dir = result_path.parent
    missing_panel = statistics_panel_paths(output_dir)[-1][1]
    assert missing_panel.exists()
    missing_panel.unlink()

    cached = load_cached_statistics_outputs(
        store,
        scope="All",
        animal_id="M1",
        thresholds=result.thresholds,
    )
    assert cached is None


def test_statistics_outputs_expand_panels_and_window_plots(tmp_path):
    short_t = np.arange(0.0, 35.0 * 60.0, 60.0, dtype=float)
    long_t = np.arange(0.0, 65.0 * 60.0, 60.0, dtype=float)
    short_radius = np.full(short_t.shape, 10.0, dtype=float)
    long_radius = np.full(long_t.shape, 10.0, dtype=float)

    short_summary = SessionSummary(
        animal_id="M1",
        exp_id="2026-01-01_01_M1",
        date="2026-01-01",
        session_number=1,
        exp_dir="/tmp/short",
        right_video=None,
        right_pickle=None,
        locomotion_csv=None,
        has_right_video=True,
        has_right_pickle=True,
        has_locomotion_csv=False,
        video_fps=30.0,
        video_frame_count=int(short_t.size),
        video_duration_sec=float(short_t[-1] + 60.0),
        source_signature={},
    )
    long_summary = SessionSummary(
        animal_id="M1",
        exp_id="2026-01-02_02_M1",
        date="2026-01-02",
        session_number=2,
        exp_dir="/tmp/long",
        right_video=None,
        right_pickle=None,
        locomotion_csv=None,
        has_right_video=True,
        has_right_pickle=True,
        has_locomotion_csv=False,
        video_fps=30.0,
        video_frame_count=int(long_t.size),
        video_duration_sec=float(long_t[-1] + 60.0),
        source_signature={},
    )

    short_bundle = _make_bundle(short_t, short_radius, summary=short_summary)
    long_bundle = _make_bundle(long_t, long_radius, summary=long_summary)

    store = MultiFakeStore([short_bundle, long_bundle])
    store.source_root = tmp_path

    result = compute_statistics(
        store,
        scope="M1",
        animal_id="M1",
        percentiles=[25.0, 50.0, 75.0],
        threshold_values=[-0.5, 0.5, 1.0],
        locomotion_threshold=0.1,
    )

    assert _state_labels_for_combined_plot() == ["small", "medium", "large", "extra_large"]
    assert "not_visible" not in _state_labels_for_combined_plot()
    assert "window_01" in result.progress_series
    assert "window_02" in result.progress_series
    assert int(result.progress_series["window_02"]["sample_size"]) == 1

    long_only_store = MultiFakeStore([long_bundle])
    long_only_store.source_root = tmp_path
    long_only_result = compute_statistics(
        long_only_store,
        scope="M1",
        animal_id="M1",
        percentiles=[25.0, 50.0, 75.0],
        threshold_values=[-0.5, 0.5, 1.0],
        locomotion_threshold=0.1,
    )

    np.testing.assert_allclose(
        np.asarray(result.progress_series["window_02"]["state_probability"], dtype=float),
        np.asarray(long_only_result.progress_series["window_02"]["state_probability"], dtype=float),
        equal_nan=True,
    )

    panel_specs = statistics_summary_panel_specs(result)
    assert [slug for slug, _, _ in panel_specs[:3]] == ["locomotion_boxplot", "face_motion_boxplot", "pupil_zscore_by_state"]
    assert "with_not_visible" in result.pupil_state_fraction_overall
    assert "without_not_visible" in result.pupil_state_fraction_overall

    selected_fig = _build_summary_figure(
        result,
        panel_keys=["locomotion_progress", "pupil_size_progress", "pupil_state_fraction_pie", "state_fraction_stacked"],
    )
    selected_titles = [ax.get_title() for ax in selected_fig.axes if ax.get_title()]
    assert any(title.startswith("Overall pupil state percentages") for title in selected_titles)
    assert any(title == "Locomotion progression across sessions" for title in selected_titles)
    assert any(title == "Mean z-scored pupil size progression across sessions" for title in selected_titles)
    assert any(title == "Pupil state fraction vs experiment length (stacked area)" for title in selected_titles)

    selected_svg, selected_png = save_statistics_summary_figure(
        tmp_path / "selected_summary",
        result,
        summary_panel_keys=["locomotion_progress", "pupil_size_progress", "pupil_state_fraction_pie", "state_fraction_stacked"],
    )
    assert selected_svg.exists()
    assert selected_png.exists()

    result_path, svg_path, png_path = save_statistics_outputs(store, result)
    assert result_path.exists()
    assert svg_path.exists()
    assert png_path.exists()

    panel_titles = [title for title, _ in statistics_panel_paths(result_path.parent)]
    assert any(title == "Locomotion by session (<= 7)" for title in panel_titles)
    assert any(title == "Pupil state fraction by session - small" for title in panel_titles)
    assert any(title == "Pupil state fraction by session - not_visible" for title in panel_titles)
    assert any(title == "Overall pupil state percentages" for title in panel_titles)
    assert any("0-30 min" in title for title in panel_titles)
    assert any("30-60 min" in title for title in panel_titles)
    assert any("first 2 sessions" in title for title in panel_titles)
    assert any("last 2 sessions" in title for title in panel_titles)
    for _, panel_path in statistics_panel_paths(result_path.parent):
        assert panel_path.exists()


def test_statistics_all_panel_uses_animal_weighting(tmp_path):
    short_t = np.arange(0.0, 35.0 * 60.0, 60.0, dtype=float)
    long_t = np.arange(0.0, 65.0 * 60.0, 60.0, dtype=float)
    radius = np.full(short_t.shape, 10.0, dtype=float)
    long_radius = np.full(long_t.shape, 10.0, dtype=float)

    bundles = []
    for animal_id in ("M1", "M2"):
        for session_number, (times, values) in enumerate(((short_t, radius), (long_t, long_radius)), start=1):
            summary = _summary(
                animal_id=animal_id,
                exp_id=f"2026-01-0{session_number}_{session_number:02d}_{animal_id}",
                date=f"2026-01-0{session_number}",
                session_number=session_number,
                exp_dir=f"/tmp/{animal_id}_{session_number}",
                video_frame_count=int(times.size),
                video_duration_sec=float(times[-1] + 60.0),
            )
            bundles.append(_make_bundle(times, values, summary=summary))

    store = MultiFakeStore(bundles)
    store.source_root = tmp_path

    result = compute_statistics(
        store,
        scope="All",
        animal_id="All",
        percentiles=[25.0, 50.0, 75.0],
        threshold_values=[-0.5, 0.5, 1.0],
        locomotion_threshold=0.1,
    )

    assert int(result.progress_series["overall"]["sample_size"]) == 2
    assert int(result.progress_series["first_2"]["sample_size"]) == 2
    assert int(result.progress_series["last_2"]["sample_size"]) == 2
    assert set(result.locomotion_progress_by_animal_values) == {"M1", "M2"}
    assert set(result.pupil_zscore_progress_by_animal_values) == {"M1", "M2"}

    panel_specs = statistics_summary_panel_specs(result)
    assert [slug for slug, _, _ in panel_specs[:3]] == ["locomotion_boxplot", "face_motion_boxplot", "pupil_zscore_by_state"]
    assert "with_not_visible" in result.pupil_state_fraction_overall
    assert "without_not_visible" in result.pupil_state_fraction_overall

    selected_fig = _build_summary_figure(
        result,
        panel_keys=["locomotion_progress", "pupil_size_progress", "pupil_state_fraction_pie", "state_fraction_stacked"],
    )
    selected_titles = [ax.get_title() for ax in selected_fig.axes if ax.get_title()]
    assert any(title.startswith("Overall pupil state percentages") for title in selected_titles)
    assert any(title == "Locomotion progression across sessions" for title in selected_titles)
    assert any(title == "Mean z-scored pupil size progression across sessions" for title in selected_titles)
    assert any(title == "Pupil state fraction vs experiment length (stacked area)" for title in selected_titles)

    selected_svg, selected_png = save_statistics_summary_figure(
        tmp_path / "selected_summary",
        result,
        summary_panel_keys=["locomotion_progress", "pupil_size_progress", "pupil_state_fraction_pie", "state_fraction_stacked"],
    )
    assert selected_svg.exists()
    assert selected_png.exists()

    result_path, svg_path, png_path = save_statistics_outputs(store, result)
    assert result_path.exists()
    assert svg_path.exists()
    assert png_path.exists()

    panel_titles = [title for title, _ in statistics_panel_paths(result_path.parent)]
    assert any(title == "Locomotion progression across sessions" for title in panel_titles)
    assert any(title == "Mean z-scored pupil size progression across sessions" for title in panel_titles)
    assert any(title == "Pupil state fraction by session (stacked area)" for title in panel_titles)
    assert any(title == "Pupil state fraction by session (stacked area, <= 7)" for title in panel_titles)
    assert any(title == "Pupil state fraction vs experiment length (stacked area)" for title in panel_titles)
    assert any(title == "Pupil state fraction by session - small" for title in panel_titles)
    assert any(title == "Pupil state fraction by session - not_visible" for title in panel_titles)
    assert any("first 2 sessions" in title for title in panel_titles)
    assert any("last 2 sessions" in title for title in panel_titles)
    for _, panel_path in statistics_panel_paths(result_path.parent):
        assert panel_path.exists()
