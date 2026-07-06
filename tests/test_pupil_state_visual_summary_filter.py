from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from habituation_analysis.data import SessionBundle, SessionSummary
from habituation_analysis import stats as statsmod


def _summary(*, animal_id: str = 'M1', exp_id: str = '2026-01-01_01_M1', date: str = '2026-01-01') -> SessionSummary:
    return SessionSummary(
        animal_id=animal_id,
        exp_id=exp_id,
        date=date,
        session_number=1,
        exp_dir='/tmp',
        right_video=None,
        right_pickle=None,
        locomotion_csv=None,
        has_right_video=True,
        has_right_pickle=True,
        has_locomotion_csv=False,
        video_fps=30.0,
        video_frame_count=6,
        video_duration_sec=0.2,
        source_signature={},
    )


def _bundle() -> SessionBundle:
    summary = _summary()
    n = 6
    t = np.arange(n, dtype=float) / 30.0
    radius = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=float)
    zeros = np.zeros(n, dtype=float)
    ones = np.ones((n, 1), dtype=bool)
    return SessionBundle(
        summary=summary,
        t=t,
        frame_observed=np.ones(n, dtype=bool),
        radius=radius,
        x=radius + 1.0,
        y=radius + 2.0,
        velocity=radius + 3.0,
        qc=zeros,
        in_eye=ones,
        brake_raw=zeros,
        brake_t=t.copy(),
        wheel_pos=np.linspace(0.0, 10.0, n, dtype=float),
        locomotion=np.linspace(0.0, 0.4, n, dtype=float),
        locomotion_t=t.copy(),
        eye_lid_x=np.zeros((n, 2), dtype=float),
        eye_lid_y=np.zeros((n, 2), dtype=float),
        eyeX=np.zeros((n, 2), dtype=float),
        eyeY=np.zeros((n, 2), dtype=float),
        pupilX=np.zeros((n, 2), dtype=float),
        pupilY=np.zeros((n, 2), dtype=float),
        source_signature={},
    )


class _Store:
    def __init__(self, bundle: SessionBundle):
        self._bundle = bundle
        self.source_root = Path('/tmp')
        self._settings: dict[str, dict] = {}

    def get_animal_settings(self, animal_id: str):
        return self._settings.setdefault(animal_id, {})

    def set_animal_settings(self, animal_id: str, settings: dict):
        self._settings[animal_id] = dict(settings)

    def statistics_cache_signature(self, scope: str, animal_id: str, thresholds: dict):
        return f'{scope}:{animal_id}:{sorted(thresholds.items())}'

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


def test_statistics_summary_panels_and_pie_exclude_no_pupil_state() -> None:
    store = _Store(_bundle())
    result = statsmod.compute_statistics(
        store,
        scope='All',
        animal_id='M1',
        percentiles=[25.0, 50.0, 75.0],
        threshold_values=[-0.5, 0.5, 1.0],
        locomotion_threshold=0.1,
    )

    assert statsmod._state_labels_for_combined_plot() == ['extra_large', 'large', 'medium', 'small']

    panel_specs = statsmod.statistics_summary_panel_specs(result)
    for _, title, config in panel_specs:
        title_lower = title.lower()
        assert 'not visible' not in title_lower
        assert 'not_visible' not in title_lower
        labels = config.get('labels')
        if labels is not None:
            assert all(not statsmod._is_no_pupil_label(label) for label in labels)

    visible_fractions = {label: float(idx + 1) for idx, label in enumerate(statsmod._state_labels_for_combined_plot())}
    fig = plt.figure(figsize=(4.0, 4.0))
    try:
        ax = fig.add_subplot(111)
        statsmod._plot_state_fraction_pie(
            ax,
            title='Overall pupil state percentages',
            values=visible_fractions,
            labels=statsmod._state_labels_for_combined_plot(),
            sample_size=1,
        )
        pie_text = {text.get_text().strip().lower() for text in ax.texts}
        assert 'not visible' not in pie_text
        assert 'not_visible' not in pie_text
        for label in statsmod._state_labels_for_combined_plot():
            assert label in pie_text
    finally:
        plt.close(fig)
