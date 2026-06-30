
from __future__ import annotations

import json
import cv2
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .data import (
    MIN_STATISTICS_SESSIONS_PER_ANIMAL,
    HabituationStore,
    SessionBundle,
    analysis_cutoff_mask,
    apply_time_mask,
)
from .plotting import save_figure, set_poster_style, style_axes


STATE_LABELS = ["small", "medium", "large", "extra_large", "not_visible"]
STATE_COLORS = ["tab:green", "tab:purple", "tab:red", "tab:brown", "tab:gray"]
MIN_EXTRA_LARGE_MISSING_SEC = 1.0
MANUAL_INTERVAL_BUFFER_SEC = 1.0
CALIBRATION_BRIGHTNESS_MARGIN = 0.10
STATISTICS_PLOT_VERSION = 5


@dataclass
class StatisticsResult:
    scope: str
    animal_id: str
    generated_at: str
    session_ids: list[str]
    eligible_session_ids: list[str]
    session_labels: list[str]
    locomotion_pct_by_session: dict[str, float]
    locomotion_pct_by_session_values: dict[str, list[float]]
    face_motion_pct_by_session: dict[str, float]
    face_motion_pct_by_session_values: dict[str, list[float]]
    face_motion_mean_by_state: dict[str, float]
    face_motion_std_by_state: dict[str, float]
    pupil_zscore_mean_by_state_values: dict[str, list[float]]
    pupil_pct_by_session: dict[str, dict[str, float]]
    pupil_pct_by_session_values: dict[str, dict[str, list[float]]]
    pupil_pct_by_session_visible: dict[str, dict[str, float]]
    pupil_pct_by_session_visible_values: dict[str, dict[str, list[float]]]
    lag_by_session: dict[str, dict[str, float]]
    lag_by_session_values: dict[str, dict[str, list[float]]]
    locomotion_progress_by_animal_values: dict[str, list[float]]
    pupil_zscore_progress_by_animal_values: dict[str, list[float]]
    progress_bins: np.ndarray
    state_probability: np.ndarray
    state_probability_std: np.ndarray
    thresholds: dict
    zscore_mean: float
    zscore_std: float
    plot_version: int = STATISTICS_PLOT_VERSION
    state_probability_count: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    progress_series: dict[str, dict] = field(default_factory=dict)
    pupil_state_fraction_overall: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "animal_id": self.animal_id,
            "generated_at": self.generated_at,
            "plot_version": int(self.plot_version),
            "session_ids": self.session_ids,
            "eligible_session_ids": self.eligible_session_ids,
            "session_labels": self.session_labels,
            "locomotion_pct_by_session": self.locomotion_pct_by_session,
            "locomotion_pct_by_session_values": self.locomotion_pct_by_session_values,
            "face_motion_pct_by_session": self.face_motion_pct_by_session,
            "face_motion_pct_by_session_values": self.face_motion_pct_by_session_values,
            "face_motion_mean_by_state": self.face_motion_mean_by_state,
            "face_motion_std_by_state": self.face_motion_std_by_state,
            "pupil_zscore_mean_by_state_values": self.pupil_zscore_mean_by_state_values,
            "pupil_pct_by_session": self.pupil_pct_by_session,
            "pupil_pct_by_session_values": self.pupil_pct_by_session_values,
            "pupil_pct_by_session_visible": self.pupil_pct_by_session_visible,
            "pupil_pct_by_session_visible_values": self.pupil_pct_by_session_visible_values,
            "lag_by_session": self.lag_by_session,
            "lag_by_session_values": self.lag_by_session_values,
            "locomotion_progress_by_animal_values": self.locomotion_progress_by_animal_values,
            "pupil_zscore_progress_by_animal_values": self.pupil_zscore_progress_by_animal_values,
            "progress_bins": self.progress_bins.tolist(),
            "state_probability": self.state_probability.tolist(),
            "state_probability_std": self.state_probability_std.tolist(),
            "state_probability_count": self.state_probability_count.tolist(),
            "progress_series": self.progress_series,
            "pupil_state_fraction_overall": self.pupil_state_fraction_overall,
            "thresholds": self.thresholds,
            "zscore_mean": float(self.zscore_mean),
            "zscore_std": float(self.zscore_std),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StatisticsResult":
        session_labels = list(data.get("session_labels", data.get("day_labels", [])))
        session_ids = list(data.get("session_ids", session_labels))
        eligible_session_ids = list(data.get("eligible_session_ids", session_ids))
        state_count = len(STATE_LABELS)

        def _state_map(raw: dict) -> dict[str, float]:
            mapped = {str(label): float(raw.get(label, float("nan"))) for label in STATE_LABELS}
            for key, value in dict(raw).items():
                mapped.setdefault(str(key), float(value))
            return mapped

        def _series_map(raw: dict) -> dict[str, list[float]]:
            mapped = {str(label): [float(v) for v in list(raw.get(label, []))] for label in STATE_LABELS}
            for key, value in dict(raw).items():
                mapped.setdefault(str(key), [float(v) for v in list(value)])
            return mapped

        def _pad_state_array(values) -> np.ndarray:
            arr = np.asarray(values, dtype=float)
            if arr.ndim != 2:
                return np.full((state_count, 0), np.nan, dtype=float)
            if arr.shape[0] == state_count:
                return arr
            padded = np.full((state_count, arr.shape[1]), np.nan, dtype=float)
            rows = min(state_count, arr.shape[0])
            padded[:rows, :arr.shape[1]] = arr[:rows, :]
            return padded

        pupil_pct_by_session = {
            str(session): {str(label): float(val) for label, val in _state_map(dict(label_dict)).items()}
            for session, label_dict in dict(data.get("pupil_pct_by_session", data.get("pupil_pct_by_day", {}))).items()
        }
        pupil_pct_by_session_values = {
            str(session): _series_map(dict(label_dict))
            for session, label_dict in dict(data.get("pupil_pct_by_session_values", data.get("pupil_pct_by_day_values", {}))).items()
        }
        pupil_pct_by_session_visible = {
            str(session): _normalize_visible_state_map(dict(label_dict))
            for session, label_dict in dict(data.get("pupil_pct_by_session_visible", pupil_pct_by_session)).items()
        }
        pupil_pct_by_session_visible_values = {
            str(session): _normalize_visible_state_value_lists(dict(label_dict))
            for session, label_dict in dict(data.get("pupil_pct_by_session_visible_values", pupil_pct_by_session_values)).items()
        }

        return cls(
            scope=str(data.get("scope", "")),
            animal_id=str(data.get("animal_id", "")),
            generated_at=str(data.get("generated_at", "")),
            session_ids=session_ids,
            eligible_session_ids=eligible_session_ids,
            session_labels=session_labels,
            locomotion_pct_by_session={str(k): float(v) for k, v in dict(data.get("locomotion_pct_by_session", data.get("locomotion_pct_by_day", {}))).items()},
            locomotion_pct_by_session_values={
                str(k): [float(v) for v in list(vals)]
                for k, vals in dict(data.get("locomotion_pct_by_session_values", data.get("locomotion_pct_by_day_values", {}))).items()
            },
            face_motion_pct_by_session={str(k): float(v) for k, v in dict(data.get("face_motion_pct_by_session", data.get("face_motion_pct_by_day", {}))).items()},
            face_motion_pct_by_session_values={
                str(k): [float(v) for v in list(vals)]
                for k, vals in dict(data.get("face_motion_pct_by_session_values", data.get("face_motion_pct_by_day_values", {}))).items()
            },
            face_motion_mean_by_state=_state_map(dict(data.get("face_motion_mean_by_state", {}))),
            face_motion_std_by_state=_state_map(dict(data.get("face_motion_std_by_state", {}))),
            pupil_zscore_mean_by_state_values=_series_map(dict(data.get("pupil_zscore_mean_by_state_values", {}))),
            pupil_pct_by_session=pupil_pct_by_session,
            pupil_pct_by_session_values=pupil_pct_by_session_values,
            pupil_pct_by_session_visible=pupil_pct_by_session_visible,
            pupil_pct_by_session_visible_values=pupil_pct_by_session_visible_values,
            lag_by_session={
                str(exp_id): {str(label): float(val) for label, val in _state_map(dict(label_dict)).items()}
                for exp_id, label_dict in dict(data.get("lag_by_session", {})).items()
            },
            lag_by_session_values={
                str(session): {
                    str(label): [float(v) for v in list(vals)]
                    for label, vals in dict(label_dict).items()
                }
                for session, label_dict in dict(data.get("lag_by_session_values", {})).items()
            },
            locomotion_progress_by_animal_values={
                str(animal_id): [float(v) for v in list(vals)]
                for animal_id, vals in dict(data.get("locomotion_progress_by_animal_values", data.get("locomotion_progress_by_animal", {}))).items()
            },
            pupil_zscore_progress_by_animal_values={
                str(animal_id): [float(v) for v in list(vals)]
                for animal_id, vals in dict(data.get("pupil_zscore_progress_by_animal_values", data.get("pupil_zscore_progress_by_animal", {}))).items()
            },
            progress_bins=np.asarray(data.get("progress_bins", []), dtype=float),
            state_probability=_pad_state_array(data.get("state_probability", [])),
            state_probability_std=_pad_state_array(data.get("state_probability_std", [])),
            thresholds=dict(data.get("thresholds", {})),
            zscore_mean=float(data.get("zscore_mean", 0.0)),
            zscore_std=float(data.get("zscore_std", 1.0)),
            plot_version=int(data.get("plot_version", 1)),
            state_probability_count=np.asarray(data.get("state_probability_count", []), dtype=float).reshape(-1),
            progress_series=dict(data.get("progress_series", {})),
            pupil_state_fraction_overall={
                str(key): {str(label): float(value) for label, value in dict(values).items()}
                for key, values in dict(data.get("pupil_state_fraction_overall", {})).items()
                if isinstance(values, dict)
            },
        )


def _visible_mask(bundle: SessionBundle, manual_masks: list[tuple[float, float]] | None = None) -> np.ndarray:
    mask = bundle.visible_mask_base.copy()
    manual_masks = manual_masks or []
    for start, end in manual_masks:
        if end <= start:
            continue
        mask[(bundle.t >= start) & (bundle.t < end)] = False
    return mask


def _interval_mask(times: np.ndarray, intervals: list[tuple[float, float]] | None = None, *, pad: float = 0.0) -> np.ndarray:
    mask = np.zeros(times.shape, dtype=bool)
    intervals = intervals or []
    for start, end in intervals:
        if end <= start:
            continue
        mask[(times >= start - pad) & (times < end + pad)] = True
    return mask


def _mask_to_intervals(times: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    times = np.asarray(times, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if times.size == 0 or mask.size == 0:
        return []
    n = min(times.size, mask.size)
    times = times[:n]
    mask = mask[:n]
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    split_points = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, split_points)
    finite_times = times[np.isfinite(times)]
    step = float(np.nanmedian(np.diff(finite_times))) if finite_times.size > 1 else 0.0
    if not np.isfinite(step) or step < 0.0:
        step = 0.0
    intervals: list[tuple[float, float]] = []
    for segment in segments:
        if segment.size == 0:
            continue
        start = float(times[segment[0]])
        end = float(times[segment[-1]]) + step
        intervals.append((start, end))
    return intervals


def _longest_interval(intervals: list[tuple[float, float]]) -> tuple[float, float] | None:
    best: tuple[float, float] | None = None
    best_duration = -np.inf
    for start, end in intervals:
        if end <= start:
            continue
        duration = float(end - start)
        if duration > best_duration:
            best = (float(start), float(end))
            best_duration = duration
    return best


def learn_extra_large_reference_band(
    times: np.ndarray,
    pupil_brightness: np.ndarray,
    interval: tuple[float, float],
) -> dict | None:
    times = np.asarray(times, dtype=float).reshape(-1)
    pupil_brightness = np.asarray(pupil_brightness, dtype=float).reshape(-1)
    if times.size == 0 or pupil_brightness.size == 0:
        return None
    n = min(times.size, pupil_brightness.size)
    times = times[:n]
    pupil_brightness = pupil_brightness[:n]
    try:
        start = float(interval[0])
        end = float(interval[1])
    except Exception:
        return None
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return None
    cal_mask = (times >= start) & (times < end) & np.isfinite(pupil_brightness)
    if not np.any(cal_mask):
        return None
    reference_values = pupil_brightness[cal_mask]
    reference_mean = float(np.nanmean(reference_values))
    reference_std = float(np.nanstd(reference_values))
    if not np.isfinite(reference_mean) or not np.isfinite(reference_std):
        return None
    band_low = float(np.clip(reference_mean - reference_std, 0.0, 255.0))
    band_high = float(np.clip(reference_mean + reference_std, 0.0, 255.0))
    if band_high < band_low:
        band_low, band_high = band_high, band_low
    return {
        "interval": (float(start), float(end)),
        "reference_mean": reference_mean,
        "reference_std": reference_std,
        "reference_band": (band_low, band_high),
        "selected_frames": int(np.sum(cal_mask)),
    }


def learn_extra_large_calibration(
    times: np.ndarray,
    pupil_brightness: np.ndarray,
    similarity: np.ndarray,
    interval: tuple[float, float],
) -> dict | None:
    reference = learn_extra_large_reference_band(times, pupil_brightness, interval)
    if reference is None:
        return None
    times = np.asarray(times, dtype=float).reshape(-1)
    similarity = np.asarray(similarity, dtype=float).reshape(-1)
    n = min(times.size, similarity.size, np.asarray(pupil_brightness, dtype=float).reshape(-1).size)
    times = times[:n]
    similarity = similarity[:n]
    try:
        start = float(interval[0])
        end = float(interval[1])
    except Exception:
        return None
    cal_mask = (times >= start) & (times < end) & np.isfinite(similarity)
    if not np.any(cal_mask):
        return None
    similarity_values = similarity[cal_mask]
    similarity_mean = float(np.nanmean(similarity_values))
    similarity_std = float(np.nanstd(similarity_values))
    if not np.isfinite(similarity_mean) or not np.isfinite(similarity_std):
        return None
    reference["similarity_mean"] = similarity_mean
    reference["similarity_std"] = similarity_std
    reference["similarity_cutoff"] = float(np.clip(similarity_mean - similarity_std, 0.0, 1.0))
    reference["selected_frames"] = int(np.sum(cal_mask))
    reference["confirmed"] = True
    return reference


def build_extra_large_mask(
    store: HabituationStore,
    bundle: SessionBundle,
    calibration: dict | None,
    manual_masks: list[tuple[float, float]] | None = None,
    *,
    manual_buffer_sec: float = 1.0,
) -> np.ndarray:
    times = np.asarray(bundle.t, dtype=float).reshape(-1)
    radius = np.asarray(bundle.radius, dtype=float).reshape(-1)
    if times.size == 0 or radius.size == 0:
        return np.zeros(times.shape, dtype=bool)
    n = min(times.size, radius.size)
    times = times[:n]
    radius = radius[:n]
    observed = np.asarray(getattr(bundle, "frame_observed", np.ones(times.shape, dtype=bool)), dtype=bool).reshape(-1)
    observed = observed[:n] if observed.size else np.ones(times.shape, dtype=bool)
    manual_mask = _interval_mask(times, manual_masks, pad=manual_buffer_sec)
    missing = ~np.isfinite(radius) & observed
    candidate = missing & ~manual_mask
    extra_large = np.zeros(times.shape, dtype=bool)
    if not np.any(candidate):
        return extra_large

    working = dict(calibration or {})
    
    confirmed = bool(working.get("confirmed", False))
    if not confirmed:
        return _extra_large_missing_mask(
            bundle,
            manual_masks,
            min_duration_sec=MIN_EXTRA_LARGE_MISSING_SEC,
            manual_buffer_sec=manual_buffer_sec,
        )
    interval = working.get("interval")
    reference_band = working.get("reference_band")
    similarity_cutoff = working.get("similarity_cutoff")
    if (reference_band is None or similarity_cutoff is None) and interval is not None:
        brightness_t, brightness = store.load_pupil_brightness(bundle.summary.exp_id)
        if brightness_t is not None and brightness is not None and brightness.size:
            reference = learn_extra_large_reference_band(brightness_t, brightness, interval)
            if reference is not None:
                sim_t, similarity = store.load_eye_similarity(bundle.summary.exp_id, reference["reference_band"])
                if sim_t is not None and similarity is not None and similarity.size:
                    learned = learn_extra_large_calibration(brightness_t, brightness, similarity, interval)
                    if learned is not None:
                        working = learned
                        reference_band = learned.get("reference_band")
                        similarity_cutoff = learned.get("similarity_cutoff")
    if reference_band is None or similarity_cutoff is None:
        return _extra_large_missing_mask(bundle, manual_masks, min_duration_sec=MIN_EXTRA_LARGE_MISSING_SEC, manual_buffer_sec=manual_buffer_sec)
    try:
        band_low, band_high = float(reference_band[0]), float(reference_band[1])
    except Exception:
        return _extra_large_missing_mask(bundle, manual_masks, min_duration_sec=MIN_EXTRA_LARGE_MISSING_SEC, manual_buffer_sec=manual_buffer_sec)
    if band_high < band_low:
        band_low, band_high = band_high, band_low
    similarity_t, similarity = store.load_eye_similarity(bundle.summary.exp_id, (band_low, band_high))
    if similarity_t is None or similarity is None or not similarity.size:
        return _extra_large_missing_mask(bundle, manual_masks, min_duration_sec=MIN_EXTRA_LARGE_MISSING_SEC, manual_buffer_sec=manual_buffer_sec)
    n = min(times.size, similarity.size)
    times = times[:n]
    radius = radius[:n]
    similarity = similarity[:n]
    manual_mask = manual_mask[:n]
    observed = observed[:n] if observed.size else np.ones(times.shape, dtype=bool)
    missing = (~np.isfinite(radius)) & observed
    candidate = missing & ~manual_mask & np.isfinite(similarity)
    extra_large = np.zeros(times.shape, dtype=bool)
    extra_large[candidate & (similarity >= float(similarity_cutoff))] = True
    return extra_large


def _calibration_threshold_from_interval(
    times: np.ndarray,
    pupil_brightness: np.ndarray,
    interval: tuple[float, float],
    *,
    margin_fraction: float = CALIBRATION_BRIGHTNESS_MARGIN,
) -> dict | None:
    times = np.asarray(times, dtype=float).reshape(-1)
    pupil_brightness = np.asarray(pupil_brightness, dtype=float).reshape(-1)
    if times.size == 0 or pupil_brightness.size == 0:
        return None
    n = min(times.size, pupil_brightness.size)
    times = times[:n]
    pupil_brightness = pupil_brightness[:n]
    try:
        start = float(interval[0])
        end = float(interval[1])
    except Exception:
        return None
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return None
    cal_mask = (times >= start) & (times < end) & np.isfinite(pupil_brightness)
    if not np.any(cal_mask):
        return None
    reference_mean = float(np.nanmean(pupil_brightness[cal_mask]))
    if not np.isfinite(reference_mean):
        return None
    return {
        "interval": (float(start), float(end)),
        "reference_mean": reference_mean,
        "threshold": float(reference_mean * (1.0 + float(margin_fraction))),
        "margin_fraction": float(margin_fraction),
        "selected_frames": int(np.sum(cal_mask)),
    }


def suggest_extra_large_calibration(
    times: np.ndarray,
    radius: np.ndarray,
    visible_mask: np.ndarray | None,
    pupil_brightness: np.ndarray,
    zscore_mean: float,
    zscore_std: float,
    threshold_values: list[float],
    *,
    margin_fraction: float = CALIBRATION_BRIGHTNESS_MARGIN,
) -> dict | None:
    times = np.asarray(times, dtype=float).reshape(-1)
    radius = np.asarray(radius, dtype=float).reshape(-1)
    pupil_brightness = np.asarray(pupil_brightness, dtype=float).reshape(-1)
    if times.size == 0 or radius.size == 0 or pupil_brightness.size == 0:
        return None
    n = min(times.size, radius.size, pupil_brightness.size)
    times = times[:n]
    radius = radius[:n]
    pupil_brightness = pupil_brightness[:n]
    if visible_mask is None:
        visible = np.isfinite(radius)
    else:
        visible = np.asarray(visible_mask, dtype=bool).reshape(-1)[:n]
    thresholds = sorted([float(t) for t in threshold_values])
    if len(thresholds) < 3:
        return None
    std = float(zscore_std)
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    z = (radius - float(zscore_mean)) / std
    z[~np.isfinite(z)] = np.nan
    base_mask = visible & np.isfinite(z)
    candidate_source = "large"
    candidate_mask = base_mask & (z >= thresholds[2])
    if not np.any(candidate_mask):
        finite_visible = base_mask
        if not np.any(finite_visible):
            return None
        fallback = float(np.nanpercentile(z[finite_visible], 75.0))
        if not np.isfinite(fallback):
            return None
        candidate_mask = finite_visible & (z >= fallback)
        candidate_source = "upper_quartile"
    intervals = _mask_to_intervals(times, candidate_mask)
    interval = _longest_interval(intervals)
    if interval is None:
        return None
    calibration = _calibration_threshold_from_interval(
        times,
        pupil_brightness,
        interval,
        margin_fraction=margin_fraction,
    )
    if calibration is None:
        return None
    calibration["candidate_source"] = candidate_source
    return calibration


def _extra_large_missing_mask(
    bundle: SessionBundle,
    manual_masks: list[tuple[float, float]] | None = None,
    *,
    min_duration_sec: float = MIN_EXTRA_LARGE_MISSING_SEC,
    manual_buffer_sec: float = 1.0,
) -> np.ndarray:
    times = np.asarray(bundle.t, dtype=float)
    if times.size == 0:
        return np.zeros(0, dtype=bool)

    manual_mask = _interval_mask(times, manual_masks, pad=manual_buffer_sec)
    observed = np.asarray(getattr(bundle, "frame_observed", np.ones(times.shape, dtype=bool)), dtype=bool).reshape(-1)
    observed = observed[: times.size] if observed.size else np.ones(times.shape, dtype=bool)
    radius = np.asarray(bundle.radius, dtype=float).reshape(-1)
    missing = (~np.isfinite(radius[: times.size])) & observed
    candidate = missing & ~manual_mask
    extra_large = np.zeros(times.shape, dtype=bool)
    if not np.any(candidate):
        return extra_large

    candidate_idx = np.flatnonzero(candidate)
    if candidate_idx.size == 0:
        return extra_large

    split_points = np.where(np.diff(candidate_idx) > 1)[0] + 1
    finite_times = times[np.isfinite(times)]
    frame_step = float(np.nanmedian(np.diff(finite_times))) if finite_times.size > 1 else 0.0
    if not np.isfinite(frame_step) or frame_step < 0.0:
        frame_step = 0.0
    for segment in np.split(candidate_idx, split_points):
        if segment.size == 0:
            continue
        start_idx = int(segment[0])
        end_idx = int(segment[-1])
        if times.size == 1:
            duration = 0.0
        else:
            duration = float(times[end_idx] - times[start_idx] + frame_step)
        if duration >= float(min_duration_sec):
            extra_large[segment] = True
    return extra_large


def _scope_sessions(store: HabituationStore, scope: str, animal_id: str):
    if scope == "All" or animal_id == "All":
        return list(store.dataset_sessions())
    return list(store.sessions_for_animal(animal_id))


def _analysis_sessions(store: HabituationStore, scope: str, animal_id: str) -> list:
    sessions = _scope_sessions(store, scope, animal_id)
    analysis_sessions = [
        s
        for s in sessions
        if s.has_right_pickle and not store.is_deeplabcut_reference_session(s.exp_id) and not store.is_session_do_not_use(s.exp_id)
    ]
    counts_by_animal: dict[str, int] = {}
    for summary in analysis_sessions:
        counts_by_animal[summary.animal_id] = counts_by_animal.get(summary.animal_id, 0) + 1
    allowed_animals = {
        animal_key
        for animal_key, count in counts_by_animal.items()
        if count >= MIN_STATISTICS_SESSIONS_PER_ANIMAL
    }
    if scope == "All" or animal_id == "All":
        return [summary for summary in analysis_sessions if summary.animal_id in allowed_animals]
    return analysis_sessions


def compute_animal_baseline(store: HabituationStore, animal_id: str, *, scope: str | None = None) -> tuple[float, float]:
    analysis_scope = "All" if animal_id == "All" or scope == "All" else animal_id
    sessions = _analysis_sessions(store, analysis_scope, animal_id)
    values = []
    for summary in sessions:
        bundle = _trim_bundle_for_cutoff(store, store.load_session_bundle(summary.exp_id))
        masks = store.load_manual_masks(summary.exp_id)
        visible = _visible_mask(bundle, masks)
        vals = bundle.radius[visible & np.isfinite(bundle.radius)]
        if vals.size:
            values.append(vals.astype(float))
    if not values:
        return 0.0, 1.0
    all_vals = np.concatenate(values)
    mean = float(np.nanmean(all_vals))
    std = float(np.nanstd(all_vals))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return mean, std


def animal_zscores(store: HabituationStore, animal_id: str, *, mean: float | None = None, std: float | None = None) -> dict[str, np.ndarray]:
    if mean is None or std is None:
        mean, std = compute_animal_baseline(store, animal_id)
    out = {}
    if animal_id == "All":
        sessions = store.dataset_sessions()
    else:
        sessions = store.sessions_for_animal(animal_id)
    for summary in sessions:
        if (
            not summary.has_right_pickle
            or store.is_deeplabcut_reference_session(summary.exp_id)
            or store.is_session_do_not_use(summary.exp_id)
        ):
            continue
        bundle = _trim_bundle_for_cutoff(store, store.load_session_bundle(summary.exp_id))
        z = (bundle.radius.astype(float) - float(mean)) / float(std)
        z[~np.isfinite(z)] = np.nan
        out[summary.exp_id] = z
    return out


def percentile_threshold_values(values: np.ndarray, percentiles: list[float]) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0.0 for _ in percentiles]
    return [float(np.nanpercentile(finite, pct)) for pct in percentiles]


def percentile_from_value(values: np.ndarray, value: float) -> float:
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return 0.0
    idx = int(np.searchsorted(finite, value, side="left"))
    return float(100.0 * idx / max(1, finite.size))


def classify_zscores(
    z: np.ndarray,
    thresholds: list[float],
    visible_mask: np.ndarray | None = None,
    extra_large_mask: np.ndarray | None = None,
    not_visible_mask: np.ndarray | None = None,
) -> np.ndarray:
    thresholds = sorted([float(t) for t in thresholds])
    state = np.full(z.shape, -1, dtype=int)
    mask = np.isfinite(z)
    if visible_mask is not None:
        mask &= visible_mask
    state[mask & (z < thresholds[0])] = 0
    state[mask & (z >= thresholds[0]) & (z < thresholds[1])] = 1
    state[mask & (z >= thresholds[1]) & (z < thresholds[2])] = 2
    state[mask & (z >= thresholds[2])] = 3
    if extra_large_mask is not None:
        # Promote long invisible stretches to the inferred extra-large state.
        state[np.asarray(extra_large_mask, dtype=bool)] = 3
    if not_visible_mask is not None:
        # Manual not-visible intervals stay separate from inferred missing stretches.
        state[np.asarray(not_visible_mask, dtype=bool)] = 4
    return state


def robust_face_motion_threshold(motion: np.ndarray) -> float:
    finite = motion[np.isfinite(motion)]
    if finite.size == 0:
        return 0.0
    med = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - med)))
    return med + 3.0 * max(mad, 1e-6)


def _session_order_map(sessions) -> dict[str, int]:
    grouped: dict[str, list] = {}
    for summary in sorted(sessions, key=lambda s: s.sort_key):
        grouped.setdefault(summary.animal_id, []).append(summary)
    mapping: dict[str, int] = {}
    for animal_sessions in grouped.values():
        for idx, summary in enumerate(animal_sessions, start=1):
            mapping[summary.exp_id] = idx
    return mapping


def _session_labels(session_order_map: dict[str, int]) -> list[str]:
    if not session_order_map:
        return []
    return [str(idx) for idx in range(1, max(session_order_map.values()) + 1)]


def _nanmean_or_nan(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _nanstd_or_nan(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.std(finite))


def _trim_bundle_for_cutoff(store: HabituationStore, bundle: SessionBundle) -> SessionBundle:
    cutoff = store.session_analysis_cutoff_sec(bundle.summary.exp_id)
    if cutoff is None:
        return bundle
    frame_mask = analysis_cutoff_mask(bundle.t, cutoff)
    locomotion_mask = analysis_cutoff_mask(bundle.locomotion_t, cutoff)
    brake_mask = analysis_cutoff_mask(bundle.brake_t, cutoff)
    return SessionBundle(
        summary=bundle.summary,
        t=apply_time_mask(bundle.t, frame_mask),
        frame_observed=apply_time_mask(bundle.frame_observed, frame_mask),
        radius=apply_time_mask(bundle.radius, frame_mask),
        x=apply_time_mask(bundle.x, frame_mask),
        y=apply_time_mask(bundle.y, frame_mask),
        velocity=apply_time_mask(bundle.velocity, frame_mask),
        qc=apply_time_mask(bundle.qc, frame_mask),
        in_eye=apply_time_mask(bundle.in_eye, frame_mask),
        brake_raw=apply_time_mask(bundle.brake_raw, brake_mask),
        brake_t=apply_time_mask(bundle.brake_t, brake_mask),
        wheel_pos=apply_time_mask(bundle.wheel_pos, locomotion_mask),
        locomotion=apply_time_mask(bundle.locomotion, locomotion_mask),
        locomotion_t=apply_time_mask(bundle.locomotion_t, locomotion_mask),
        eye_lid_x=apply_time_mask(bundle.eye_lid_x, frame_mask),
        eye_lid_y=apply_time_mask(bundle.eye_lid_y, frame_mask),
        eyeX=apply_time_mask(bundle.eyeX, frame_mask),
        eyeY=apply_time_mask(bundle.eyeY, frame_mask),
        pupilX=apply_time_mask(bundle.pupilX, frame_mask),
        pupilY=apply_time_mask(bundle.pupilY, frame_mask),
        source_signature=bundle.source_signature,
    )


def compute_statistics(
    store: HabituationStore,
    *,
    scope: str,
    animal_id: str,
    percentiles: list[float],
    threshold_values: list[float],
    locomotion_threshold: float,
    missing_buffer_sec: float = 1.0,
    progress_cb=None,
) -> StatisticsResult:
    sessions = sorted(_analysis_sessions(store, scope, animal_id), key=lambda s: s.sort_key)
    eligible = [s for s in sessions if store.effective_session_duration_sec(s.exp_id) >= 1800.0]
    mean, std = compute_animal_baseline(store, animal_id, scope=scope)

    session_order_map = _session_order_map(sessions)
    session_labels = _session_labels(session_order_map)
    locomotion_values_by_session: dict[str, list[float]] = {label: [] for label in session_labels}
    face_motion_values_by_session: dict[str, list[float]] = {label: [] for label in session_labels}
    face_motion_by_state_values: dict[str, list[float]] = {label: [] for label in STATE_LABELS}
    pupil_zscore_mean_by_state_values: dict[str, list[float]] = {label: [] for label in STATE_LABELS}
    locomotion_progress_by_animal_values: dict[str, list[float]] = {}
    pupil_zscore_progress_by_animal_values: dict[str, list[float]] = {}
    pupil_values_by_session: dict[str, dict[str, list[float]]] = {
        label: {state_label: [] for state_label in STATE_LABELS} for label in session_labels
    }
    pupil_visible_values_by_session: dict[str, dict[str, list[float]]] = {
        label: {state_label: [] for state_label in STATE_LABELS} for label in session_labels
    }
    lag_values_by_session: dict[str, dict[str, list[float]]] = {
        label: {state_label: [] for state_label in STATE_LABELS} for label in session_labels
    }

    total = max(1, len(sessions))
    for idx, summary in enumerate(sessions):
        if progress_cb:
            progress_cb(idx / total, f"Processing {summary.exp_id}")
        bundle = _trim_bundle_for_cutoff(store, store.load_session_bundle(summary.exp_id))
        manual_masks = store.load_manual_masks(summary.exp_id)
        visible = _visible_mask(bundle, manual_masks)
        not_visible_mask = _interval_mask(np.asarray(bundle.t, dtype=float), manual_masks)

        calibration = store.session_extra_large_calibration(summary.exp_id)
        extra_large_mask = build_extra_large_mask(
            store,
            bundle,
            calibration,
            manual_masks,
            manual_buffer_sec=missing_buffer_sec,
        )

        z = (bundle.radius.astype(float) - mean) / std
        z[~np.isfinite(z)] = np.nan
        state = classify_zscores(z, threshold_values, visible, extra_large_mask, not_visible_mask)

        session_label = str(session_order_map.get(summary.exp_id, 0))

        valid_motion = np.isfinite(bundle.locomotion)
        locomotion_pct = float(np.nanmean((bundle.locomotion[valid_motion] > locomotion_threshold).astype(float))) if np.any(valid_motion) else float("nan")
        locomotion_values_by_session.setdefault(session_label, []).append(locomotion_pct)
        locomotion_progress_by_animal_values.setdefault(summary.animal_id, []).append(locomotion_pct)

        face_t, face_motion = store.load_face_motion(summary.exp_id)

        if face_t is not None and face_motion is not None and face_motion.size:
            face_t = np.asarray(face_t, dtype=float)
            face_motion = np.asarray(face_motion, dtype=float)

            face_cutoff = store.session_analysis_cutoff_sec(summary.exp_id)
            if face_cutoff is not None:
                face_mask = analysis_cutoff_mask(face_t, face_cutoff)
                face_t = apply_time_mask(face_t, face_mask)
                face_motion = apply_time_mask(face_motion, face_mask)

            target_t = np.asarray(bundle.t, dtype=float)
            observed = np.asarray(bundle.frame_observed, dtype=bool)

            aligned_face_motion = np.full(target_t.shape, np.nan, dtype=float)

            valid_src = np.isfinite(face_t) & np.isfinite(face_motion)

            if np.sum(valid_src) >= 2:
                src_t = face_t[valid_src]
                src_motion = face_motion[valid_src]

                in_range = (
                    np.isfinite(target_t)
                    & (target_t >= np.nanmin(src_t))
                    & (target_t <= np.nanmax(src_t))
                )

                aligned_face_motion[in_range] = np.interp(
                    target_t[in_range],
                    src_t,
                    src_motion,
                )

            # Do not count inserted dropped-frame placeholders
            aligned_face_motion[~observed] = np.nan

            face_thr = robust_face_motion_threshold(aligned_face_motion)

            valid_face = np.isfinite(aligned_face_motion)
            face_pct = (
                float(np.nanmean((aligned_face_motion[valid_face] > face_thr).astype(float)))
                if np.any(valid_face)
                else float("nan")
            )

            aligned_state = np.asarray(state, dtype=int)

            for state_id, state_label in enumerate(STATE_LABELS):
                mask = valid_face & (aligned_state == state_id)
                if np.any(mask):
                    face_motion_by_state_values[state_label].append(
                        float(np.nanmean(aligned_face_motion[mask]))
                    )
        else:
            face_pct = float("nan")

        face_motion_values_by_session.setdefault(session_label, []).append(face_pct)
        valid_radius = np.isfinite(z)
        if np.any(valid_radius):
            pupil_zscore_progress_by_animal_values.setdefault(summary.animal_id, []).append(float(np.nanmean(z[valid_radius])))
        else:
            pupil_zscore_progress_by_animal_values.setdefault(summary.animal_id, []).append(float("nan"))
        for state_id, state_label in enumerate(STATE_LABELS):
            mask = valid_radius & (state == state_id)
            if np.any(mask):
                pupil_zscore_mean_by_state_values[state_label].append(float(np.nanmean(z[mask])))
        finite_visible = state >= 0
        denom = float(np.sum(finite_visible)) if np.sum(finite_visible) else 1.0
        visible_only_mask = finite_visible & (state != STATE_LABELS.index("not_visible"))
        visible_denom = float(np.sum(visible_only_mask))
        for state_id, state_label in enumerate(STATE_LABELS):
            pct = float(np.sum(state == state_id) / denom)
            pupil_values_by_session.setdefault(session_label, {label: [] for label in STATE_LABELS})[state_label].append(pct)
            if state_label == "not_visible":
                visible_pct = float("nan")
            elif visible_denom > 0.0:
                visible_pct = float(np.sum(state == state_id) / visible_denom)
            else:
                visible_pct = float("nan")
            pupil_visible_values_by_session.setdefault(session_label, {label: [] for label in STATE_LABELS})[state_label].append(visible_pct)

        first_minute = bundle.t >= 60.0
        for state_id, state_label in enumerate(STATE_LABELS):
            idxs = np.where(first_minute & (state == state_id))[0]
            if idxs.size:
                lag_values_by_session.setdefault(session_label, {label: [] for label in STATE_LABELS})[state_label].append(float(bundle.t[idxs[0]] - 60.0))
            else:
                lag_values_by_session.setdefault(session_label, {label: [] for label in STATE_LABELS})[state_label].append(float("nan"))

    locomotion_pct_by_session = {label: _nanmean_or_nan(values) for label, values in locomotion_values_by_session.items()}
    face_motion_pct_by_session = {label: _nanmean_or_nan(values) for label, values in face_motion_values_by_session.items()}
    face_motion_mean_by_state = {label: _nanmean_or_nan(values) for label, values in face_motion_by_state_values.items()}
    face_motion_std_by_state = {label: _nanstd_or_nan(values) for label, values in face_motion_by_state_values.items()}
    pupil_pct_by_session = {
        label: {state_label: _nanmean_or_nan(values[state_label]) for state_label in STATE_LABELS}
        for label, values in pupil_values_by_session.items()
    }
    pupil_pct_by_session_visible = {
        label: {state_label: _nanmean_or_nan(values[state_label]) for state_label in STATE_LABELS}
        for label, values in pupil_visible_values_by_session.items()
    }
    pupil_state_fraction_overall = _build_overall_pupil_state_fractions(pupil_pct_by_session, pupil_pct_by_session_visible)
    lag_by_session = {
        label: {state_label: _nanmean_or_nan(values[state_label]) for state_label in STATE_LABELS}
        for label, values in lag_values_by_session.items()
    }

    progress_bins = np.linspace(0, 100, 100, endpoint=False)
    n_states = len(STATE_LABELS)
    state_probability_sum = np.zeros((n_states, 100), dtype=float)
    state_probability_sq_sum = np.zeros((n_states, 100), dtype=float)
    state_probability_count = np.zeros((n_states, 100), dtype=float)
    selected_for_progress = eligible or sessions
    for summary in selected_for_progress:
        bundle = _trim_bundle_for_cutoff(store, store.load_session_bundle(summary.exp_id))
        manual_masks = store.load_manual_masks(summary.exp_id)
        visible = _visible_mask(bundle, manual_masks)
        not_visible_mask = _interval_mask(np.asarray(bundle.t, dtype=float), manual_masks)

        calibration = store.session_extra_large_calibration(summary.exp_id)
        extra_large_mask = build_extra_large_mask(
            store,
            bundle,
            calibration,
            manual_masks,
            manual_buffer_sec=missing_buffer_sec,
        )

        z = (bundle.radius.astype(float) - mean) / std
        z[~np.isfinite(z)] = np.nan
        state = classify_zscores(z, threshold_values, visible, extra_large_mask, not_visible_mask)
        progress = np.clip((bundle.t / max(store.effective_session_duration_sec(summary.exp_id), 1e-6)) * 100.0, 0.0, 100.0)
        bins = np.clip(progress.astype(int), 0, 99)
        for b in range(100):
            mask = bins == b
            if not np.any(mask):
                continue
            valid = mask & (state >= 0)
            if not np.any(valid):
                continue
            for state_id in range(n_states):
                fraction = float(np.mean(state[valid] == state_id))
                if np.isfinite(fraction):
                    state_probability_sum[state_id, b] += fraction
                    state_probability_sq_sum[state_id, b] += fraction * fraction
                    state_probability_count[state_id, b] += 1.0
    state_probability = np.divide(
        state_probability_sum,
        state_probability_count,
        out=np.full((n_states, 100), np.nan, dtype=float),
        where=state_probability_count > 0,
    )
    state_probability_var = np.divide(
        state_probability_sq_sum,
        state_probability_count,
        out=np.full((n_states, 100), np.nan, dtype=float),
        where=state_probability_count > 0,
    ) - np.square(state_probability)
    state_probability_std = np.sqrt(np.clip(state_probability_var, 0.0, None))
    if scope == "All" and animal_id == "All":
        overall_series = _build_animal_weighted_progress_series(
            store,
            selected_for_progress,
            mean=mean,
            std=std,
            threshold_values=threshold_values,
            locomotion_threshold=locomotion_threshold,
            missing_buffer_sec=missing_buffer_sec,
        )
    else:
        overall_series = _build_progress_state_series(
            store,
            selected_for_progress,
            mean=mean,
            std=std,
            threshold_values=threshold_values,
            locomotion_threshold=locomotion_threshold,
            missing_buffer_sec=missing_buffer_sec,
            bin_count=100,
        )
    progress_series = {"overall": overall_series}
    if len(selected_for_progress) >= 2:
        if scope == "All" and animal_id == "All":
            progress_series["first_2"] = _build_animal_weighted_progress_subset_series(
                store,
                selected_for_progress,
                mean=mean,
                std=std,
                threshold_values=threshold_values,
                locomotion_threshold=locomotion_threshold,
                missing_buffer_sec=missing_buffer_sec,
                take_last=False,
            )
            progress_series["last_2"] = _build_animal_weighted_progress_subset_series(
                store,
                selected_for_progress,
                mean=mean,
                std=std,
                threshold_values=threshold_values,
                locomotion_threshold=locomotion_threshold,
                missing_buffer_sec=missing_buffer_sec,
                take_last=True,
            )
        else:
            progress_series["first_2"] = _build_progress_state_series(
                store,
                selected_for_progress[:2],
                mean=mean,
                std=std,
                threshold_values=threshold_values,
                locomotion_threshold=locomotion_threshold,
                missing_buffer_sec=missing_buffer_sec,
                bin_count=100,
            )
            progress_series["last_2"] = _build_progress_state_series(
                store,
                selected_for_progress[-2:],
                mean=mean,
                std=std,
                threshold_values=threshold_values,
                locomotion_threshold=locomotion_threshold,
                missing_buffer_sec=missing_buffer_sec,
                bin_count=100,
            )
    max_duration = max((float(store.effective_session_duration_sec(summary.exp_id)) for summary in selected_for_progress), default=0.0)
    n_windows = int(max_duration // 1800.0)
    sessions_by_id = {summary.exp_id: summary for summary in selected_for_progress}
    for idx in range(n_windows):
        start_sec = float(idx * 1800.0)
        end_sec = float((idx + 1) * 1800.0)
        window_sessions = [
            summary
            for summary in selected_for_progress
            if float(store.effective_session_duration_sec(summary.exp_id)) >= end_sec
        ]
        if not window_sessions:
            continue
        progress_series[f"window_{idx + 1:02d}"] = _build_progress_state_series(
            store,
            window_sessions,
            mean=mean,
            std=std,
            threshold_values=threshold_values,
            locomotion_threshold=locomotion_threshold,
            missing_buffer_sec=missing_buffer_sec,
            bin_count=30,
            window_start_sec=start_sec,
            window_end_sec=end_sec,
        )

    return StatisticsResult(
        scope=scope,
        animal_id=animal_id,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        session_ids=[s.exp_id for s in sessions],
        eligible_session_ids=[s.exp_id for s in eligible],
        session_labels=session_labels,
        locomotion_pct_by_session=locomotion_pct_by_session,
        locomotion_pct_by_session_values=locomotion_values_by_session,
        face_motion_pct_by_session=face_motion_pct_by_session,
        face_motion_pct_by_session_values=face_motion_values_by_session,
        face_motion_mean_by_state=face_motion_mean_by_state,
        face_motion_std_by_state=face_motion_std_by_state,
        pupil_zscore_mean_by_state_values=pupil_zscore_mean_by_state_values,
        pupil_pct_by_session=pupil_pct_by_session,
        pupil_pct_by_session_values=pupil_values_by_session,
        pupil_pct_by_session_visible=pupil_pct_by_session_visible,
        pupil_pct_by_session_visible_values=pupil_visible_values_by_session,
        lag_by_session=lag_by_session,
        lag_by_session_values=lag_values_by_session,
        locomotion_progress_by_animal_values=locomotion_progress_by_animal_values,
        pupil_zscore_progress_by_animal_values=pupil_zscore_progress_by_animal_values,
        progress_bins=progress_bins,
        state_probability=state_probability,
        state_probability_std=state_probability_std,
        state_probability_count=state_probability_count[0].copy(),
        progress_series=progress_series,
        pupil_state_fraction_overall=pupil_state_fraction_overall,
        thresholds={
            "percentiles": percentiles,
            "threshold_values": threshold_values,
            "locomotion_threshold": locomotion_threshold,
            "missing_buffer_sec": missing_buffer_sec,
        },
        zscore_mean=float(mean),
        zscore_std=float(std),
        plot_version=STATISTICS_PLOT_VERSION,
    )



SESSION_INDEX_LIMIT = 6


def _finite_series(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _boxplot(
    ax,
    data: list[np.ndarray],
    labels: list[str],
    *,
    colors: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    sample_sizes: list[int] | None = None,
) -> None:
    if sample_sizes is not None:
        labels = [f"{label}\nn={sample_sizes[idx]}" for idx, label in enumerate(labels)]
    if not any(arr.size for arr in data):
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center", va="center")
        style_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        return
    plot_data = [arr if arr.size else np.array([np.nan], dtype=float) for arr in data]
    bp = ax.boxplot(
        plot_data,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "0.35", "linewidth": 1.2},
        capprops={"color": "0.35", "linewidth": 1.2},
        flierprops={"marker": "o", "markerfacecolor": "0.4", "markeredgecolor": "0.4", "markersize": 3, "alpha": 0.35},
        widths=0.6,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor("0.35")
        patch.set_linewidth(1.2)
    style_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90 if len(labels) > 4 else 0, fontsize=8)


def _session_sample_sizes(values_by_session: dict[str, dict[str, list[float]] | list[float]], sessions: list[str], *, state_label: str | None = None) -> list[int]:
    sample_sizes: list[int] = []
    for session in sessions:
        values = values_by_session.get(session, {})
        if state_label is None:
            if isinstance(values, dict):
                first = next(iter(values.values()), [])
                sample_sizes.append(int(_finite_series(first).size))
            else:
                sample_sizes.append(int(_finite_series(values).size))
        else:
            if isinstance(values, dict):
                sample_sizes.append(int(_finite_series(values.get(state_label, [])).size))
            else:
                sample_sizes.append(int(_finite_series(values).size))
    return sample_sizes


def _session_labels_with_counts(labels: list[str], counts: list[int]) -> list[str]:
    return [f"{label}\nn={count}" for label, count in zip(labels, counts)]


def _session_subset(sessions: list[str], *, limit: int | None = None, first_n: int | None = None, last_n: int | None = None) -> list[str]:
    subset = list(sessions)
    if limit is not None:
        subset = [label for label in subset if label.isdigit() and int(label) <= limit]
    if first_n is not None:
        subset = subset[:first_n]
    if last_n is not None:
        subset = subset[-last_n:]
    return subset


def _state_labels_for_combined_plot() -> list[str]:
    return [label for label in STATE_LABELS if label != "not_visible"]


def _normalize_visible_state_map(values: dict[str, float]) -> dict[str, float]:
    visible_labels = _state_labels_for_combined_plot()
    total = float(sum(float(values.get(label, 0.0)) for label in visible_labels if np.isfinite(values.get(label, np.nan))))
    out: dict[str, float] = {}
    for label in STATE_LABELS:
        value = float(values.get(label, float("nan")))
        if label == "not_visible":
            out[label] = float("nan")
        elif total > 0.0 and np.isfinite(value):
            out[label] = float(value / total)
        else:
            out[label] = float("nan")
    return out


def _normalize_visible_state_value_lists(values: dict[str, list[float]]) -> dict[str, list[float]]:
    visible_labels = _state_labels_for_combined_plot()
    max_len = max((len(list(values.get(label, []))) for label in STATE_LABELS), default=0)
    out = {label: [] for label in STATE_LABELS}
    for idx in range(max_len):
        total = 0.0
        per_label: dict[str, float] = {}
        for label in visible_labels:
            items = list(values.get(label, []))
            value = float(items[idx]) if idx < len(items) else float("nan")
            per_label[label] = value
            if np.isfinite(value):
                total += value
        for label in visible_labels:
            value = per_label.get(label, float("nan"))
            out[label].append(float(value / total) if total > 0.0 and np.isfinite(value) else float("nan"))
        out["not_visible"].append(float("nan"))
    return out


def _pupil_fraction_values(result: StatisticsResult, *, include_not_visible: bool) -> dict[str, dict[str, list[float]]]:
    if include_not_visible:
        return result.pupil_pct_by_session_values
    if result.pupil_pct_by_session_visible_values:
        return result.pupil_pct_by_session_visible_values
    return {
        session: _normalize_visible_state_value_lists(values)
        for session, values in result.pupil_pct_by_session_values.items()
    }


def _build_overall_pupil_state_fractions(
    pupil_pct_by_session: dict[str, dict[str, float]],
    pupil_pct_by_session_visible: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    session_means = {
        label: _nanmean_or_nan([values.get(label, float("nan")) for values in pupil_pct_by_session.values()])
        for label in STATE_LABELS
    }
    visible_source = pupil_pct_by_session_visible or {
        session: _normalize_visible_state_map(values)
        for session, values in pupil_pct_by_session.items()
    }
    visible_only = {
        label: _nanmean_or_nan([values.get(label, float("nan")) for values in visible_source.values()])
        for label in _state_labels_for_combined_plot()
    }
    return {
        "with_not_visible": session_means,
        "without_not_visible": visible_only,
    }


def _plot_state_fraction_pies(ax, *, title: str, fractions: dict[str, dict[str, float]], sample_size: int, sample_size_unit: str = "sessions") -> None:
    ax.set_axis_off()
    ax.set_title(f"{title} (n={sample_size} {sample_size_unit})", pad=14)
    fig = ax.figure
    bbox = ax.get_position()
    bottom = bbox.y0 + 0.03 * bbox.height
    height = bbox.height * 0.83
    width = bbox.width * 0.46
    gap = bbox.width * 0.08
    left_ax = fig.add_axes([bbox.x0, bottom, width, height])
    right_ax = fig.add_axes([bbox.x0 + width + gap, bottom, width, height])

    def _draw_pie(pie_ax, values: dict[str, float], pie_title: str, labels: list[str]) -> None:
        sizes = [float(values.get(label, float("nan"))) for label in labels]
        sizes = [0.0 if not np.isfinite(v) or v < 0.0 else float(v) for v in sizes]
        total = float(np.sum(sizes))
        if total <= 0.0:
            pie_ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=pie_ax.transAxes)
            pie_ax.set_axis_off()
            return
        pie_ax.pie(
            sizes,
            labels=labels,
            colors=[STATE_COLORS[STATE_LABELS.index(label)] for label in labels],
            startangle=90,
            counterclock=False,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 3.0 else "",
            textprops={"fontsize": 14},
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        )
        pie_ax.set_title(pie_title, fontsize=18, pad=8)
        pie_ax.set_aspect("equal")

    _draw_pie(left_ax, fractions.get("with_not_visible", {}), "Including not visible", STATE_LABELS)
    _draw_pie(right_ax, fractions.get("without_not_visible", {}), "Excluding not visible", _state_labels_for_combined_plot())


def _plot_session_metric_boxplot(
    ax,
    result: StatisticsResult,
    sessions: list[str],
    metric_values_by_session: dict[str, list[float]],
    *,
    title: str,
    ylabel: str,
    color: str,
    xlabel: str = "Session",
) -> None:
    data = [_finite_series(metric_values_by_session.get(session, [])) for session in sessions]
    counts = [int(arr.size) for arr in data]
    _boxplot(
        ax,
        data,
        sessions,
        colors=[color] * max(1, len(sessions)),
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        sample_sizes=counts,
    )
    if ylabel == "Fraction":
        ax.set_ylim(0.0, 1.0)


def _plot_state_fraction_by_session(
    ax,
    result: StatisticsResult,
    sessions: list[str],
    *,
    title: str,
    labels: list[str],
    include_not_visible: bool = True,
) -> None:
    values_by_session = _pupil_fraction_values(result, include_not_visible=include_not_visible)
    x = np.arange(len(sessions), dtype=float)
    reference_label = labels[0] if labels else STATE_LABELS[0]
    sample_sizes = _session_sample_sizes(values_by_session, sessions, state_label=reference_label)
    tick_labels = _session_labels_with_counts(sessions, sample_sizes)
    for label in labels:
        session_values = [
            _finite_series(values_by_session.get(session, {}).get(label, []))
            for session in sessions
        ]
        means = np.array([float(np.nanmean(vals)) if vals.size else np.nan for vals in session_values], dtype=float)
        sds = np.array([float(np.nanstd(vals)) if vals.size else np.nan for vals in session_values], dtype=float)
        lower = np.clip(means - sds, 0.0, 1.0)
        upper = np.clip(means + sds, 0.0, 1.0)
        ax.fill_between(x, lower, upper, color=STATE_COLORS[STATE_LABELS.index(label)], alpha=0.18)
        ax.plot(x, means, marker="o", color=STATE_COLORS[STATE_LABELS.index(label)], label=label)
    style_axes(ax, title=title, xlabel="Session", ylabel="Fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))


def _plot_state_fraction_stacked_by_session(
    ax,
    result: StatisticsResult,
    sessions: list[str],
    *,
    title: str,
    labels: list[str],
    include_not_visible: bool = False,
) -> None:
    values_by_session = _pupil_fraction_values(result, include_not_visible=include_not_visible)
    x = np.arange(len(sessions), dtype=float)
    reference_label = labels[0] if labels else STATE_LABELS[0]
    sample_sizes = _session_sample_sizes(values_by_session, sessions, state_label=reference_label)
    tick_labels = _session_labels_with_counts(sessions, sample_sizes)
    stacked = []
    for label in labels:
        session_values = [
            _finite_series(values_by_session.get(session, {}).get(label, []))
            for session in sessions
        ]
        stacked.append(np.array([float(np.nanmean(vals)) if vals.size else np.nan for vals in session_values], dtype=float))
    ax.stackplot(x, *stacked, labels=labels, colors=[STATE_COLORS[STATE_LABELS.index(label)] for label in labels], alpha=0.85)
    style_axes(ax, title=title, xlabel="Session", ylabel="Fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))


def _plot_single_state_fraction_by_session(
    ax,
    result: StatisticsResult,
    sessions: list[str],
    *,
    title: str,
    state_label: str,
    include_not_visible: bool = True,
) -> None:
    _plot_state_fraction_by_session(ax, result, sessions, title=title, labels=[state_label], include_not_visible=include_not_visible)


def _plot_lag_by_session(ax, result: StatisticsResult, sessions: list[str], *, title: str) -> None:
    x = np.arange(len(sessions), dtype=float)
    sample_sizes = _session_sample_sizes(result.lag_by_session_values, sessions)
    tick_labels = _session_labels_with_counts(sessions, sample_sizes)
    for i, label in enumerate(STATE_LABELS):
        lag_vals = [result.lag_by_session.get(session, {}).get(label, np.nan) for session in sessions]
        ax.plot(x, lag_vals, marker="o", color=STATE_COLORS[i], label=label)
    style_axes(ax, title=title, xlabel="Session", ylabel="Lag (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))


def _state_zscore_boxplot_data(result: StatisticsResult) -> tuple[list[np.ndarray], list[str]]:
    states = list(STATE_LABELS)
    data = [_finite_series(result.pupil_zscore_mean_by_state_values.get(state, [])) for state in states]
    return data, states


def _build_progress_state_series(
    store: HabituationStore,
    sessions: list,
    *,
    mean: float,
    std: float,
    threshold_values: list[float],
    locomotion_threshold: float,
    missing_buffer_sec: float,
    bin_count: int,
    window_start_sec: float | None = None,
    window_end_sec: float | None = None,
) -> dict:
    n_states = len(STATE_LABELS)
    state_probability_sum = np.zeros((n_states, bin_count), dtype=float)
    state_probability_sq_sum = np.zeros((n_states, bin_count), dtype=float)
    state_probability_count = np.zeros(bin_count, dtype=float)
    sample_size = 0

    for summary in sessions:
        bundle = _trim_bundle_for_cutoff(store, store.load_session_bundle(summary.exp_id))
        duration = float(store.effective_session_duration_sec(summary.exp_id))
        if window_start_sec is None or window_end_sec is None:
            if duration <= 0.0:
                continue
            time_values = np.asarray(bundle.t, dtype=float)
            bin_values = np.clip((time_values / max(duration, 1e-6)) * 100.0, 0.0, 100.0)
            bin_indices = np.clip(bin_values.astype(int), 0, bin_count - 1)
        else:
            if duration < window_end_sec:
                continue
            time_values = np.asarray(bundle.t, dtype=float)
            mask = np.isfinite(time_values) & (time_values >= window_start_sec) & (time_values < window_end_sec)
            if not np.any(mask):
                continue
            window_minutes = (time_values[mask] - window_start_sec) / 60.0
            bin_indices = np.clip(np.floor(window_minutes).astype(int), 0, bin_count - 1)
            time_values = time_values[mask]
        manual_masks = store.load_manual_masks(summary.exp_id)
        visible = _visible_mask(bundle, manual_masks)
        not_visible_mask = _interval_mask(np.asarray(bundle.t, dtype=float), manual_masks)
        calibration = store.session_extra_large_calibration(summary.exp_id)
        extra_large_mask = build_extra_large_mask(
            store,
            bundle,
            calibration,
            manual_masks,
            manual_buffer_sec=missing_buffer_sec,
        )
        z = (bundle.radius.astype(float) - mean) / std
        z[~np.isfinite(z)] = np.nan
        state = classify_zscores(z, threshold_values, visible, extra_large_mask, not_visible_mask)
        if window_start_sec is not None and window_end_sec is not None:
            state = state[np.isfinite(np.asarray(bundle.t, dtype=float)) & (np.asarray(bundle.t, dtype=float) >= window_start_sec) & (np.asarray(bundle.t, dtype=float) < window_end_sec)]
        sample_size += 1
        for b in range(bin_count):
            mask = bin_indices == b
            valid = mask & (state >= 0)
            if not np.any(valid):
                continue
            state_probability_count[b] += 1.0
            for state_id in range(n_states):
                fraction = float(np.mean(state[valid] == state_id))
                if np.isfinite(fraction):
                    state_probability_sum[state_id, b] += fraction
                    state_probability_sq_sum[state_id, b] += fraction * fraction

    mean_values = np.divide(
        state_probability_sum,
        state_probability_count,
        out=np.full((n_states, bin_count), np.nan, dtype=float),
        where=state_probability_count > 0,
    )
    state_probability_var = np.divide(
        state_probability_sq_sum,
        state_probability_count,
        out=np.full((n_states, bin_count), np.nan, dtype=float),
        where=state_probability_count > 0,
    ) - np.square(mean_values)
    state_probability_std = np.sqrt(np.clip(state_probability_var, 0.0, None))

    if window_start_sec is None or window_end_sec is None:
        bins = np.linspace(0.0, 100.0, bin_count, endpoint=False)
    else:
        bins = np.arange(bin_count, dtype=float) + 0.5

    return {
        "kind": "absolute_window" if window_start_sec is not None and window_end_sec is not None else "relative",
        "bins": bins.tolist(),
        "state_probability": mean_values.tolist(),
        "state_probability_std": state_probability_std.tolist(),
        "state_probability_count": state_probability_count.tolist(),
        "sample_size": int(sample_size),
        "window_start_sec": float(window_start_sec) if window_start_sec is not None else None,
        "window_end_sec": float(window_end_sec) if window_end_sec is not None else None,
        "sample_size_unit": "sessions",
    }


def _combine_progress_series(series_list: list[dict]) -> dict:
    if not series_list:
        return {
            "kind": "relative",
            "bins": [],
            "state_probability": [],
            "state_probability_std": [],
            "state_probability_count": [],
            "sample_size": 0,
            "window_start_sec": None,
            "window_end_sec": None,
            "sample_size_unit": "sessions",
        }
    bins = np.asarray(series_list[0].get("bins", []), dtype=float)
    state_probability = np.asarray([np.asarray(series.get("state_probability", []), dtype=float) for series in series_list], dtype=float)
    state_probability_std = np.asarray([np.asarray(series.get("state_probability_std", []), dtype=float) for series in series_list], dtype=float)
    valid = np.isfinite(state_probability)
    count = np.sum(valid, axis=0)
    sum_values = np.nansum(state_probability, axis=0)
    mean = np.divide(sum_values, count, out=np.full_like(sum_values, np.nan, dtype=float), where=count > 0)
    sq_sum = np.nansum(np.square(state_probability), axis=0)
    var = np.divide(sq_sum, count, out=np.full_like(sum_values, np.nan, dtype=float), where=count > 0) - np.square(mean)
    std = np.sqrt(np.clip(var, 0.0, None))
    sample_size = len(series_list)
    return {
        "kind": str(series_list[0].get("kind", "relative")),
        "bins": bins.tolist(),
        "state_probability": mean.tolist(),
        "state_probability_std": std.tolist(),
        "state_probability_count": [sample_size] * int(bins.size),
        "sample_size": int(sample_size),
        "window_start_sec": series_list[0].get("window_start_sec"),
        "window_end_sec": series_list[0].get("window_end_sec"),
        "sample_size_unit": str(series_list[0].get("sample_size_unit", "sessions")),
    }


def _build_animal_weighted_progress_series(
    store: HabituationStore,
    sessions: list,
    *,
    mean: float,
    std: float,
    threshold_values: list[float],
    locomotion_threshold: float,
    missing_buffer_sec: float,
) -> dict:
    sessions_by_animal: dict[str, list] = {}
    for summary in sessions:
        sessions_by_animal.setdefault(summary.animal_id, []).append(summary)

    animal_series: list[dict] = []
    for animal_id, animal_sessions in sessions_by_animal.items():
        ordered = sorted(animal_sessions, key=lambda s: s.sort_key)
        selected = ordered[:2]
        if len(selected) < 2:
            continue
        per_session_series = [
            _build_progress_state_series(
                store,
                [summary],
                mean=mean,
                std=std,
                threshold_values=threshold_values,
                locomotion_threshold=locomotion_threshold,
                missing_buffer_sec=missing_buffer_sec,
                bin_count=100,
            )
            for summary in selected
        ]
        animal_series.append(_combine_progress_series(per_session_series))

    combined = _combine_progress_series(animal_series)
    combined["sample_size"] = len(animal_series)
    combined["sample_size_unit"] = "animals"
    return combined


def _build_animal_weighted_progress_subset_series(
    store: HabituationStore,
    sessions: list,
    *,
    mean: float,
    std: float,
    threshold_values: list[float],
    locomotion_threshold: float,
    missing_buffer_sec: float,
    take_last: bool,
) -> dict:
    sessions_by_animal: dict[str, list] = {}
    for summary in sessions:
        sessions_by_animal.setdefault(summary.animal_id, []).append(summary)

    animal_series: list[dict] = []
    for animal_id, animal_sessions in sessions_by_animal.items():
        ordered = sorted(animal_sessions, key=lambda s: s.sort_key)
        selected = ordered[-2:] if take_last else ordered[:2]
        if len(selected) < 2:
            continue
        animal_series.append(
            _build_progress_state_series(
                store,
                selected,
                mean=mean,
                std=std,
                threshold_values=threshold_values,
                locomotion_threshold=locomotion_threshold,
                missing_buffer_sec=missing_buffer_sec,
                bin_count=100,
            )
        )

    combined = _combine_progress_series(animal_series)
    combined["sample_size"] = len(animal_series)
    combined["sample_size_unit"] = "animals"
    return combined


def _progress_series_payload(
    store: HabituationStore,
    result: StatisticsResult,
    *,
    selected_session_ids: list[str],
    first_last_label: str | None = None,
) -> dict:
    sessions_by_id = {summary.exp_id: summary for summary in store.dataset_sessions()}
    selected_sessions = [sessions_by_id[exp_id] for exp_id in selected_session_ids if exp_id in sessions_by_id]
    payload = {
        "selected_session_ids": selected_session_ids,
        "overall": _build_progress_state_series(
            store,
            selected_sessions,
            mean=result.zscore_mean,
            std=result.zscore_std,
            threshold_values=list(result.thresholds.get("threshold_values", [])),
            locomotion_threshold=float(result.thresholds.get("locomotion_threshold", 0.35)),
            missing_buffer_sec=float(result.thresholds.get("missing_buffer_sec", 1.0)),
            bin_count=100,
        ),
    }
    return payload


def _animal_progress_series_payload(values_by_animal: dict[str, list[float]]) -> dict:
    animals = sorted(values_by_animal)
    if not animals:
        return {
            "animals": [],
            "bins": [],
            "values_by_animal": {},
            "mean_values": [],
            "count_values": [],
            "sample_size": 0,
            "sample_size_unit": "animals",
        }
    max_len = max((len(values_by_animal.get(animal, [])) for animal in animals), default=0)
    if max_len <= 0:
        return {
            "animals": animals,
            "bins": [],
            "values_by_animal": {animal: [float(v) for v in list(values_by_animal.get(animal, []))] for animal in animals},
            "mean_values": [],
            "count_values": [],
            "sample_size": len(animals),
            "sample_size_unit": "animals",
        }
    matrix = np.full((len(animals), max_len), np.nan, dtype=float)
    values_map: dict[str, list[float]] = {}
    for row, animal in enumerate(animals):
        arr = np.asarray(values_by_animal.get(animal, []), dtype=float).reshape(-1)
        values_map[animal] = [float(v) for v in arr.tolist()]
        if arr.size:
            matrix[row, : arr.size] = arr
    count_values = np.sum(np.isfinite(matrix), axis=0)
    sum_values = np.nansum(matrix, axis=0)
    mean_values = np.divide(sum_values, count_values, out=np.full(max_len, np.nan, dtype=float), where=count_values > 0)
    return {
        "animals": animals,
        "bins": np.arange(1, max_len + 1, dtype=float).tolist(),
        "values_by_animal": values_map,
        "mean_values": mean_values.tolist(),
        "count_values": count_values.tolist(),
        "sample_size": len(animals),
        "sample_size_unit": "animals",
    }


def _format_progress_title(title: str, sample_size: int, series: dict) -> str:
    unit = str(series.get("sample_size_unit", "sessions"))
    extra = f" (n={sample_size} {unit})" if sample_size else ""
    if series.get("kind") == "absolute_window" and series.get("window_start_sec") is not None and series.get("window_end_sec") is not None:
        start_min = int(float(series["window_start_sec"]) // 60)
        end_min = int(float(series["window_end_sec"]) // 60)
        return f"{title} - {start_min}-{end_min} min{extra}"
    return f"{title}{extra}"


def _plot_progress_series(ax, series: dict, *, title: str, xlabel: str) -> None:
    bins = np.asarray(series.get("bins", []), dtype=float)
    state_probability = np.asarray(series.get("state_probability", []), dtype=float)
    state_probability_std = np.asarray(series.get("state_probability_std", []), dtype=float)
    for i, label in enumerate(STATE_LABELS):
        mean = state_probability[i] if i < state_probability.shape[0] else np.asarray([])
        std = state_probability_std[i] if i < state_probability_std.shape[0] else np.asarray([])
        lower = np.clip(mean - std, 0.0, 1.0)
        upper = np.clip(mean + std, 0.0, 1.0)
        ax.fill_between(bins, lower, upper, color=STATE_COLORS[i], alpha=0.18)
        ax.plot(bins, mean, label=label, color=STATE_COLORS[i])
    style_axes(ax, title=title, xlabel=xlabel, ylabel="Fraction")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    counts = np.asarray(series.get("state_probability_count", []), dtype=float)
    if counts.size:
        ax.text(
            0.99,
            0.02,
            f"bin n: {int(counts[0])} -> {int(counts[-1])}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="0.25",
        )


def _plot_progress_stacked_area(ax, series: dict, *, title: str, xlabel: str) -> None:
    bins = np.asarray(series.get("bins", []), dtype=float)
    state_probability = np.asarray(series.get("state_probability", []), dtype=float)
    visible_labels = _state_labels_for_combined_plot()
    plot_values = []
    for label in visible_labels:
        idx = STATE_LABELS.index(label)
        values = state_probability[idx] if idx < state_probability.shape[0] else np.asarray([])
        plot_values.append(values)
    if plot_values:
        stacked = np.vstack(plot_values)
        totals = np.nansum(stacked, axis=0)
        normalized = [np.divide(values, totals, out=np.full_like(values, np.nan, dtype=float), where=totals > 0) for values in plot_values]
        ax.stackplot(bins, *normalized, colors=[STATE_COLORS[STATE_LABELS.index(label)] for label in visible_labels], labels=visible_labels, alpha=0.85)
    style_axes(ax, title=title, xlabel=xlabel, ylabel="Fraction")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    counts = np.asarray(series.get("state_probability_count", []), dtype=float)
    if counts.size:
        ax.text(
            0.99,
            0.02,
            f"bin n: {int(counts[0])} -> {int(counts[-1])}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="0.25",
        )


def _plot_animal_progress_lines(ax, values_by_animal: dict[str, list[float]], *, title: str, ylabel: str, average_label: str) -> None:
    series = _animal_progress_series_payload(values_by_animal)
    bins = np.asarray(series.get("bins", []), dtype=float)
    animals = list(series.get("animals", []))
    values_map = dict(series.get("values_by_animal", {}))
    for animal_id in animals:
        values = np.asarray(values_map.get(animal_id, []), dtype=float)
        if not values.size:
            continue
        x = np.arange(1, values.size + 1, dtype=float)
        ax.plot(x, values, marker="o", alpha=0.75, label=animal_id)
    mean_values = np.asarray(series.get("mean_values", []), dtype=float)
    if mean_values.size:
        ax.plot(bins, mean_values, color="black", linewidth=2.5, marker="o", label=f"{average_label} (n={int(series.get('sample_size', 0))} animals)")
    counts = np.asarray(series.get("count_values", []), dtype=float)
    tick_labels = [f"{int(bin_idx)}\n(n={int(count)})" for bin_idx, count in zip(bins, counts)] if bins.size else []
    style_axes(ax, title=_format_progress_title(title, int(series.get("sample_size", 0)), series), xlabel="Session", ylabel=ylabel)
    ax.set_xticks(bins)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))


def _progress_window_specs(result: StatisticsResult) -> list[tuple[str, str, dict]]:
    specs: list[tuple[str, str, dict]] = []
    for key, series in sorted(
        ((key, value) for key, value in result.progress_series.items() if key.startswith("window_")),
        key=lambda item: float(item[1].get("window_start_sec", 0.0)),
    ):
        start_min = int(float(series.get("window_start_sec", 0.0)) // 60)
        end_min = int(float(series.get("window_end_sec", 0.0)) // 60)
        title = f"Pupil state fraction vs experiment length - {start_min}-{end_min} min"
        specs.append((f"progress_{key}", title, {"kind": "progress_stacked", "series_key": key}))
    return specs


def _statistics_panel_specs(result: StatisticsResult) -> list[tuple[str, str, dict]]:
    specs: list[tuple[str, str, dict]] = []

    session_specs = [
        ("locomotion_boxplot", "Locomotion by session", {"kind": "session_boxplot", "metric": "locomotion", "subset": None}),
        ("locomotion_boxplot_le6", "Locomotion by session (<= 6)", {"kind": "session_boxplot", "metric": "locomotion", "subset": "le6"}),
        ("face_motion_boxplot", "Face motion by session", {"kind": "session_boxplot", "metric": "face_motion", "subset": None}),
        ("face_motion_boxplot_le6", "Face motion by session (<= 6)", {"kind": "session_boxplot", "metric": "face_motion", "subset": "le6"}),
        ("pupil_state_fraction_by_session_stacked", "Pupil state fraction by session (stacked area)", {"kind": "state_fraction_stacked", "subset": None, "labels": _state_labels_for_combined_plot(), "include_not_visible": False}),
        ("pupil_state_fraction_by_session_stacked_le6", "Pupil state fraction by session (stacked area, <= 6)", {"kind": "state_fraction_stacked", "subset": "le6", "labels": _state_labels_for_combined_plot(), "include_not_visible": False}),
        ("pupil_state_fraction_pie", "Overall pupil state percentages", {"kind": "state_fraction_pie"}),
    ]
    for label in STATE_LABELS:
        session_specs.append((
            f"pupil_state_fraction_by_session_{label}_with_not_visible",
            f"Pupil state fraction by session - {label} (including not visible)",
            {"kind": "single_state", "state_label": label, "subset": None, "include_not_visible": True},
        ))
        session_specs.append((
            f"pupil_state_fraction_by_session_{label}_with_not_visible_le6",
            f"Pupil state fraction by session - {label} (including not visible, <= 6)",
            {"kind": "single_state", "state_label": label, "subset": "le6", "include_not_visible": True},
        ))
        if label != "not_visible":
            session_specs.append((
                f"pupil_state_fraction_by_session_{label}_without_not_visible",
                f"Pupil state fraction by session - {label} (excluding not visible)",
                {"kind": "single_state", "state_label": label, "subset": None, "include_not_visible": False},
            ))
            session_specs.append((
                f"pupil_state_fraction_by_session_{label}_without_not_visible_le6",
                f"Pupil state fraction by session - {label} (excluding not visible, <= 6)",
                {"kind": "single_state", "state_label": label, "subset": "le6", "include_not_visible": False},
            ))
    session_specs.extend(
        [
            ("lag_by_state", "Lag to first pupil state after 1 min by session", {"kind": "lag", "subset": None}),
            ("lag_by_state_le6", "Lag to first pupil state after 1 min by session (<= 6)", {"kind": "lag", "subset": "le6"}),
        ]
    )
    specs.extend(session_specs)

    specs.append(("locomotion_progress", "Locomotion progression across sessions", {"kind": "animal_progress", "metric": "locomotion"}))
    specs.append(("pupil_size_progress", "Mean z-scored pupil size progression across sessions", {"kind": "animal_progress", "metric": "pupil_zscore"}))
    specs.append(("state_fraction_stacked", "Pupil state fraction vs experiment length (stacked area)", {"kind": "progress_stacked", "series_key": "overall"}))
    specs.extend(_progress_window_specs(result))
    if "first_2" in result.progress_series:
        specs.append(("state_fraction_first_2_sessions_stacked", "Pupil state fraction vs experiment length - first 2 sessions (stacked area)", {"kind": "progress_stacked", "series_key": "first_2"}))
    if "last_2" in result.progress_series:
        specs.append(("state_fraction_last_2_sessions_stacked", "Pupil state fraction vs experiment length - last 2 sessions (stacked area)", {"kind": "progress_stacked", "series_key": "last_2"}))
    return specs


def statistics_panel_paths(output_dir: Path) -> list[tuple[str, Path]]:
    output_dir = Path(output_dir)
    result_path = output_dir / "statistics.json"
    if result_path.exists():
        try:
            result = StatisticsResult.from_dict(json.loads(result_path.read_text()))
            specs = _statistics_panel_specs(result)
            return [
                (title, output_dir / f"statistics_panel_{idx:02d}_{slug}.png")
                for idx, (slug, title, _config) in enumerate(specs, start=1)
            ]
        except Exception:
            pass
    return []


def _statistics_panel_export_complete(output_dir: Path, result: StatisticsResult) -> bool:
    output_dir = Path(output_dir)
    summary_path = output_dir / "statistics_summary.png"
    if not summary_path.exists():
        return False
    for idx, (slug, _title, _config) in enumerate(_statistics_panel_specs(result), start=1):
        if not (output_dir / f"statistics_panel_{idx:02d}_{slug}.png").exists():
            return False
    return True


def _session_subset_for_config(result: StatisticsResult, subset: str | None) -> list[str]:
    if subset and subset.startswith("le") and subset[2:].isdigit():
        return _session_subset(result.session_labels, limit=int(subset[2:]))
    return list(result.session_labels)


def _build_panel_figure(result: StatisticsResult, config: dict, title: str) -> Figure:
    kind = config["kind"]
    if kind == "session_boxplot":
        sessions = _session_subset_for_config(result, config.get("subset"))
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.subplots()
        metric = config["metric"]
        if metric == "locomotion":
            _plot_session_metric_boxplot(ax, result, sessions, result.locomotion_pct_by_session_values, title=title, ylabel="Fraction", color="tab:blue")
        else:
            _plot_session_metric_boxplot(ax, result, sessions, result.face_motion_pct_by_session_values, title=title, ylabel="Fraction", color="tab:orange")
        return fig
    if kind == "state_fraction":
        sessions = _session_subset_for_config(result, config.get("subset"))
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.subplots()
        _plot_state_fraction_by_session(
            ax,
            result,
            sessions,
            title=title,
            labels=config.get("labels", _state_labels_for_combined_plot()),
            include_not_visible=bool(config.get("include_not_visible", True)),
        )
        return fig
    if kind == "state_fraction_stacked":
        sessions = _session_subset_for_config(result, config.get("subset"))
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.subplots()
        _plot_state_fraction_stacked_by_session(
            ax,
            result,
            sessions,
            title=title,
            labels=config.get("labels", _state_labels_for_combined_plot()),
            include_not_visible=bool(config.get("include_not_visible", False)),
        )
        return fig
    if kind == "state_fraction_pie":
        fractions = result.pupil_state_fraction_overall or _build_overall_pupil_state_fractions(result.pupil_pct_by_session, result.pupil_pct_by_session_visible)
        fig = Figure(figsize=(13, 6), constrained_layout=True)
        ax = fig.subplots()
        _plot_state_fraction_pies(ax, title=title, fractions=fractions, sample_size=len(result.pupil_pct_by_session), sample_size_unit="sessions")
        return fig
    if kind == "single_state":
        sessions = _session_subset_for_config(result, config.get("subset"))
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.subplots()
        _plot_single_state_fraction_by_session(
            ax,
            result,
            sessions,
            title=title,
            state_label=config["state_label"],
            include_not_visible=bool(config.get("include_not_visible", True)),
        )
        return fig
    if kind == "lag":
        sessions = _session_subset_for_config(result, config.get("subset"))
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.subplots()
        _plot_lag_by_session(ax, result, sessions, title=title)
        return fig
    if kind == "animal_progress":
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.subplots()
        if config["metric"] == "locomotion":
            _plot_animal_progress_lines(ax, result.locomotion_progress_by_animal_values, title=title, ylabel="Fraction", average_label="Average")
        else:
            _plot_animal_progress_lines(ax, result.pupil_zscore_progress_by_animal_values, title=title, ylabel="Mean z-score", average_label="Average")
        return fig
    if kind == "progress":
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.subplots()
        series = result.progress_series.get(config["series_key"], {})
        title = _format_progress_title(title, int(series.get("sample_size", 0)), series)
        xlabel = "Progress (%)" if series.get("kind") != "absolute_window" else "Minutes into 30-minute window"
        _plot_progress_series(ax, series, title=title, xlabel=xlabel)
        return fig
    if kind == "progress_stacked":
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.subplots()
        series = result.progress_series.get(config["series_key"], {})
        title = _format_progress_title(title, int(series.get("sample_size", 0)), series)
        xlabel = "Progress (%)" if series.get("kind") != "absolute_window" else "Minutes into 30-minute window"
        _plot_progress_stacked_area(ax, series, title=title, xlabel=xlabel)
        return fig
    raise ValueError(f"Unknown panel kind: {kind}")


def _build_statistics_panel_figures(result: StatisticsResult) -> list[Figure]:
    return [_build_panel_figure(result, config, title) for _, title, config in _statistics_panel_specs(result)]


def _summary_panel_specs(result: StatisticsResult) -> list[tuple[str, str, dict]]:
    return [
        ("locomotion_boxplot", "Locomotion by session", {"kind": "session_boxplot", "metric": "locomotion"}),
        ("locomotion_boxplot_le6", "Locomotion by session (<= 6)", {"kind": "session_boxplot", "metric": "locomotion", "subset": "le6"}),
        ("face_motion_boxplot", "Face motion by session", {"kind": "session_boxplot", "metric": "face_motion"}),
        ("face_motion_boxplot_le6", "Face motion by session (<= 6)", {"kind": "session_boxplot", "metric": "face_motion", "subset": "le6"}),
        ("pupil_zscore_by_state", "Mean z-scored pupil size by state", {"kind": "state_zscore_boxplot"}),
        ("pupil_state_fraction_by_session_stacked", "Pupil state fraction by session (stacked area)", {"kind": "state_fraction_stacked", "include_not_visible": False}),
        ("pupil_state_fraction_by_session_stacked_le6", "Pupil state fraction by session (stacked area, <= 6)", {"kind": "state_fraction_stacked", "subset": "le6", "include_not_visible": False}),
        ("pupil_state_fraction_pie", "Overall pupil state percentages", {"kind": "state_fraction_pie"}),
        ("lag_by_state", "Lag to first pupil state after 1 min by session", {"kind": "lag"}),
        ("lag_by_state_le6", "Lag to first pupil state after 1 min by session (<= 6)", {"kind": "lag", "subset": "le6"}),
        ("locomotion_progress", "Locomotion progression across sessions", {"kind": "animal_progress", "metric": "locomotion"}),
        ("pupil_size_progress", "Mean z-scored pupil size progression across sessions", {"kind": "animal_progress", "metric": "pupil_zscore"}),
        ("state_fraction_stacked", "Pupil state fraction vs experiment length (stacked area)", {"kind": "progress_stacked", "series_key": "overall"}),
    ]


def statistics_summary_panel_specs(result: StatisticsResult) -> list[tuple[str, str, dict]]:
    return _summary_panel_specs(result)


def _selected_summary_panel_specs(result: StatisticsResult, panel_keys: list[str] | None) -> list[tuple[str, str, dict]]:
    specs = _summary_panel_specs(result)
    if panel_keys is None:
        return specs
    spec_map = {slug: (slug, title, config) for slug, title, config in specs}
    selected: list[tuple[str, str, dict]] = []
    for key in panel_keys:
        if key in spec_map:
            selected.append(spec_map[key])
    return selected


def _draw_summary_panel(ax, result: StatisticsResult, config: dict, title: str) -> None:
    sessions = _session_subset_for_config(result, config.get("subset"))
    kind = config["kind"]
    if kind == "session_boxplot":
        metric = config.get("metric", "locomotion")
        if metric == "locomotion":
            _plot_session_metric_boxplot(ax, result, sessions, result.locomotion_pct_by_session_values, title=title, ylabel="Fraction", color="tab:blue")
        else:
            _plot_session_metric_boxplot(ax, result, sessions, result.face_motion_pct_by_session_values, title=title, ylabel="Fraction", color="tab:orange")
        return
    if kind == "state_zscore_boxplot":
        state_data, states = _state_zscore_boxplot_data(result)
        _boxplot(ax, state_data, states, colors=STATE_COLORS, title=title, xlabel="Pupil state", ylabel="Mean z-score")
        return
    if kind == "state_fraction":
        _plot_state_fraction_by_session(ax, result, sessions, title=title, labels=_state_labels_for_combined_plot(), include_not_visible=bool(config.get("include_not_visible", True)))
        return
    if kind == "state_fraction_stacked":
        _plot_state_fraction_stacked_by_session(ax, result, sessions, title=title, labels=_state_labels_for_combined_plot(), include_not_visible=bool(config.get("include_not_visible", False)))
        return
    if kind == "state_fraction_pie":
        fractions = result.pupil_state_fraction_overall or _build_overall_pupil_state_fractions(result.pupil_pct_by_session, result.pupil_pct_by_session_visible)
        _plot_state_fraction_pies(ax, title=title, fractions=fractions, sample_size=len(result.pupil_pct_by_session), sample_size_unit="sessions")
        return
    if kind == "lag":
        _plot_lag_by_session(ax, result, sessions, title=title)
        return
    if kind == "animal_progress":
        if config["metric"] == "locomotion":
            _plot_animal_progress_lines(ax, result.locomotion_progress_by_animal_values, title=title, ylabel="Fraction", average_label="Average")
        else:
            _plot_animal_progress_lines(ax, result.pupil_zscore_progress_by_animal_values, title=title, ylabel="Mean z-score", average_label="Average")
        return
    if kind == "progress":
        series = result.progress_series.get(config["series_key"], {})
        title = _format_progress_title(title, int(series.get("sample_size", 0)), series)
        _plot_progress_series(ax, series, title=title, xlabel="Progress (%)")
        return
    if kind == "progress_stacked":
        series = result.progress_series.get(config["series_key"], {})
        title = _format_progress_title(title, int(series.get("sample_size", 0)), series)
        _plot_progress_stacked_area(ax, series, title=title, xlabel="Progress (%)")
        return
    raise ValueError(f"Unknown summary panel kind: {kind}")


def _build_summary_figure(result: StatisticsResult, panel_keys: list[str] | None = None) -> Figure:
    specs = _selected_summary_panel_specs(result, panel_keys)
    n_panels = max(1, len(specs))
    if n_panels <= 2:
        cols = 1
    elif n_panels <= 6:
        cols = 2
    else:
        cols = 3
    rows = int(np.ceil(n_panels / cols))
    fig = Figure(figsize=(8.5 * cols, 5.8 * rows), constrained_layout=True)
    axes = np.asarray(fig.subplots(rows, cols, squeeze=False)).ravel()
    for ax, (_, title, config) in zip(axes, specs):
        _draw_summary_panel(ax, result, config, title)
    for ax in axes[len(specs):]:
        ax.axis("off")
    fig.suptitle(f"Habituation statistics - {result.scope} / {result.animal_id}", y=1.02)
    return fig


def save_statistics_summary_figure(output_dir: Path, result: StatisticsResult, *, summary_panel_keys: list[str] | None = None) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    fig = _build_summary_figure(result, panel_keys=summary_panel_keys)
    return save_figure(fig, "statistics_summary", output_dir)


def save_statistics_outputs(store: HabituationStore, result: StatisticsResult, *, summary_panel_keys: list[str] | None = None) -> tuple[Path, Path, Path]:
    set_poster_style()
    scope_dir = store.source_root / "gui_output" / "stats"
    scope_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.generated_at.replace(":", "").replace("-", "").replace("T", "_")
    output_dir = scope_dir / f"{result.scope}_{result.animal_id}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "statistics.json"
    payload = result.to_dict()
    payload["cache_signature"] = store.statistics_cache_signature(result.scope, result.animal_id, result.thresholds)
    payload["plot_version"] = STATISTICS_PLOT_VERSION
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    settings = store.get_animal_settings(result.animal_id)
    settings["last_stats_signature"] = payload["cache_signature"]
    settings["last_stats_output_dir"] = str(output_dir)
    store.set_animal_settings(result.animal_id, settings)

    svg_path, png_path = save_statistics_summary_figure(output_dir, result, summary_panel_keys=summary_panel_keys)

    specs = _statistics_panel_specs(result)
    for idx, (slug, title, config) in enumerate(specs, start=1):
        fig_panel = _build_panel_figure(result, config, title)
        save_figure(fig_panel, f"statistics_panel_{idx:02d}_{slug}", output_dir)

    return result_path, svg_path, png_path


def load_cached_statistics_outputs(
    store: HabituationStore,
    *,
    scope: str,
    animal_id: str,
    thresholds: dict,
) -> tuple[StatisticsResult, tuple[Path, Path, Path]] | None:
    expected_signature = store.statistics_cache_signature(scope, animal_id, thresholds)
    stats_root = store.source_root / "gui_output" / "stats"
    if not stats_root.exists():
        return None

    direct_dir_str = str(store.get_animal_settings(animal_id).get("last_stats_output_dir", "")).strip()
    direct_result = Path(direct_dir_str) / "statistics.json" if direct_dir_str else None
    if direct_result is not None and direct_result.exists():
        try:
            data = json.loads(direct_result.read_text())
        except Exception:
            data = None
        if data and data.get("cache_signature") == expected_signature and int(data.get("plot_version", 1)) == STATISTICS_PLOT_VERSION:
            result = StatisticsResult.from_dict(data)
            output_dir = direct_result.parent
            svg_path = output_dir / "statistics_summary.svg"
            png_path = output_dir / "statistics_summary.png"
            if png_path.exists() and _statistics_panel_export_complete(output_dir, result):
                return result, (direct_result, svg_path, png_path)

    candidates = sorted(
        stats_root.glob(f"{scope}_{animal_id}_*/statistics.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for result_path in candidates:
        try:
            data = json.loads(result_path.read_text())
        except Exception:
            continue
        if data.get("cache_signature") != expected_signature:
            continue
        if int(data.get("plot_version", 1)) != STATISTICS_PLOT_VERSION:
            continue
        result = StatisticsResult.from_dict(data)
        output_dir = result_path.parent
        svg_path = output_dir / "statistics_summary.svg"
        png_path = output_dir / "statistics_summary.png"
        if not png_path.exists() or not _statistics_panel_export_complete(output_dir, result):
            continue
        return result, (result_path, svg_path, png_path)
    return None
