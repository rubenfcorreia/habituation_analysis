
from __future__ import annotations

import json
import cv2
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .data import HabituationStore, SessionBundle, analysis_cutoff_mask, apply_time_mask
from .plotting import save_figure, set_poster_style, style_axes


STATE_LABELS = ["small", "medium", "large", "extra_large", "not_visible"]
STATE_COLORS = ["tab:green", "tab:purple", "tab:red", "tab:brown", "tab:gray"]
MIN_EXTRA_LARGE_MISSING_SEC = 1.0
MANUAL_INTERVAL_BUFFER_SEC = 1.0
CALIBRATION_BRIGHTNESS_MARGIN = 0.10


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
    lag_by_session: dict[str, dict[str, float]]
    progress_bins: np.ndarray
    state_probability: np.ndarray
    state_probability_std: np.ndarray
    thresholds: dict
    zscore_mean: float
    zscore_std: float

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "animal_id": self.animal_id,
            "generated_at": self.generated_at,
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
            "lag_by_session": self.lag_by_session,
            "progress_bins": self.progress_bins.tolist(),
            "state_probability": self.state_probability.tolist(),
            "state_probability_std": self.state_probability_std.tolist(),
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
            pupil_pct_by_session={
                str(session): {str(label): float(val) for label, val in _state_map(dict(label_dict)).items()}
                for session, label_dict in dict(data.get("pupil_pct_by_session", data.get("pupil_pct_by_day", {}))).items()
            },
            pupil_pct_by_session_values={
                str(session): {
                    str(label): [float(v) for v in list(vals)]
                    for label, vals in dict(label_dict).items()
                }
                for session, label_dict in dict(data.get("pupil_pct_by_session_values", data.get("pupil_pct_by_day_values", {}))).items()
            },
            lag_by_session={
                str(exp_id): {str(label): float(val) for label, val in _state_map(dict(label_dict)).items()}
                for exp_id, label_dict in dict(data.get("lag_by_session", {})).items()
            },
            progress_bins=np.asarray(data.get("progress_bins", []), dtype=float),
            state_probability=_pad_state_array(data.get("state_probability", [])),
            state_probability_std=_pad_state_array(data.get("state_probability_std", [])),
            thresholds=dict(data.get("thresholds", {})),
            zscore_mean=float(data.get("zscore_mean", 0.0)),
            zscore_std=float(data.get("zscore_std", 1.0)),
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
    return [
        s
        for s in sessions
        if s.has_right_pickle and not store.is_deeplabcut_reference_session(s.exp_id) and not store.is_session_do_not_use(s.exp_id)
    ]


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
    pupil_values_by_session: dict[str, dict[str, list[float]]] = {
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
        for state_id, state_label in enumerate(STATE_LABELS):
            mask = valid_radius & (state == state_id)
            if np.any(mask):
                pupil_zscore_mean_by_state_values[state_label].append(float(np.nanmean(z[mask])))
        finite_visible = state >= 0
        denom = float(np.sum(finite_visible)) if np.sum(finite_visible) else 1.0
        for state_id, state_label in enumerate(STATE_LABELS):
            pct = float(np.sum(state == state_id) / denom)
            pupil_values_by_session.setdefault(session_label, {label: [] for label in STATE_LABELS})[state_label].append(pct)

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
    lag_by_session = {
        label: {state_label: _nanmean_or_nan(values[state_label]) for state_label in STATE_LABELS}
        for label, values in lag_values_by_session.items()
    }

    progress_bins = np.linspace(0, 100, 100, endpoint=False)
    n_states = len(STATE_LABELS)
    state_probability_sum = np.zeros((n_states, 100), dtype=float)
    state_probability_sq_sum = np.zeros((n_states, 100), dtype=float)
    state_probability_count = np.zeros((n_states, 100), dtype=float)
    for summary in eligible or sessions:
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
        lag_by_session=lag_by_session,
        progress_bins=progress_bins,
        state_probability=state_probability,
        state_probability_std=state_probability_std,
        thresholds={
            "percentiles": percentiles,
            "threshold_values": threshold_values,
            "locomotion_threshold": locomotion_threshold,
            "missing_buffer_sec": missing_buffer_sec,
        },
        zscore_mean=float(mean),
        zscore_std=float(std),
    )


STATISTICS_PANEL_SPECS = [
    ("locomotion_boxplot", "Locomotion by session"),
    ("face_motion_boxplot", "Face motion by session"),
    ("pupil_zscore_by_state", "Mean z-scored pupil size by state"),
    ("pupil_state_fraction_by_session", "Pupil state fraction by session"),
    ("lag_by_state", "Lag to first pupil state after 1 min by session"),
    ("state_fraction", "Pupil state fraction vs experiment length"),
]


def statistics_panel_paths(output_dir: Path) -> list[tuple[str, Path]]:
    output_dir = Path(output_dir)
    return [
        (title, output_dir / f"statistics_panel_{idx:02d}_{slug}.png")
        for idx, (slug, title) in enumerate(STATISTICS_PANEL_SPECS, start=1)
    ]


def _finite_series(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _boxplot(ax, data: list[np.ndarray], labels: list[str], *, colors: list[str], title: str, xlabel: str, ylabel: str) -> None:
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


def _plot_state_fraction_by_session(ax, result: StatisticsResult, sessions: list[str], *, title: str) -> None:
    x = np.arange(len(sessions), dtype=float)
    for i, label in enumerate(STATE_LABELS):
        session_values = [
            _finite_series(result.pupil_pct_by_session_values.get(session, {}).get(label, []))
            for session in sessions
        ]
        means = np.array([float(np.nanmean(vals)) if vals.size else np.nan for vals in session_values], dtype=float)
        sds = np.array([float(np.nanstd(vals)) if vals.size else np.nan for vals in session_values], dtype=float)
        lower = np.clip(means - sds, 0.0, 1.0)
        upper = np.clip(means + sds, 0.0, 1.0)
        ax.fill_between(x, lower, upper, color=STATE_COLORS[i], alpha=0.18)
        ax.plot(x, means, marker="o", color=STATE_COLORS[i], label=label)
    style_axes(ax, title=title, xlabel="Session", ylabel="Fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=90, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))


def _state_zscore_boxplot_data(result: StatisticsResult) -> tuple[list[np.ndarray], list[str]]:
    states = list(STATE_LABELS)
    data = [_finite_series(result.pupil_zscore_mean_by_state_values.get(state, [])) for state in states]
    return data, states


def _build_statistics_panel_figures(result: StatisticsResult) -> list[Figure]:
    sessions = result.session_labels
    figures: list[Figure] = []

    fig = Figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.subplots()
    locomotion_data = [_finite_series(result.locomotion_pct_by_session_values.get(session, [])) for session in sessions]
    _boxplot(ax, locomotion_data, sessions, colors=["tab:blue"] * max(1, len(sessions)), title="Locomotion by session", xlabel="Session", ylabel="Fraction")
    ax.set_ylim(0.0, 1.0)
    figures.append(fig)

    fig = Figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.subplots()
    face_data = [_finite_series(result.face_motion_pct_by_session_values.get(session, [])) for session in sessions]
    _boxplot(ax, face_data, sessions, colors=["tab:orange"] * max(1, len(sessions)), title="Face motion by session", xlabel="Session", ylabel="Fraction")
    ax.set_ylim(0.0, 1.0)
    figures.append(fig)

    fig = Figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.subplots()
    state_data, states = _state_zscore_boxplot_data(result)
    _boxplot(ax, state_data, states, colors=STATE_COLORS, title="Mean z-scored pupil size by state", xlabel="Pupil state", ylabel="Mean z-score")
    figures.append(fig)

    fig = Figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.subplots()
    _plot_state_fraction_by_session(ax, result, sessions, title="Pupil state fraction by session")
    figures.append(fig)

    fig = Figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.subplots()
    lag_labels = result.session_labels
    lag_x = np.arange(len(lag_labels))
    for i, label in enumerate(STATE_LABELS):
        lag_vals = [result.lag_by_session.get(exp_id, {}).get(label, np.nan) for exp_id in lag_labels]
        ax.plot(lag_x, lag_vals, marker="o", color=STATE_COLORS[i], label=label)
    style_axes(ax, title="Lag to first pupil state after 1 min by session", xlabel="Session", ylabel="Lag (s)")
    ax.set_xticks(lag_x)
    ax.set_xticklabels(lag_labels, rotation=90, fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    figures.append(fig)

    fig = Figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.subplots()
    for i, label in enumerate(STATE_LABELS):
        mean = result.state_probability[i]
        std = result.state_probability_std[i]
        lower = np.clip(mean - std, 0.0, 1.0)
        upper = np.clip(mean + std, 0.0, 1.0)
        ax.fill_between(result.progress_bins, lower, upper, color=STATE_COLORS[i], alpha=0.18)
        ax.plot(result.progress_bins, mean, label=label, color=STATE_COLORS[i])
    style_axes(ax, title="Pupil state fraction vs experiment length", xlabel="Progress (%)", ylabel="Fraction")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    figures.append(fig)

    return figures


def save_statistics_outputs(store: HabituationStore, result: StatisticsResult) -> tuple[Path, Path, Path]:
    set_poster_style()
    scope_dir = store.source_root / "gui_output" / "stats"
    scope_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.generated_at.replace(":", "").replace("-", "").replace("T", "_")
    output_dir = scope_dir / f"{result.scope}_{result.animal_id}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "statistics.json"
    payload = result.to_dict()
    payload["cache_signature"] = store.statistics_cache_signature(result.scope, result.animal_id, result.thresholds)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    settings = store.get_animal_settings(result.animal_id)
    settings["last_stats_signature"] = payload["cache_signature"]
    settings["last_stats_output_dir"] = str(output_dir)
    store.set_animal_settings(result.animal_id, settings)

    fig = Figure(figsize=(16, 14), constrained_layout=True)
    axes = fig.subplots(3, 2).ravel()
    sessions = result.session_labels

    locomotion_data = [_finite_series(result.locomotion_pct_by_session_values.get(session, [])) for session in sessions]
    face_data = [_finite_series(result.face_motion_pct_by_session_values.get(session, [])) for session in sessions]
    state_data, states = _state_zscore_boxplot_data(result)

    ax = axes[0]
    _boxplot(ax, locomotion_data, sessions, colors=["tab:blue"] * max(1, len(sessions)), title="Locomotion by session", xlabel="Session", ylabel="Fraction")
    ax.set_ylim(0.0, 1.0)

    ax = axes[1]
    _boxplot(ax, face_data, sessions, colors=["tab:orange"] * max(1, len(sessions)), title="Face motion by session", xlabel="Session", ylabel="Fraction")
    ax.set_ylim(0.0, 1.0)

    ax = axes[2]
    _boxplot(ax, state_data, states, colors=STATE_COLORS, title="Mean z-scored pupil size by state", xlabel="Pupil state", ylabel="Mean z-score")

    ax = axes[3]
    _plot_state_fraction_by_session(ax, result, sessions, title="Pupil state fraction by session")

    ax = axes[4]
    lag_labels = result.session_labels
    lag_x = np.arange(len(lag_labels))
    for i, label in enumerate(STATE_LABELS):
        lag_vals = [result.lag_by_session.get(exp_id, {}).get(label, np.nan) for exp_id in lag_labels]
        ax.plot(lag_x, lag_vals, marker="o", color=STATE_COLORS[i], label=label)
    style_axes(ax, title="Lag to first pupil state after 1 min by session", xlabel="Session", ylabel="Lag (s)")
    ax.set_xticks(lag_x)
    ax.set_xticklabels(lag_labels, rotation=90, fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    ax = axes[5]
    for i, label in enumerate(STATE_LABELS):
        mean = result.state_probability[i]
        std = result.state_probability_std[i]
        lower = np.clip(mean - std, 0.0, 1.0)
        upper = np.clip(mean + std, 0.0, 1.0)
        ax.fill_between(result.progress_bins, lower, upper, color=STATE_COLORS[i], alpha=0.18)
        ax.plot(result.progress_bins, mean, label=label, color=STATE_COLORS[i])
    style_axes(ax, title="Pupil state fraction vs experiment length", xlabel="Progress (%)", ylabel="Fraction")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"Habituation statistics - {result.scope} / {result.animal_id}", y=1.02)
    svg_path, png_path = save_figure(fig, "statistics_summary", output_dir)

    for idx, fig_panel in enumerate(_build_statistics_panel_figures(result), start=1):
        slug = STATISTICS_PANEL_SPECS[idx - 1][0]
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
        if data and data.get("cache_signature") == expected_signature:
            result = StatisticsResult.from_dict(data)
            output_dir = direct_result.parent
            svg_path = output_dir / "statistics_summary.svg"
            png_path = output_dir / "statistics_summary.png"
            if png_path.exists():
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
        result = StatisticsResult.from_dict(data)
        output_dir = result_path.parent
        svg_path = output_dir / "statistics_summary.svg"
        png_path = output_dir / "statistics_summary.png"
        if not png_path.exists():
            continue
        return result, (result_path, svg_path, png_path)
    return None
