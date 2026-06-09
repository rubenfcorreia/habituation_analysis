
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .data import HabituationStore, SessionBundle
from .plotting import save_figure, set_poster_style, style_axes


STATE_LABELS = ["small", "medium", "large", "extra_large"]
MIN_EXTRA_LARGE_MISSING_SEC = 1.0
MANUAL_INTERVAL_BUFFER_SEC = 1.0


@dataclass
class StatisticsResult:
    scope: str
    animal_id: str
    generated_at: str
    session_ids: list[str]
    eligible_session_ids: list[str]
    day_labels: list[str]
    locomotion_pct_by_day: dict[str, float]
    face_motion_pct_by_day: dict[str, float]
    pupil_pct_by_day: dict[str, dict[str, float]]
    lag_by_session: dict[str, dict[str, float]]
    progress_bins: np.ndarray
    state_probability: np.ndarray
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
            "day_labels": self.day_labels,
            "locomotion_pct_by_day": self.locomotion_pct_by_day,
            "face_motion_pct_by_day": self.face_motion_pct_by_day,
            "pupil_pct_by_day": self.pupil_pct_by_day,
            "lag_by_session": self.lag_by_session,
            "progress_bins": self.progress_bins.tolist(),
            "state_probability": self.state_probability.tolist(),
            "thresholds": self.thresholds,
            "zscore_mean": float(self.zscore_mean),
            "zscore_std": float(self.zscore_std),
        }


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
    missing = ~np.asarray(bundle.visible_mask_base, dtype=bool)
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
        bundle = store.load_session_bundle(summary.exp_id)
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
        bundle = store.load_session_bundle(summary.exp_id)
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
        state[np.asarray(extra_large_mask, dtype=bool) & (state < 0)] = 3
    return state


def robust_face_motion_threshold(motion: np.ndarray) -> float:
    finite = motion[np.isfinite(motion)]
    if finite.size == 0:
        return 0.0
    med = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - med)))
    return med + 3.0 * max(mad, 1e-6)


def _session_day(summary) -> str:
    return summary.date or summary.exp_id.split("_")[0]


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
    sessions = _analysis_sessions(store, scope, animal_id)
    eligible = [s for s in sessions if float(s.video_duration_sec or 0.0) >= 1800.0]
    mean, std = compute_animal_baseline(store, animal_id, scope=scope)

    locomotion_pct_by_day: dict[str, list[float]] = {}
    face_motion_pct_by_day: dict[str, list[float]] = {}
    pupil_pct_by_day: dict[str, dict[str, list[float]]] = {}
    lag_by_session: dict[str, dict[str, float]] = {}
    day_to_sessions: dict[str, list[str]] = {}

    total = max(1, len(sessions))
    for idx, summary in enumerate(sessions):
        if progress_cb:
            progress_cb(idx / total, f"Processing {summary.exp_id}")
        bundle = store.load_session_bundle(summary.exp_id)
        manual_masks = store.load_manual_masks(summary.exp_id)
        visible = _visible_mask(bundle, manual_masks)
        extra_large_mask = _extra_large_missing_mask(bundle, manual_masks, manual_buffer_sec=missing_buffer_sec)
        day = _session_day(summary)
        day_to_sessions.setdefault(day, []).append(summary.exp_id)

        z = (bundle.radius.astype(float) - mean) / std
        z[~np.isfinite(z)] = np.nan
        state = classify_zscores(z, threshold_values, visible, extra_large_mask)

        valid_motion = np.isfinite(bundle.locomotion)
        locomotion_pct = float(np.nanmean((bundle.locomotion[valid_motion] > locomotion_threshold).astype(float))) if np.any(valid_motion) else float("nan")
        locomotion_pct_by_day.setdefault(day, []).append(locomotion_pct)

        face_t, face_motion = store.load_face_motion(summary.exp_id)
        if face_t is not None and face_motion is not None and face_motion.size:
            face_thr = robust_face_motion_threshold(face_motion)
            face_pct = float(np.nanmean((face_motion > face_thr).astype(float)))
        else:
            face_pct = float("nan")
        face_motion_pct_by_day.setdefault(day, []).append(face_pct)

        pupil_pct_by_day.setdefault(day, {label: [] for label in STATE_LABELS})
        finite_visible = state >= 0
        denom = float(np.sum(finite_visible)) if np.sum(finite_visible) else 1.0
        for state_id, label in enumerate(STATE_LABELS):
            pct = float(np.sum(state == state_id) / denom)
            pupil_pct_by_day[day][label].append(pct)

        lag_by_session[summary.exp_id] = {}
        first_minute = bundle.t >= 60.0
        for state_id, label in enumerate(STATE_LABELS):
            idxs = np.where(first_minute & (state == state_id))[0]
            if idxs.size:
                lag_by_session[summary.exp_id][label] = float(bundle.t[idxs[0]] - 60.0)
            else:
                lag_by_session[summary.exp_id][label] = float("nan")

    locomotion_pct_by_day = {day: float(np.nanmean(vals)) if vals else float("nan") for day, vals in locomotion_pct_by_day.items()}
    face_motion_pct_by_day = {day: float(np.nanmean(vals)) if vals else float("nan") for day, vals in face_motion_pct_by_day.items()}
    pupil_pct_by_day = {
        day: {label: float(np.nanmean(vals)) if vals else float("nan") for label, vals in label_dict.items()}
        for day, label_dict in pupil_pct_by_day.items()
    }

    progress_bins = np.linspace(0, 100, 100, endpoint=False)
    state_probability = np.zeros((4, 100), dtype=float)
    combined_counts = np.zeros((4, 100), dtype=float)
    combined_valid = np.zeros(100, dtype=float)
    for summary in eligible or sessions:
        bundle = store.load_session_bundle(summary.exp_id)
        manual_masks = store.load_manual_masks(summary.exp_id)
        visible = _visible_mask(bundle, manual_masks)
        extra_large_mask = _extra_large_missing_mask(bundle, manual_masks, manual_buffer_sec=missing_buffer_sec)
        z = (bundle.radius.astype(float) - mean) / std
        z[~np.isfinite(z)] = np.nan
        state = classify_zscores(z, threshold_values, visible, extra_large_mask)
        progress = np.clip((bundle.t / max(summary.duration_sec, 1e-6)) * 100.0, 0.0, 100.0)
        bins = np.clip(progress.astype(int), 0, 99)
        for b in range(100):
            mask = bins == b
            if not np.any(mask):
                continue
            valid = mask & (state >= 0)
            if not np.any(valid):
                continue
            combined_valid[b] += float(np.sum(valid))
            for state_id in range(4):
                combined_counts[state_id, b] += float(np.sum(state[valid] == state_id))
    for b in range(100):
        if combined_valid[b] > 0:
            state_probability[:, b] = combined_counts[:, b] / combined_valid[b]

    return StatisticsResult(
        scope=scope,
        animal_id=animal_id,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        session_ids=[s.exp_id for s in sessions],
        eligible_session_ids=[s.exp_id for s in eligible],
        day_labels=sorted(set(day_to_sessions.keys())),
        locomotion_pct_by_day=locomotion_pct_by_day,
        face_motion_pct_by_day=face_motion_pct_by_day,
        pupil_pct_by_day=pupil_pct_by_day,
        lag_by_session=lag_by_session,
        progress_bins=progress_bins,
        state_probability=state_probability,
        thresholds={
            "percentiles": percentiles,
            "threshold_values": threshold_values,
            "locomotion_threshold": locomotion_threshold,
            "missing_buffer_sec": missing_buffer_sec,
        },
        zscore_mean=float(mean),
        zscore_std=float(std),
    )


def save_statistics_outputs(store: HabituationStore, result: StatisticsResult) -> tuple[Path, Path, Path]:
    set_poster_style()
    scope_dir = store.source_root / "gui_output" / "stats"
    scope_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.generated_at.replace(":", "").replace("-", "").replace("T", "_")
    output_dir = scope_dir / f"{result.scope}_{result.animal_id}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "statistics.json"
    result_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    fig = Figure(figsize=(16, 14), constrained_layout=True)
    axes = fig.subplots(3, 2).ravel()

    days = result.day_labels
    x = np.arange(len(days))
    loc = [result.locomotion_pct_by_day.get(day, np.nan) for day in days]
    face = [result.face_motion_pct_by_day.get(day, np.nan) for day in days]

    ax = axes[0]
    ax.bar(x, loc, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45, ha="right")
    style_axes(ax, title="Locomotion % by day", xlabel="Day", ylabel="Fraction")

    ax = axes[1]
    ax.bar(x, face, color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45, ha="right")
    style_axes(ax, title="Face motion % by day", xlabel="Day", ylabel="Fraction")

    ax = axes[2]
    bottom = np.zeros(len(days))
    colors = ["tab:green", "tab:purple", "tab:red", "tab:brown"]
    for i, label in enumerate(STATE_LABELS):
        vals = [result.pupil_pct_by_day.get(day, {}).get(label, 0.0) for day in days]
        ax.bar(x, vals, bottom=bottom, label=label, color=colors[i])
        bottom += np.asarray(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45, ha="right")
    style_axes(ax, title="Pupil states by day", xlabel="Day", ylabel="Fraction")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    ax = axes[3]
    lag_vals = []
    lag_labels = []
    for exp_id, vals in result.lag_by_session.items():
        first = next((v for v in vals.values() if np.isfinite(v)), np.nan)
        lag_vals.append(first)
        lag_labels.append(exp_id)
    ax.plot(range(len(lag_vals)), lag_vals, marker="o", color="tab:blue")
    ax.set_xticks(range(len(lag_labels)))
    ax.set_xticklabels(lag_labels, rotation=90, fontsize=8)
    style_axes(ax, title="Lag to first pupil state after 1 min", xlabel="ExpID", ylabel="Lag (s)")

    ax = axes[4]
    for i, label in enumerate(STATE_LABELS):
        ax.plot(result.progress_bins, result.state_probability[i], label=label, color=colors[i])
    style_axes(ax, title="Pupil state probability vs experiment length", xlabel="Progress (%)", ylabel="Probability")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    axes[5].axis("off")
    fig.suptitle(f"Habituation statistics - {result.scope} / {result.animal_id}", y=1.02)
    svg_path, png_path = save_figure(fig, "statistics_summary", output_dir)
    return result_path, svg_path, png_path
