from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import pickle
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import pandas as pd
from scipy import interpolate


SOURCE_ROOT = Path("/data/common/habituation")
REMOTE_ROOT = Path("/data/Remote_Repository")
GUI_OUTPUT_ROOT = SOURCE_ROOT / "gui_output"
INDEX_PATH = GUI_OUTPUT_ROOT / "dataset_index.json"
SETTINGS_PATH = GUI_OUTPUT_ROOT / "settings.json"
SESSION_STATE_DIR = GUI_OUTPUT_ROOT / "session_state"
SESSION_CACHE_DIR = GUI_OUTPUT_ROOT / "session_cache"
FACE_MOTION_CACHE_DIR = GUI_OUTPUT_ROOT / "face_motion_cache"
PUPIL_BRIGHTNESS_CACHE_DIR = GUI_OUTPUT_ROOT / "pupil_brightness_cache"
STATS_DIR = GUI_OUTPUT_ROOT / "stats"
APP_STATE_PATH = GUI_OUTPUT_ROOT / "app_state.json"

CACHE_VERSION = 9
STATISTICS_RESULTS_VERSION = 9
MIN_ANALYSIS_SESSION_DURATION_SEC = 1800.0
SESSION_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d+_[A-Za-z0-9]+$")
EYE_SIMILARITY_CACHE_DIR = GUI_OUTPUT_ROOT / "eye_similarity_cache"
EYE_SIMILARITY_CACHE_VERSION = 1


def ensure_output_dirs() -> None:
    for path in (
        GUI_OUTPUT_ROOT,
        SESSION_STATE_DIR,
        SESSION_CACHE_DIR,
        FACE_MOTION_CACHE_DIR,
        PUPIL_BRIGHTNESS_CACHE_DIR,
        EYE_SIMILARITY_CACHE_DIR,
        STATS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def animal_id_from_exp_id(exp_id: str) -> str:
    parts = exp_id.split("_")
    if len(parts) < 3:
        return "unknown"
    return parts[2]


def _file_signature(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"exists": False, "mtime_ns": None, "size": None, "path": None}
    stat = path.stat()
    return {
        "exists": True,
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
        "path": str(path),
    }


def _jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)


def _coerce_optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def analysis_cutoff_mask(times: np.ndarray, cutoff_sec: float | None) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return np.zeros(0, dtype=bool)
    cutoff = _coerce_optional_float(cutoff_sec)
    if cutoff is None:
        return np.ones(times.shape, dtype=bool)
    return np.isfinite(times) & (times <= cutoff)


def apply_time_mask(values, mask: np.ndarray):
    arr = np.asarray(values)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.ndim == 0:
        return arr
    if arr.shape[0] == 0 or mask.size == 0:
        return arr[:0]
    n = min(arr.shape[0], mask.size)
    return arr[:n][mask[:n]]


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


DEFAULT_PUPIL_PERCENTILES = [25.0, 50.0, 75.0]


def _normalize_percentiles(values, fallback: Iterable[float] | None = None) -> list[float]:
    fallback = list(fallback) if fallback is not None else list(DEFAULT_PUPIL_PERCENTILES)
    try:
        out = [float(v) for v in values]
    except Exception:
        return fallback
    if len(out) != 3 or not all(np.isfinite(out)):
        return fallback
    return out


def _pupil_percentile_signature(values: Iterable[float]) -> str:
    payload = json.dumps([float(v) for v in values], separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def _band_signature(values: Iterable[float]) -> str:
    payload = json.dumps([float(v) for v in values], separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def _safe_exp_name(name: str) -> bool:
    return bool(SESSION_NAME_RE.match(name))


def resolve_locomotion_csv(remote_root: Path, animal_id: str, exp_id: str) -> Path | None:
    exp_dir = remote_root / animal_id / exp_id
    if not exp_dir.exists():
        return None
    exact = exp_dir / f"{exp_id}_frame_times.csv"
    if exact.exists():
        return exact
    matches = sorted(exp_dir.glob("*_frame_times.csv"))
    if matches:
        return matches[0]
    return None


def resolve_wheel_pickle(source_root: Path, remote_root: Path, animal_id: str, exp_id: str) -> Path | None:
    candidates = [
        source_root / animal_id / exp_id / "recordings" / "wheel.pickle",
        source_root / animal_id / exp_id / "wheel.pickle",
        remote_root / animal_id / exp_id / "recordings" / "wheel.pickle",
        remote_root / animal_id / exp_id / "wheel.pickle",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_session_paths(source_root: Path, remote_root: Path, animal_id: str, exp_id: str) -> dict:
    exp_dir = source_root / animal_id / exp_id
    right_video = exp_dir / f"{exp_id}_eye1_right.avi"
    right_pickle = exp_dir / "recordings" / "dlcEyeRight.pickle"
    locomotion_csv = resolve_locomotion_csv(remote_root, animal_id, exp_id)
    wheel_pickle = resolve_wheel_pickle(source_root, remote_root, animal_id, exp_id)
    return {
        "exp_dir": exp_dir,
        "right_video": right_video if right_video.exists() else None,
        "right_pickle": right_pickle if right_pickle.exists() else None,
        "locomotion_csv": locomotion_csv if locomotion_csv and locomotion_csv.exists() else None,
        "wheel_pickle": wheel_pickle if wheel_pickle and wheel_pickle.exists() else None,
    }


def _locomotion_source_signature(paths: dict) -> dict:
    wheel_pickle = paths.get("wheel_pickle")
    if wheel_pickle is not None:
        return {
            "kind": "wheel_pickle",
            "path": str(wheel_pickle),
            "signature": _file_signature(wheel_pickle),
        }
    csv_path = paths.get("locomotion_csv")
    return {
        "kind": "frame_times_csv",
        "path": str(csv_path) if csv_path is not None else None,
        "signature": _file_signature(csv_path),
    }


def _video_time_axis(summary: SessionSummary, frame_count: int | None = None) -> np.ndarray:
    n = int(frame_count if frame_count is not None else (summary.video_frame_count or 0))
    if n <= 0 and summary.video_duration_sec and summary.video_fps:
        n = int(round(float(summary.video_duration_sec) * float(summary.video_fps)))
    if n <= 0:
        return np.array([], dtype=float)
    fps = float(summary.video_fps or 0.0)
    if fps > 0:
        return np.arange(n, dtype=float) / fps
    duration = float(summary.video_duration_sec or 0.0)
    if duration > 0:
        return np.linspace(0.0, duration, n, endpoint=False, dtype=float)
    return np.arange(n, dtype=float)


def _load_timestamp_series(csv_path: Path | None) -> np.ndarray:
    if csv_path is None or not csv_path.exists():
        return np.array([], dtype=float)
    try:
        df = pd.read_csv(csv_path, usecols=["timestamp"])
    except Exception:
        return np.array([], dtype=float)
    if "timestamp" not in df.columns or df.empty:
        return np.array([], dtype=float)
    timestamps = np.asarray(pd.to_numeric(df["timestamp"], errors="coerce"), dtype=float).reshape(-1)
    timestamps = timestamps[np.isfinite(timestamps)]
    if timestamps.size == 0:
        return np.array([], dtype=float)
    return timestamps.astype(float)


def _resample_series_to_times(values, source_t: np.ndarray, target_t: np.ndarray):
    values = np.asarray(values)
    source_t = np.asarray(source_t, dtype=float).reshape(-1)
    target_t = np.asarray(target_t, dtype=float).reshape(-1)
    if target_t.size == 0:
        return np.asarray(values[:0])
    if values.size == 0:
        return np.asarray(values[:0])
    if values.shape[0] == target_t.size:
        return np.asarray(values)
    if source_t.size != values.shape[0]:
        source_t = np.linspace(0.0, float(max(values.shape[0] - 1, 1)), values.shape[0], dtype=float)
    finite = np.isfinite(source_t)
    source_t = source_t[finite]
    if source_t.size == 0:
        return np.asarray(values[:0])
    if values.ndim == 1:
        series = np.asarray(values, dtype=float).reshape(-1)
        series = series[finite]
        if series.size == 0:
            return np.zeros(target_t.shape, dtype=float)
        if source_t.size == 1:
            return np.full(target_t.shape, float(series[0]), dtype=float)
        order = np.argsort(source_t)
        source_t = source_t[order]
        series = series[order]
        unique_t, unique_idx = np.unique(source_t, return_index=True)
        series = series[unique_idx]
        if unique_t.size == 1:
            return np.full(target_t.shape, float(series[0]), dtype=float)
        return np.interp(target_t, unique_t, series, left=float(series[0]), right=float(series[-1])).astype(float)
    flat = np.asarray(values, dtype=float).reshape(values.shape[0], -1)
    flat = flat[finite]
    if flat.size == 0:
        return np.zeros((target_t.size,) + values.shape[1:], dtype=float)
    if source_t.size == 1:
        out = np.repeat(flat[:1], target_t.size, axis=0)
        return out.reshape((target_t.size,) + values.shape[1:]).astype(float)
    order = np.argsort(source_t)
    source_t = source_t[order]
    flat = flat[order]
    unique_t, unique_idx = np.unique(source_t, return_index=True)
    flat = flat[unique_idx]
    if unique_t.size == 1:
        out = np.repeat(flat[:1], target_t.size, axis=0)
        return out.reshape((target_t.size,) + values.shape[1:]).astype(float)
    out = np.empty((target_t.size, flat.shape[1]), dtype=float)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(target_t, unique_t, flat[:, i], left=float(flat[0, i]), right=float(flat[-1, i]))
    return out.reshape((target_t.size,) + values.shape[1:]).astype(float)


def _frame_time_axis(summary: SessionSummary, *, frame_count: int | None = None) -> np.ndarray:
    csv_times = _load_timestamp_series(Path(summary.locomotion_csv) if summary.locomotion_csv else None)
    if csv_times.size:
        return csv_times
    return _video_time_axis(summary, frame_count=frame_count)


def _frame_interval_guess(summary: SessionSummary, timestamps: np.ndarray) -> float:
    fps = float(summary.video_fps or 0.0)
    if fps > 0.0:
        interval = 1.0 / fps
        if np.isfinite(interval) and interval > 0.0:
            return float(interval)
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size > 1:
        deltas = np.diff(times)
        positive = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        if positive.size:
            return float(np.nanmedian(positive))
    return 0.0


def _pad_frame_array(values: np.ndarray, missing_count: int):
    arr = np.asarray(values)
    if arr.ndim == 0 or missing_count <= 0:
        return arr
    pad_shape = (int(missing_count),) + arr.shape[1:]
    if arr.dtype == np.bool_:
        pad = np.zeros(pad_shape, dtype=bool)
    elif np.issubdtype(arr.dtype, np.floating):
        pad = np.full(pad_shape, np.nan, dtype=arr.dtype)
    else:
        arr = arr.astype(float, copy=False)
        pad = np.full(pad_shape, np.nan, dtype=float)
    return np.concatenate([arr, pad], axis=0)


def _pad_frame_row(values: np.ndarray):
    arr = np.asarray(values)
    if arr.ndim == 0 or arr.shape[0] == 0:
        return arr[:0] if arr.ndim > 0 else arr
    return _pad_frame_array(arr[:1], 1)[-1:]


def _expand_frame_gaps(
    summary: SessionSummary,
    times: np.ndarray,
    *values: np.ndarray,
    gap_factor: float = 1.5,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    times = np.asarray(times, dtype=float).reshape(-1)
    arrays = [np.asarray(value) for value in values]
    if times.size == 0:
        return times, [arr[:0] if arr.ndim > 0 else arr for arr in arrays], np.zeros(0, dtype=bool)

    lengths = [times.size]
    for arr in arrays:
        if arr.ndim > 0 and arr.shape[0] > 0:
            lengths.append(arr.shape[0])
    n = min(lengths)
    times = times[:n]
    arrays = [arr[:n] if arr.ndim > 0 and arr.shape[0] > 0 else arr for arr in arrays]
    if times.size <= 1:
        return times, arrays, np.ones(times.shape, dtype=bool)

    expected_interval = _frame_interval_guess(summary, times)
    if not np.isfinite(expected_interval) or expected_interval <= 0.0:
        return times, arrays, np.ones(times.shape, dtype=bool)

    pad_rows = [_pad_frame_row(arr) for arr in arrays]
    expanded_times: list[float] = []
    observed_mask: list[bool] = []
    expanded_arrays: list[list[np.ndarray]] = [[] for _ in arrays]

    for idx, current_t in enumerate(times):
        if idx > 0:
            prev_t = float(times[idx - 1])
            delta = float(current_t) - prev_t
            if np.isfinite(delta) and delta > expected_interval * gap_factor:
                missing_count = int(np.rint(delta / expected_interval)) - 1
                if missing_count > 0:
                    for missing_idx in range(1, missing_count + 1):
                        expanded_times.append(prev_t + expected_interval * missing_idx)
                        observed_mask.append(False)
                        for out, pad_row in zip(expanded_arrays, pad_rows):
                            if pad_row.size:
                                out.append(pad_row)
        expanded_times.append(float(current_t))
        observed_mask.append(True)
        for out, arr in zip(expanded_arrays, arrays):
            if arr.ndim > 0 and arr.shape[0] > 0:
                out.append(arr[idx : idx + 1])

    expanded_arrays_out = [
        np.concatenate(parts, axis=0) if parts else (arr[:0] if arr.ndim > 0 else arr)
        for parts, arr in zip(expanded_arrays, arrays)
    ]
    return np.asarray(expanded_times, dtype=float), expanded_arrays_out, np.asarray(observed_mask, dtype=bool)


def _frame_observed_mask(bundle: SessionBundle) -> np.ndarray:
    times = np.asarray(bundle.t, dtype=float).reshape(-1)
    observed = np.asarray(getattr(bundle, "frame_observed", np.ones(times.shape, dtype=bool)), dtype=bool).reshape(-1)
    if observed.size >= times.size:
        return observed[: times.size]
    out = np.ones(times.shape, dtype=bool)
    out[: observed.size] = observed
    return out


def _video_metadata(video_path: Path | None) -> dict:
    if video_path is None or not video_path.exists():
        return {"fps": None, "frame_count": None, "duration_sec": None}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"fps": None, "frame_count": None, "duration_sec": None}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration_sec = float(frame_count / fps) if fps else None
    return {"fps": fps or None, "frame_count": frame_count or None, "duration_sec": duration_sec}


@dataclass
class SessionSummary:
    animal_id: str
    exp_id: str
    date: str
    session_number: int
    exp_dir: str
    right_video: str | None
    right_pickle: str | None
    locomotion_csv: str | None
    has_right_video: bool
    has_right_pickle: bool
    has_locomotion_csv: bool
    video_fps: float | None
    video_frame_count: int | None
    video_duration_sec: float | None
    source_signature: dict

    @property
    def duration_sec(self) -> float:
        # Use the longer observed session span so overall plots/stats do not
        # truncate locomotion when the right video is shorter.
        video_duration = float(self.video_duration_sec or 0.0)
        locomotion_duration = 0.0
        if self.locomotion_csv and Path(self.locomotion_csv).exists():
            try:
                df = pd.read_csv(self.locomotion_csv, usecols=["timestamp"])
                if not df.empty:
                    locomotion_duration = float(df["timestamp"].iloc[-1])
            except Exception:
                pass
        return max(video_duration, locomotion_duration)

    @property
    def sort_key(self):
        return (self.date, self.session_number, self.exp_id)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionSummary":
        return cls(**data)


@dataclass
class DatasetIndex:
    generated_at: str
    source_root: str
    remote_root: str
    sessions: list[SessionSummary] = field(default_factory=list)

    def sessions_by_animal(self) -> dict[str, list[SessionSummary]]:
        out: dict[str, list[SessionSummary]] = {}
        for summary in sorted(self.sessions, key=lambda s: s.sort_key):
            out.setdefault(summary.animal_id, []).append(summary)
        return out

    def session_by_id(self) -> dict[str, SessionSummary]:
        return {s.exp_id: s for s in self.sessions}

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "source_root": self.source_root,
            "remote_root": self.remote_root,
            "sessions": [s.to_dict() for s in self.sessions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetIndex":
        return cls(
            generated_at=data.get("generated_at", ""),
            source_root=data.get("source_root", str(SOURCE_ROOT)),
            remote_root=data.get("remote_root", str(REMOTE_ROOT)),
            sessions=[SessionSummary.from_dict(item) for item in data.get("sessions", [])],
        )


@dataclass
class SessionBundle:
    summary: SessionSummary
    t: np.ndarray
    frame_observed: np.ndarray
    radius: np.ndarray
    x: np.ndarray
    y: np.ndarray
    velocity: np.ndarray
    qc: np.ndarray
    in_eye: np.ndarray
    brake_raw: np.ndarray
    brake_t: np.ndarray
    wheel_pos: np.ndarray
    locomotion: np.ndarray
    locomotion_t: np.ndarray
    eye_lid_x: np.ndarray
    eye_lid_y: np.ndarray
    eyeX: np.ndarray
    eyeY: np.ndarray
    pupilX: np.ndarray
    pupilY: np.ndarray
    source_signature: dict

    @property
    def visible_mask_base(self) -> np.ndarray:
        visible = np.isfinite(self.radius)
        if self.frame_observed.shape == self.radius.shape:
            visible = visible & self.frame_observed.astype(bool)
        if self.in_eye.ndim == 2 and self.in_eye.shape[0] == self.radius.shape[0]:
            visible = visible & np.all(self.in_eye.astype(bool), axis=1)
        if self.qc.shape == self.radius.shape:
            visible = visible & (self.qc == 0)
        return visible


class HabituationStore:
    def __init__(self, source_root: Path = SOURCE_ROOT, remote_root: Path = REMOTE_ROOT):
        ensure_output_dirs()
        self.source_root = Path(source_root)
        self.remote_root = Path(remote_root)
        self.index: DatasetIndex | None = None
        self.settings = self.load_settings()

    @property
    def source_version(self) -> int:
        return CACHE_VERSION

    def load_index(self, prefer_cache: bool = True) -> DatasetIndex:
        if prefer_cache and INDEX_PATH.exists():
            try:
                data = _load_json(INDEX_PATH, {})
                index = DatasetIndex.from_dict(data)
                self.index = index
                return index
            except Exception:
                pass
        index = self.refresh_index()
        return index

    def refresh_index(self) -> DatasetIndex:
        sessions: list[SessionSummary] = []
        for animal_dir in sorted([p for p in self.source_root.iterdir() if p.is_dir()]):
            animal_id = animal_dir.name
            for exp_dir in sorted([p for p in animal_dir.iterdir() if p.is_dir() and _safe_exp_name(p.name)]):
                exp_id = exp_dir.name
                parts = exp_id.split("_")
                date = parts[0] if parts else ""
                session_number = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                paths = resolve_session_paths(self.source_root, self.remote_root, animal_id, exp_id)
                video_meta = _video_metadata(paths["right_video"])
                sig = {
                    "right_video": _file_signature(paths["right_video"]),
                    "right_pickle": _file_signature(paths["right_pickle"]),
                    "locomotion_source": _locomotion_source_signature(paths),
                    "cache_version": self.source_version,
                }
                sessions.append(
                    SessionSummary(
                        animal_id=animal_id,
                        exp_id=exp_id,
                        date=date,
                        session_number=session_number,
                        exp_dir=str(exp_dir),
                        right_video=str(paths["right_video"]) if paths["right_video"] else None,
                        right_pickle=str(paths["right_pickle"]) if paths["right_pickle"] else None,
                        locomotion_csv=str(paths["locomotion_csv"]) if paths["locomotion_csv"] else None,
                        has_right_video=paths["right_video"] is not None,
                        has_right_pickle=paths["right_pickle"] is not None,
                        has_locomotion_csv=paths["locomotion_csv"] is not None,
                        video_fps=video_meta["fps"],
                        video_frame_count=video_meta["frame_count"],
                        video_duration_sec=video_meta["duration_sec"],
                        source_signature=sig,
                    )
                )
        index = DatasetIndex(
            generated_at=datetime.now().isoformat(timespec="seconds"),
            source_root=str(self.source_root),
            remote_root=str(self.remote_root),
            sessions=sorted(sessions, key=lambda s: s.sort_key),
        )
        _save_json(INDEX_PATH, index.to_dict())
        self.index = index
        return index

    def load_settings(self) -> dict:
        default = {
            "global": {
                "locomotion_threshold": 0.35,
                "face_motion_threshold_mad": 3.0,
                "pupil_percentile_cutoffs": list(DEFAULT_PUPIL_PERCENTILES),
                "pupil_percentile_signature": _pupil_percentile_signature(DEFAULT_PUPIL_PERCENTILES),
                "pupil_missing_buffer_sec": 1.0,
                "stats_dirty": True,
            },
            "animals": {},
        }
        settings = _load_json(SETTINGS_PATH, default)
        if not isinstance(settings, dict):
            settings = default
        if not isinstance(settings.get("global"), dict):
            settings["global"] = {}
        if not isinstance(settings.get("animals"), dict):
            settings["animals"] = {}
        global_settings = settings["global"]
        changed = False
        global_settings.setdefault("locomotion_threshold", default["global"]["locomotion_threshold"])
        global_settings.setdefault("face_motion_threshold_mad", default["global"]["face_motion_threshold_mad"])
        if "pupil_missing_buffer_sec" not in global_settings:
            changed = True
        global_settings.setdefault("pupil_missing_buffer_sec", default["global"]["pupil_missing_buffer_sec"])
        global_settings.setdefault("stats_dirty", True)
        legacy_global_values = global_settings.get("pupil_percentile_cutoffs", None)
        global_percentiles = _normalize_percentiles(legacy_global_values)
        if legacy_global_values != global_percentiles:
            global_settings["pupil_percentile_cutoffs"] = global_percentiles
            changed = True
        if legacy_global_values in (None, [], ()):
            fallback_percentiles = None
            legacy_source = settings["animals"].get("All")
            if isinstance(legacy_source, dict) and "percentile_cutoffs" in legacy_source:
                fallback_percentiles = _normalize_percentiles(
                    legacy_source.get("percentile_cutoffs", []), fallback=global_percentiles
                )
            if fallback_percentiles is None:
                for animal_settings in settings["animals"].values():
                    if isinstance(animal_settings, dict) and "percentile_cutoffs" in animal_settings:
                        fallback_percentiles = _normalize_percentiles(
                            animal_settings.get("percentile_cutoffs", []), fallback=global_percentiles
                        )
                        break
            if fallback_percentiles is not None:
                global_percentiles = fallback_percentiles
                global_settings["pupil_percentile_cutoffs"] = [float(v) for v in global_percentiles]
                changed = True
        global_signature = _pupil_percentile_signature(global_percentiles)
        if global_settings.get("pupil_percentile_signature") != global_signature:
            global_settings["pupil_percentile_signature"] = global_signature
            changed = True

        for animal_id, animal_settings in list(settings["animals"].items()):
            if not isinstance(animal_settings, dict):
                animal_settings = {}
                settings["animals"][animal_id] = animal_settings
                changed = True
            legacy_cutoffs = animal_settings.pop("percentile_cutoffs", None) if "percentile_cutoffs" in animal_settings else None
            if legacy_cutoffs is not None:
                legacy_cutoffs = _normalize_percentiles(legacy_cutoffs, fallback=global_percentiles)
                animal_settings.setdefault("threshold_signature", _pupil_percentile_signature(legacy_cutoffs))
                changed = True
            animal_settings.setdefault("threshold_signature", "")
            animal_settings.setdefault("threshold_values", [0.0, 0.5, 1.0])
            animal_settings.setdefault("zscore_mean", None)
            animal_settings.setdefault("zscore_std", None)
            animal_settings.setdefault("baseline_dirty", True)
            animal_settings.setdefault("stats_dirty", True)

        if changed:
            _save_json(SETTINGS_PATH, settings)
        return settings

    def save_settings(self) -> None:
        _save_json(SETTINGS_PATH, self.settings)

    def global_pupil_percentile_cutoffs(self) -> list[float]:
        global_settings = self.settings.get("global", {})
        return _normalize_percentiles(global_settings.get("pupil_percentile_cutoffs", []))

    def pupil_percentile_signature(self, percentiles: Iterable[float] | None = None) -> str:
        if percentiles is None:
            percentiles = self.global_pupil_percentile_cutoffs()
        return _pupil_percentile_signature(_normalize_percentiles(percentiles))

    def set_global_pupil_percentile_cutoffs(self, percentiles: Iterable[float], *, mark_stats_dirty: bool = True) -> list[float]:
        values = _normalize_percentiles(percentiles)
        self.settings.setdefault("global", {})
        self.settings["global"]["pupil_percentile_cutoffs"] = [float(v) for v in values]
        self.settings["global"]["pupil_percentile_signature"] = self.pupil_percentile_signature(values)
        if mark_stats_dirty:
            self.mark_stats_dirty()
        else:
            self.save_settings()
        return values

    def global_pupil_missing_buffer_sec(self) -> float:
        global_settings = self.settings.get("global", {})
        value = global_settings.get("pupil_missing_buffer_sec", 1.0)
        try:
            return float(value)
        except Exception:
            return 1.0

    def set_global_pupil_missing_buffer_sec(self, value: float, *, mark_stats_dirty: bool = True) -> float:
        self.settings.setdefault("global", {})
        buffer_sec = max(0.0, float(value))
        self.settings["global"]["pupil_missing_buffer_sec"] = buffer_sec
        if mark_stats_dirty:
            self.mark_all_animals_stats_dirty()
        else:
            self.save_settings()
        return buffer_sec

    def load_app_state(self) -> dict:
        default = {
            "selected_animal": "All",
            "selected_exp_id": "",
            "view_mode": "Session",
        }
        state = _load_json(APP_STATE_PATH, default)
        if not isinstance(state, dict):
            state = default
        for key, value in default.items():
            state.setdefault(key, value)
        return state

    def save_app_state(self, state: dict) -> None:
        _save_json(APP_STATE_PATH, state)

    def session_state_path(self, exp_id: str) -> Path:
        return SESSION_STATE_DIR / f"{exp_id}.json"

    def base_cache_path(self, exp_id: str) -> Path:
        return SESSION_CACHE_DIR / f"{exp_id}_base.pkl"

    def face_motion_cache_path(self, exp_id: str) -> Path:
        return FACE_MOTION_CACHE_DIR / f"{exp_id}_face_motion.pkl"

    def pupil_brightness_cache_path(self, exp_id: str) -> Path:
        return PUPIL_BRIGHTNESS_CACHE_DIR / f"{exp_id}_pupil_brightness.pkl"

    def eye_similarity_cache_path(self, exp_id: str, band: tuple[float, float]) -> Path:
        low, high = float(band[0]), float(band[1])
        if high < low:
            low, high = high, low
        signature = _band_signature((low, high))
        return EYE_SIMILARITY_CACHE_DIR / f"{exp_id}_eye_similarity_{signature}.pkl"

    def is_session_forced_do_not_use(self, exp_id: str) -> bool:
        summary = self.get_session_summary(exp_id)
        if summary is None:
            return False
        if summary.animal_id == "TEST":
            return True
        return float(summary.video_duration_sec or 0.0) < MIN_ANALYSIS_SESSION_DURATION_SEC

    def load_session_state(self, exp_id: str) -> dict:
        path = self.session_state_path(exp_id)
        default = {
            "manual_masks": [],
            "manual_masks_timebase": None,
            "extra_large_calibration_interval": None,
            "extra_large_calibration_reference_mean": None,
            "extra_large_calibration_reference_std": None,
            "extra_large_calibration_reference_band": None,
            "extra_large_calibration_similarity_mean": None,
            "extra_large_calibration_similarity_std": None,
            "extra_large_calibration_similarity_cutoff": None,
            "extra_large_calibration_threshold": None,
            "extra_large_calibration_confirmed": False,
            "last_stats_signature": "",
            "last_stats_output_dir": "",
            "stats_dirty": True,
            "preprocessed": False,
            "deeplabcut_reference": False,
            "do_not_use": False,
            "analysis_cutoff_sec": None,
            "timebase_aligned": False,
            "threshold_signature": "",
        }
        state = _load_json(path, default)
        if not isinstance(state, dict):
            state = {}
        state.setdefault("manual_masks", [])
        state.setdefault("manual_masks_timebase", None)
        state.setdefault("extra_large_calibration_interval", None)
        state.setdefault("extra_large_calibration_reference_mean", None)
        state.setdefault("extra_large_calibration_reference_std", None)
        state.setdefault("extra_large_calibration_reference_band", None)
        state.setdefault("extra_large_calibration_similarity_mean", None)
        state.setdefault("extra_large_calibration_similarity_std", None)
        state.setdefault("extra_large_calibration_similarity_cutoff", None)
        state.setdefault("extra_large_calibration_threshold", None)
        state.setdefault("extra_large_calibration_confirmed", False)
        state.setdefault("last_stats_signature", "")
        state.setdefault("last_stats_output_dir", "")
        state.setdefault("stats_dirty", True)
        state.setdefault("preprocessed", False)
        state.setdefault("deeplabcut_reference", False)
        state.setdefault("do_not_use", False)
        state.setdefault("analysis_cutoff_sec", None)
        state.setdefault("timebase_aligned", False)
        state.setdefault("threshold_signature", "")
        state["analysis_cutoff_sec"] = _coerce_optional_float(state.get("analysis_cutoff_sec"))
        state["timebase_aligned"] = bool(state.get("timebase_aligned", False))
        forced_do_not_use = self.is_session_forced_do_not_use(exp_id)
        if forced_do_not_use and not state.get("do_not_use", False):
            state["do_not_use"] = True
            _save_json(path, state)
            self.mark_all_animals_stats_dirty()
        return state

    def save_session_state(self, exp_id: str, state: dict) -> None:
        _save_json(self.session_state_path(exp_id), state)

    def session_threshold_signature(self, exp_id: str) -> str:
        state = self.load_session_state(exp_id)
        return str(state.get("threshold_signature", ""))

    def session_extra_large_calibration(self, exp_id: str) -> dict | None:
        state = self.load_session_state(exp_id)
        if not bool(state.get("extra_large_calibration_confirmed", False)):
            return None
        interval = state.get("extra_large_calibration_interval", None)
        if interval is None:
            return None
        try:
            start, end = float(interval[0]), float(interval[1])
        except Exception:
            return None
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            return None
        reference_band = state.get("extra_large_calibration_reference_band", None)
        similarity_cutoff = _coerce_optional_float(state.get("extra_large_calibration_similarity_cutoff", None))
        reference_mean = _coerce_optional_float(state.get("extra_large_calibration_reference_mean", None))
        reference_std = _coerce_optional_float(state.get("extra_large_calibration_reference_std", None))
        similarity_mean = _coerce_optional_float(state.get("extra_large_calibration_similarity_mean", None))
        similarity_std = _coerce_optional_float(state.get("extra_large_calibration_similarity_std", None))
        if (
            reference_band is not None
            and similarity_cutoff is not None
            and reference_mean is not None
            and reference_std is not None
        ):
            try:
                band_low, band_high = float(reference_band[0]), float(reference_band[1])
            except Exception:
                band_low = band_high = np.nan
            if np.isfinite(band_low) and np.isfinite(band_high):
                if band_high < band_low:
                    band_low, band_high = band_high, band_low
                out = {
                    "interval": (float(start), float(end)),
                    "reference_mean": float(reference_mean),
                    "reference_std": float(reference_std),
                    "reference_band": (float(band_low), float(band_high)),
                    "similarity_cutoff": float(similarity_cutoff),
                    "confirmed": True,
                }
                if similarity_mean is not None and similarity_std is not None:
                    out["similarity_mean"] = float(similarity_mean)
                    out["similarity_std"] = float(similarity_std)
                return out
        legacy_threshold = _coerce_optional_float(state.get("extra_large_calibration_threshold", None))
        if legacy_threshold is None:
            return None
        return {
            "interval": (float(start), float(end)),
            "threshold": float(legacy_threshold),
            "confirmed": True,
            "legacy": True,
        }

    def set_session_extra_large_calibration(
        self,
        exp_id: str,
        calibration: dict | None,
        *,
        confirmed: bool = True,
    ) -> dict | None:
        state = self.load_session_state(exp_id)
        if not calibration:
            state["extra_large_calibration_interval"] = None
            state["extra_large_calibration_reference_mean"] = None
            state["extra_large_calibration_reference_std"] = None
            state["extra_large_calibration_reference_band"] = None
            state["extra_large_calibration_similarity_mean"] = None
            state["extra_large_calibration_similarity_std"] = None
            state["extra_large_calibration_similarity_cutoff"] = None
            state["extra_large_calibration_threshold"] = None
            state["extra_large_calibration_confirmed"] = False
        else:
            interval = calibration.get("interval", None)
            if interval is None:
                state["extra_large_calibration_interval"] = None
                state["extra_large_calibration_reference_mean"] = None
                state["extra_large_calibration_reference_std"] = None
                state["extra_large_calibration_reference_band"] = None
                state["extra_large_calibration_similarity_mean"] = None
                state["extra_large_calibration_similarity_std"] = None
                state["extra_large_calibration_similarity_cutoff"] = None
                state["extra_large_calibration_threshold"] = None
                state["extra_large_calibration_confirmed"] = False
            else:
                start, end = float(interval[0]), float(interval[1])
                if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                    state["extra_large_calibration_interval"] = None
                    state["extra_large_calibration_reference_mean"] = None
                    state["extra_large_calibration_reference_std"] = None
                    state["extra_large_calibration_reference_band"] = None
                    state["extra_large_calibration_similarity_mean"] = None
                    state["extra_large_calibration_similarity_std"] = None
                    state["extra_large_calibration_similarity_cutoff"] = None
                    state["extra_large_calibration_threshold"] = None
                    state["extra_large_calibration_confirmed"] = False
                else:
                    state["extra_large_calibration_interval"] = [float(start), float(end)]
                    reference_mean = _coerce_optional_float(calibration.get("reference_mean"))
                    reference_std = _coerce_optional_float(calibration.get("reference_std"))
                    reference_band = calibration.get("reference_band", None)
                    similarity_mean = _coerce_optional_float(calibration.get("similarity_mean"))
                    similarity_std = _coerce_optional_float(calibration.get("similarity_std"))
                    similarity_cutoff = _coerce_optional_float(calibration.get("similarity_cutoff"))
                    if reference_mean is None or reference_std is None or similarity_cutoff is None:
                        state["extra_large_calibration_reference_mean"] = None
                        state["extra_large_calibration_reference_std"] = None
                        state["extra_large_calibration_reference_band"] = None
                        state["extra_large_calibration_similarity_mean"] = None
                        state["extra_large_calibration_similarity_std"] = None
                        state["extra_large_calibration_similarity_cutoff"] = None
                        state["extra_large_calibration_threshold"] = None
                        state["extra_large_calibration_confirmed"] = False
                    else:
                        try:
                            band_low, band_high = float(reference_band[0]), float(reference_band[1])
                        except Exception:
                            band_low = band_high = np.nan
                        if not np.isfinite(band_low) or not np.isfinite(band_high):
                            state["extra_large_calibration_reference_mean"] = None
                            state["extra_large_calibration_reference_std"] = None
                            state["extra_large_calibration_reference_band"] = None
                            state["extra_large_calibration_similarity_mean"] = None
                            state["extra_large_calibration_similarity_std"] = None
                            state["extra_large_calibration_similarity_cutoff"] = None
                            state["extra_large_calibration_threshold"] = None
                            state["extra_large_calibration_confirmed"] = False
                        else:
                            if band_high < band_low:
                                band_low, band_high = band_high, band_low
                            state["extra_large_calibration_reference_mean"] = float(reference_mean)
                            state["extra_large_calibration_reference_std"] = float(reference_std)
                            state["extra_large_calibration_reference_band"] = [float(band_low), float(band_high)]
                            state["extra_large_calibration_similarity_mean"] = float(similarity_mean) if similarity_mean is not None else None
                            state["extra_large_calibration_similarity_std"] = float(similarity_std) if similarity_std is not None else None
                            state["extra_large_calibration_similarity_cutoff"] = float(similarity_cutoff)
                            state["extra_large_calibration_threshold"] = None
                            state["extra_large_calibration_confirmed"] = bool(confirmed)
        state["stats_dirty"] = True
        state["last_stats_signature"] = ""
        self.save_session_state(exp_id, state)
        self.mark_all_animals_stats_dirty()
        return self.session_extra_large_calibration(exp_id)

    def clear_session_extra_large_calibration(self, exp_id: str) -> None:
        self.set_session_extra_large_calibration(exp_id, None)

    def is_session_preprocessed(self, exp_id: str) -> bool:
        state = self.load_session_state(exp_id)
        return bool(state.get("preprocessed", False))

    def is_deeplabcut_reference_session(self, exp_id: str) -> bool:
        state = self.load_session_state(exp_id)
        return bool(state.get("deeplabcut_reference", False))

    def is_session_do_not_use(self, exp_id: str) -> bool:
        if self.is_session_forced_do_not_use(exp_id):
            return True
        state = self.load_session_state(exp_id)
        return bool(state.get("do_not_use", False))

    def session_analysis_cutoff_sec(self, exp_id: str) -> float | None:
        state = self.load_session_state(exp_id)
        return _coerce_optional_float(state.get("analysis_cutoff_sec"))

    def effective_session_duration_sec(self, exp_id: str) -> float:
        summary = self.get_session_summary(exp_id)
        if summary is None:
            return 0.0
        cutoff = self.session_analysis_cutoff_sec(exp_id)
        if cutoff is None:
            return float(summary.duration_sec)
        video_duration = float(summary.video_duration_sec or 0.0)
        if video_duration > 0.0:
            return float(min(video_duration, cutoff))
        return float(min(float(summary.duration_sec), cutoff))

    def set_session_analysis_cutoff(self, exp_id: str, cutoff_sec: float | None) -> float | None:
        state = self.load_session_state(exp_id)
        normalized = _coerce_optional_float(cutoff_sec)
        state["analysis_cutoff_sec"] = normalized
        if normalized is not None:
            state["timebase_aligned"] = True
        self.save_session_state(exp_id, state)
        self.mark_all_animals_stats_dirty()
        return normalized

    def is_session_timebase_aligned(self, exp_id: str) -> bool:
        state = self.load_session_state(exp_id)
        return bool(state.get("timebase_aligned", False))

    def set_session_timebase_aligned(self, exp_id: str, aligned: bool) -> bool:
        state = self.load_session_state(exp_id)
        state["timebase_aligned"] = bool(aligned)
        self.save_session_state(exp_id, state)
        return bool(aligned)


    def deeplabcut_reference_sessions(self, animal_id: str | None = None) -> list[SessionSummary]:
        if animal_id is None or animal_id == "All":
            sessions = self.dataset_sessions()
        else:
            sessions = self.sessions_for_animal(animal_id)
        return [summary for summary in sessions if self.is_deeplabcut_reference_session(summary.exp_id)]

    def do_not_use_sessions(self, animal_id: str | None = None) -> list[SessionSummary]:
        if animal_id is None or animal_id == "All":
            sessions = self.dataset_sessions()
        else:
            sessions = self.sessions_for_animal(animal_id)
        return [
            summary
            for summary in sessions
            if self.is_session_do_not_use(summary.exp_id) and not self.is_deeplabcut_reference_session(summary.exp_id)
        ]

    def set_session_preprocessed(self, exp_id: str, preprocessed: bool, *, threshold_signature: str | None = None) -> None:
        state = self.load_session_state(exp_id)
        state["preprocessed"] = bool(preprocessed)
        if threshold_signature is None:
            threshold_signature = self.pupil_percentile_signature()
        state["threshold_signature"] = str(threshold_signature)
        self.save_session_state(exp_id, state)

    def set_session_deeplabcut_reference(self, exp_id: str, reference: bool) -> None:
        state = self.load_session_state(exp_id)
        state["deeplabcut_reference"] = bool(reference)
        self.save_session_state(exp_id, state)
        self.mark_all_animals_stats_dirty()

    def set_session_do_not_use(self, exp_id: str, do_not_use: bool) -> None:
        if self.is_session_forced_do_not_use(exp_id):
            do_not_use = True
        state = self.load_session_state(exp_id)
        state["do_not_use"] = bool(do_not_use)
        self.save_session_state(exp_id, state)
        self.mark_all_animals_stats_dirty()

    def _current_source_signature(self, summary: SessionSummary) -> dict:
        paths = resolve_session_paths(
            self.source_root,
            self.remote_root,
            summary.animal_id,
            summary.exp_id,
        )
        return {
            "right_video": _file_signature(paths["right_video"]),
            "right_pickle": _file_signature(paths["right_pickle"]),
            "locomotion_source": _locomotion_source_signature(paths),
            "cache_version": self.source_version,
        }

    def _signature_matches(self, a: dict, b: dict) -> bool:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    def get_session_summary(self, exp_id: str) -> SessionSummary | None:
        if self.index is None:
            self.load_index()
        if self.index is None:
            return None
        return self.index.session_by_id().get(exp_id)

    def load_frame_timestamps(self, exp_id: str) -> np.ndarray:
        summary = self.get_session_summary(exp_id)
        if summary is None:
            return np.array([], dtype=float)
        return _frame_time_axis(summary, frame_count=summary.video_frame_count)

    def load_session_bundle(self, exp_id: str, *, force_rebuild: bool = False) -> SessionBundle:
        summary = self.get_session_summary(exp_id)
        if summary is None:
            raise FileNotFoundError(f"Unknown expID: {exp_id}")
        cache_path = self.base_cache_path(exp_id)
        current_sig = self._current_source_signature(summary)
        if not force_rebuild and cache_path.exists():
            try:
                cached = _load_pickle(cache_path)
                if (
                    isinstance(cached, dict)
                    and cached.get("cache_version") == self.source_version
                    and self._signature_matches(cached.get("source_signature", {}), current_sig)
                ):
                    return SessionBundle(
                        summary=summary,
                        t=np.asarray(cached["t"], dtype=float),
                        frame_observed=np.asarray(cached.get("frame_observed", np.ones_like(cached["t"], dtype=bool)), dtype=bool),
                        radius=np.asarray(cached["radius"], dtype=float),
                        x=np.asarray(cached["x"], dtype=float),
                        y=np.asarray(cached["y"], dtype=float),
                        velocity=np.asarray(cached["velocity"], dtype=float),
                        qc=np.asarray(cached["qc"], dtype=float),
                        in_eye=np.asarray(cached["in_eye"], dtype=bool),
                        brake_raw=np.asarray(cached["brake_raw"], dtype=float),
                        brake_t=np.asarray(cached.get("brake_t", []), dtype=float),
                        wheel_pos=np.asarray(cached["wheel_pos"], dtype=float),
                        locomotion=np.asarray(cached["locomotion"], dtype=float),
                        locomotion_t=np.asarray(cached["locomotion_t"], dtype=float),
                        eye_lid_x=np.asarray(cached.get("eye_lid_x", []), dtype=float),
                        eye_lid_y=np.asarray(cached.get("eye_lid_y", []), dtype=float),
                        eyeX=np.asarray(cached.get("eyeX", []), dtype=float),
                        eyeY=np.asarray(cached.get("eyeY", []), dtype=float),
                        pupilX=np.asarray(cached.get("pupilX", []), dtype=float),
                        pupilY=np.asarray(cached.get("pupilY", []), dtype=float),
                        source_signature=cached.get("source_signature", current_sig),
                    )
            except Exception:
                pass
        bundle = self._build_session_bundle(summary, current_sig=current_sig)
        return bundle

    def _load_eye_pickle(self, path: Path) -> dict:
        with path.open("rb") as f:
            return pickle.load(f)

    def _build_session_bundle(self, summary: SessionSummary, *, current_sig: dict) -> SessionBundle:
        if summary.right_pickle is None or not Path(summary.right_pickle).exists():
            raise FileNotFoundError(f"Missing right-eye pickle for {summary.exp_id}")
        eye = self._load_eye_pickle(Path(summary.right_pickle))
        radius = np.asarray(eye.get("radius", []), dtype=float).reshape(-1)
        x = np.asarray(eye.get("x", np.full(radius.shape, np.nan)), dtype=float).reshape(-1)
        y = np.asarray(eye.get("y", np.full(radius.shape, np.nan)), dtype=float).reshape(-1)
        velocity = np.asarray(eye.get("velocity", np.full(radius.shape, np.nan)), dtype=float).reshape(-1)
        qc = np.asarray(eye.get("qc", np.zeros(radius.shape)), dtype=float).reshape(-1)
        in_eye = np.asarray(eye.get("inEye", np.ones((radius.shape[0], 1), dtype=bool)))
        if in_eye.ndim == 1:
            in_eye = in_eye[:, None]
        eye_lid_x = np.asarray(eye.get("eye_lid_x", []), dtype=float)
        eye_lid_y = np.asarray(eye.get("eye_lid_y", []), dtype=float)
        eyeX = np.asarray(eye.get("eyeX", []), dtype=float)
        eyeY = np.asarray(eye.get("eyeY", []), dtype=float)
        pupilX = np.asarray(eye.get("pupilX", []), dtype=float)
        pupilY = np.asarray(eye.get("pupilY", []), dtype=float)

        try:
            brake_t, brake_raw = self._load_lock_trace(summary.animal_id, summary.exp_id)
        except Exception:
            brake_t = np.array([], dtype=float)
            brake_raw = np.array([], dtype=float)
        wheel_trace = self._load_existing_wheel_trace(summary.animal_id, summary.exp_id)
        if wheel_trace is None:
            locomotion_t, _, wheel_pos, locomotion = self._load_locomotion_trace(
                summary.animal_id, summary.exp_id
            )
        else:
            locomotion_t, _, wheel_pos, locomotion = wheel_trace
        frame_count = int(summary.video_frame_count or radius.shape[0] or 0)
        source_t = _video_time_axis(summary, frame_count=frame_count if frame_count > 0 else radius.shape[0])
        frame_t = _load_timestamp_series(Path(summary.locomotion_csv) if summary.locomotion_csv else None)
        if frame_t.size == 0 and radius.size:
            frame_t = source_t[: radius.shape[0]] if source_t.size else np.arange(radius.shape[0], dtype=float)
        if frame_t.size == 0 and radius.size:
            frame_t = np.arange(radius.shape[0], dtype=float)
        if frame_t.size == 0:
            raise ValueError(f"No aligned samples found for {summary.exp_id}")
        if radius.size and frame_t.size != radius.size:
            radius = _resample_series_to_times(radius, source_t, frame_t)
            x = _resample_series_to_times(x, source_t, frame_t)
            y = _resample_series_to_times(y, source_t, frame_t)
            velocity = _resample_series_to_times(velocity, source_t, frame_t)
            qc = _resample_series_to_times(qc, source_t, frame_t)
            in_eye = _resample_series_to_times(in_eye.astype(float), source_t, frame_t) > 0.5
            eye_lid_x = _resample_series_to_times(eye_lid_x, source_t, frame_t)
            eye_lid_y = _resample_series_to_times(eye_lid_y, source_t, frame_t)
            eyeX = _resample_series_to_times(eyeX, source_t, frame_t)
            eyeY = _resample_series_to_times(eyeY, source_t, frame_t)
            pupilX = _resample_series_to_times(pupilX, source_t, frame_t)
            pupilY = _resample_series_to_times(pupilY, source_t, frame_t)
        else:
            n = min(len(radius), len(frame_t))
            if n <= 0:
                raise ValueError(f"No aligned samples found for {summary.exp_id}")
            radius = radius[:n]
            x = x[:n]
            y = y[:n]
            velocity = velocity[:n]
            qc = qc[:n]
            in_eye = in_eye[:n]
            eye_lid_x = eye_lid_x[:n] if eye_lid_x.ndim >= 1 else eye_lid_x
            eye_lid_y = eye_lid_y[:n] if eye_lid_y.ndim >= 1 else eye_lid_y
            eyeX = eyeX[:n] if eyeX.ndim >= 1 else eyeX
            eyeY = eyeY[:n] if eyeY.ndim >= 1 else eyeY
            pupilX = pupilX[:n] if pupilX.ndim >= 1 else pupilX
            pupilY = pupilY[:n] if pupilY.ndim >= 1 else pupilY
            frame_t = frame_t[:n]
        t, padded_arrays, frame_observed = _expand_frame_gaps(
            summary,
            frame_t,
            radius,
            x,
            y,
            velocity,
            qc,
            in_eye,
            eye_lid_x,
            eye_lid_y,
            eyeX,
            eyeY,
            pupilX,
            pupilY,
        )
        (
            radius,
            x,
            y,
            velocity,
            qc,
            in_eye,
            eye_lid_x,
            eye_lid_y,
            eyeX,
            eyeY,
            pupilX,
            pupilY,
        ) = padded_arrays

        bundle = SessionBundle(
            summary=summary,
            t=t.astype(float),
            frame_observed=frame_observed.astype(bool),
            radius=radius.astype(np.float32),
            x=x.astype(np.float32),
            y=y.astype(np.float32),
            velocity=velocity.astype(np.float32),
            qc=qc.astype(np.float32),
            in_eye=in_eye.astype(bool),
            brake_raw=brake_raw.astype(np.float32),
            brake_t=brake_t.astype(np.float64),
            wheel_pos=wheel_pos.astype(np.float32),
            locomotion=locomotion.astype(np.float32),
            locomotion_t=locomotion_t.astype(np.float32),
            eye_lid_x=eye_lid_x.astype(np.float32),
            eye_lid_y=eye_lid_y.astype(np.float32),
            eyeX=eyeX.astype(np.float32),
            eyeY=eyeY.astype(np.float32),
            pupilX=pupilX.astype(np.float32),
            pupilY=pupilY.astype(np.float32),
            source_signature=current_sig,
        )
        payload = {
            "cache_version": self.source_version,
            "source_signature": current_sig,
            "t": bundle.t.astype(np.float64),
            "frame_observed": bundle.frame_observed.astype(bool),
            "radius": bundle.radius.astype(np.float32),
            "x": bundle.x.astype(np.float32),
            "y": bundle.y.astype(np.float32),
            "velocity": bundle.velocity.astype(np.float32),
            "qc": bundle.qc.astype(np.float32),
            "in_eye": bundle.in_eye.astype(bool),
            "brake_raw": bundle.brake_raw.astype(np.float32),
            "brake_t": bundle.brake_t.astype(np.float64),
            "wheel_pos": bundle.wheel_pos.astype(np.float32),
            "locomotion": bundle.locomotion.astype(np.float32),
            "locomotion_t": bundle.locomotion_t.astype(np.float32),
            "eye_lid_x": bundle.eye_lid_x.astype(np.float32),
            "eye_lid_y": bundle.eye_lid_y.astype(np.float32),
            "eyeX": bundle.eyeX.astype(np.float32),
            "eyeY": bundle.eyeY.astype(np.float32),
            "pupilX": bundle.pupilX.astype(np.float32),
            "pupilY": bundle.pupilY.astype(np.float32),
        }
        _save_pickle(self.base_cache_path(summary.exp_id), payload)
        return bundle

    def _load_existing_wheel_trace(self, animal_id: str, exp_id: str):
        wheel_path = resolve_wheel_pickle(self.source_root, self.remote_root, animal_id, exp_id)
        if wheel_path is None or not wheel_path.exists():
            return None
        try:
            with wheel_path.open("rb") as f:
                wheel = pickle.load(f)
        except Exception:
            return None
        if not isinstance(wheel, dict):
            return None

        wheel_t = np.asarray(wheel.get("t", []), dtype=float).reshape(-1)
        wheel_pos = wheel.get("position_smoothed", wheel.get("position", []))
        wheel_pos = np.asarray(wheel_pos, dtype=float).reshape(-1)
        wheel_speed = np.asarray(wheel.get("speed", []), dtype=float).reshape(-1)
        if wheel_speed.size == 0 and wheel_pos.size >= 2:
            wheel_speed = np.diff(wheel_pos)
            wheel_speed = np.append(wheel_speed, wheel_speed[-1] if wheel_speed.size else 0.0)
            if wheel_t.size >= 2:
                dt = np.diff(wheel_t)
                finite_dt = dt[np.isfinite(dt) & (dt > 0)]
                if finite_dt.size:
                    wheel_speed = wheel_speed / float(np.nanmedian(finite_dt))
                else:
                    wheel_speed = wheel_speed * 20.0
            else:
                wheel_speed = wheel_speed * 20.0
        if wheel_t.size == 0 and wheel_speed.size:
            wheel_t = np.arange(wheel_speed.size, dtype=float) / 20.0
        if wheel_pos.size == 0 and wheel_speed.size:
            wheel_pos = np.zeros_like(wheel_speed, dtype=float)
        n = min(wheel_t.size, wheel_pos.size, wheel_speed.size)
        if n <= 0:
            return None
        brake_raw = np.zeros(n, dtype=float)
        return wheel_t[:n].astype(float), brake_raw, wheel_pos[:n].astype(float), wheel_speed[:n].astype(float)

    def _load_raw_locomotion_csv(self, animal_id: str, exp_id: str):
        csv_path = resolve_locomotion_csv(self.remote_root, animal_id, exp_id)
        if csv_path is None or not csv_path.exists():
            raise FileNotFoundError(f"Missing locomotion CSV for {exp_id}")
        df = pd.read_csv(csv_path)
        if "timestamp" not in df.columns or "arduino_data" not in df.columns:
            raise ValueError(f"Unexpected locomotion CSV format: {csv_path}")
        timestamps = np.asarray(pd.to_numeric(df["timestamp"], errors="coerce"), dtype=float)
        brake_raw = np.zeros(len(df), dtype=float)
        wheel_pos_raw = np.zeros(len(df), dtype=float)
        for i, value in enumerate(df["arduino_data"].astype(str).to_list()):
            parts = [p.strip() for p in value.split(";")]
            if len(parts) >= 1:
                try:
                    brake_raw[i] = float(parts[0])
                except Exception:
                    brake_raw[i] = np.nan
            if len(parts) >= 2:
                try:
                    wheel_pos_raw[i] = float(parts[1])
                except Exception:
                    wheel_pos_raw[i] = np.nan
        valid = np.isfinite(timestamps) & np.isfinite(wheel_pos_raw)
        if not np.any(valid):
            return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
        timestamps = timestamps[valid]
        brake_raw = brake_raw[valid]
        wheel_pos_raw = wheel_pos_raw[valid]
        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        brake_raw = brake_raw[order]
        wheel_pos_raw = wheel_pos_raw[order]
        return timestamps.astype(float), brake_raw.astype(float), wheel_pos_raw.astype(float)

    def _load_lock_trace(self, animal_id: str, exp_id: str) -> tuple[np.ndarray, np.ndarray]:
        timestamps, brake_raw, _ = self._load_raw_locomotion_csv(animal_id, exp_id)
        return timestamps.astype(float), brake_raw.astype(float)

    def _load_locomotion_trace(self, animal_id: str, exp_id: str):
        timestamps, brake_raw, wheel_pos_raw = self._load_raw_locomotion_csv(animal_id, exp_id)
        if timestamps.size == 0:
            return np.array([], dtype=float), brake_raw[:0], np.array([], dtype=float), np.array([], dtype=float)

        wheel_pos = np.asarray(wheel_pos_raw, dtype=float)
        if wheel_pos.size >= 2:
            wheel_pos_dif = np.diff(wheel_pos)
            wheel_pos_dif[wheel_pos_dif > 50000] -= 2**16
            wheel_pos_dif[wheel_pos_dif < -50000] += 2**16
            wheel_pos = np.cumsum(wheel_pos_dif)
            wheel_pos = np.append(wheel_pos, wheel_pos[-1])
        elif wheel_pos.size == 1:
            wheel_pos = np.asarray([float(wheel_pos[0])], dtype=float)
        else:
            wheel_pos = np.array([], dtype=float)

        resample_freq = 20.0
        wheel_linear_timescale = np.arange(0, np.floor(float(timestamps[-1])), 1.0 / resample_freq)
        if wheel_linear_timescale.size == 0:
            return wheel_linear_timescale.astype(float), brake_raw.astype(float), np.array([], dtype=float), np.array([], dtype=float)

        if wheel_pos.size >= 2:
            wheel_resampler = interpolate.interp1d(
                timestamps,
                wheel_pos,
                kind="linear",
                fill_value=(float(wheel_pos[0]), float(wheel_pos[-1])),
                bounds_error=False,
            )
            wheel_pos_resampled = np.asarray(wheel_resampler(wheel_linear_timescale), dtype=float)
        elif wheel_pos.size == 1:
            wheel_pos_resampled = np.full(wheel_linear_timescale.shape, float(wheel_pos[0]), dtype=float)
        else:
            wheel_pos_resampled = np.zeros_like(wheel_linear_timescale, dtype=float)

        smooth_window = 10
        if wheel_pos_resampled.size >= smooth_window:
            wheel_pos_smooth = np.convolve(wheel_pos_resampled, np.ones(smooth_window) / smooth_window, mode="same")
            wheel_pos_smooth[0:smooth_window] = wheel_pos_resampled[0]
            wheel_pos_smooth[-smooth_window:] = wheel_pos_resampled[-1]
        else:
            wheel_pos_smooth = wheel_pos_resampled.astype(float).copy()
        if wheel_pos_smooth.size:
            wheel_velocity = np.diff(wheel_pos_smooth)
            wheel_velocity = np.append(wheel_velocity, wheel_velocity[-1] if wheel_velocity.size else 0.0)
        else:
            wheel_velocity = np.array([], dtype=float)
        wheel_velocity = wheel_velocity * resample_freq
        return wheel_linear_timescale.astype(float), brake_raw.astype(float), wheel_pos_resampled.astype(float), wheel_velocity.astype(float)
    def load_face_motion(self, exp_id: str, *, force_recompute: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
        summary = self.get_session_summary(exp_id)
        if summary is None:
            raise FileNotFoundError(f"Unknown expID: {exp_id}")
        if not summary.has_right_video or summary.right_video is None or not Path(summary.right_video).exists():
            return None, None
        cache_path = self.face_motion_cache_path(exp_id)
        current_sig = self._current_source_signature(summary)
        if not force_recompute and cache_path.exists():
            try:
                cached = _load_pickle(cache_path)
                if (
                    isinstance(cached, dict)
                    and cached.get("cache_version") == self.source_version
                    and self._signature_matches(cached.get("source_signature", {}), current_sig)
                ):
                    return (
                        np.asarray(cached["face_motion_t"], dtype=float),
                        np.asarray(cached["face_motion"], dtype=float),
                    )
            except Exception:
                pass
        t, motion = self._compute_face_motion(summary)
        if t is None or motion is None:
            return None, None
        payload = {
            "cache_version": self.source_version,
            "source_signature": current_sig,
            "face_motion_t": np.asarray(t, dtype=np.float64),
            "face_motion": np.asarray(motion, dtype=np.float32),
        }
        _save_pickle(cache_path, payload)
        return np.asarray(t, dtype=float), np.asarray(motion, dtype=float)

    def _compute_face_motion(self, summary: SessionSummary):
        if summary.right_video is None or not Path(summary.right_video).exists():
            return None, None
        bundle = self.load_session_bundle(summary.exp_id)
        cap = cv2.VideoCapture(summary.right_video)
        if not cap.isOpened():
            return None, None
        observed = _frame_observed_mask(bundle)
        t = np.asarray(bundle.t, dtype=float).reshape(-1)
        motion = np.full(t.shape, np.nan, dtype=float)
        prev_gray = None
        any_frame = False
        for frame_idx in range(t.size):
            if not observed[frame_idx]:
                continue
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                motion[frame_idx] = 0.0
            else:
                motion[frame_idx] = float(np.sum(cv2.absdiff(gray, prev_gray)))
            prev_gray = gray
            any_frame = True
        cap.release()
        if not any_frame:
            return None, None
        return t, motion

    def load_pupil_brightness(self, exp_id: str, *, force_recompute: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
        summary = self.get_session_summary(exp_id)
        if summary is None:
            raise FileNotFoundError(f"Unknown expID: {exp_id}")
        if not summary.has_right_video or summary.right_video is None or not Path(summary.right_video).exists():
            return None, None
        cache_path = self.pupil_brightness_cache_path(exp_id)
        current_sig = self._current_source_signature(summary)
        if not force_recompute and cache_path.exists():
            try:
                cached = _load_pickle(cache_path)
                if (
                    isinstance(cached, dict)
                    and cached.get("cache_version") == self.source_version
                    and self._signature_matches(cached.get("source_signature", {}), current_sig)
                ):
                    return (
                        np.asarray(cached["pupil_brightness_t"], dtype=float),
                        np.asarray(cached["pupil_brightness"], dtype=float),
                    )
            except Exception:
                pass
        t, brightness = self._compute_pupil_brightness(summary)
        if t is None or brightness is None:
            return None, None
        payload = {
            "cache_version": self.source_version,
            "source_signature": current_sig,
            "pupil_brightness_t": np.asarray(t, dtype=np.float64),
            "pupil_brightness": np.asarray(brightness, dtype=np.float32),
        }
        _save_pickle(cache_path, payload)
        return np.asarray(t, dtype=float), np.asarray(brightness, dtype=float)

    def load_eye_similarity(
        self,
        exp_id: str,
        band: tuple[float, float],
        *,
        force_recompute: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        summary = self.get_session_summary(exp_id)
        if summary is None:
            raise FileNotFoundError(f"Unknown expID: {exp_id}")
        if not summary.has_right_video or summary.right_video is None or not Path(summary.right_video).exists():
            return None, None
        low, high = float(band[0]), float(band[1])
        if high < low:
            low, high = high, low
        cache_path = self.eye_similarity_cache_path(exp_id, (low, high))
        current_sig = self._current_source_signature(summary)
        band_sig = _band_signature((low, high))
        if not force_recompute and cache_path.exists():
            try:
                cached = _load_pickle(cache_path)
                if (
                    isinstance(cached, dict)
                    and cached.get("cache_version") == EYE_SIMILARITY_CACHE_VERSION
                    and cached.get("band_signature") == band_sig
                    and self._signature_matches(cached.get("source_signature", {}), current_sig)
                ):
                    return (
                        np.asarray(cached["eye_similarity_t"], dtype=float),
                        np.asarray(cached["eye_similarity"], dtype=float),
                    )
            except Exception:
                pass
        t, similarity = self._compute_eye_similarity(summary, (low, high))
        if t is None or similarity is None:
            return None, None
        payload = {
            "cache_version": EYE_SIMILARITY_CACHE_VERSION,
            "source_signature": current_sig,
            "band_signature": band_sig,
            "reference_band": [float(low), float(high)],
            "eye_similarity_t": np.asarray(t, dtype=np.float64),
            "eye_similarity": np.asarray(similarity, dtype=np.float32),
        }
        _save_pickle(cache_path, payload)
        return np.asarray(t, dtype=float), np.asarray(similarity, dtype=float)

    def _compute_eye_similarity(self, summary: SessionSummary, band: tuple[float, float]):
        if summary.right_video is None or not Path(summary.right_video).exists():
            return None, None
        bundle = self.load_session_bundle(summary.exp_id)
        cap = cv2.VideoCapture(summary.right_video)
        if not cap.isOpened():
            return None, None
        observed = _frame_observed_mask(bundle)
        t = np.asarray(bundle.t, dtype=float).reshape(-1)
        similarity = np.full(t.shape, np.nan, dtype=float)
        low, high = float(band[0]), float(band[1])
        if high < low:
            low, high = high, low
        any_frame = False
        for frame_idx in range(t.size):
            if not observed[frame_idx]:
                continue
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(gray.shape[:2], dtype=np.uint8)
            if frame_idx < bundle.eye_lid_x.size and frame_idx < bundle.eye_lid_y.size:
                eye_x = np.asarray(bundle.eye_lid_x[frame_idx], dtype=float).reshape(-1)
                eye_y = np.asarray(bundle.eye_lid_y[frame_idx], dtype=float).reshape(-1)
                pts = []
                for x, y in zip(eye_x[: min(eye_x.size, eye_y.size)].ravel(), eye_y[: min(eye_x.size, eye_y.size)].ravel()):
                    if np.isfinite(x) and np.isfinite(y):
                        pts.append((int(round(x)), int(round(y))))
                if len(pts) >= 3:
                    poly = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [poly], 1)
                    if frame_idx < bundle.pupilX.size and frame_idx < bundle.pupilY.size:
                        pupil_x = np.asarray(bundle.pupilX[frame_idx], dtype=float).reshape(-1)
                        pupil_y = np.asarray(bundle.pupilY[frame_idx], dtype=float).reshape(-1)
                        pupil_pts = []
                        for x, y in zip(pupil_x[: min(pupil_x.size, pupil_y.size)].ravel(), pupil_y[: min(pupil_x.size, pupil_y.size)].ravel()):
                            if np.isfinite(x) and np.isfinite(y):
                                pupil_pts.append((int(round(x)), int(round(y))))
                        if len(pupil_pts) >= 3:
                            pupil_poly = np.asarray(pupil_pts, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(mask, [pupil_poly], 0)
            value = float("nan")
            if np.any(mask):
                region = gray[mask.astype(bool)]
                if region.size:
                    value = float(np.mean((region >= low) & (region <= high)))
            similarity[frame_idx] = value
            any_frame = True
        cap.release()
        if not any_frame:
            return None, None
        return t, similarity

    def _compute_pupil_brightness(self, summary: SessionSummary):
        if summary.right_video is None or not Path(summary.right_video).exists():
            return None, None
        bundle = self.load_session_bundle(summary.exp_id)
        cap = cv2.VideoCapture(summary.right_video)
        if not cap.isOpened():
            return None, None
        observed = _frame_observed_mask(bundle)
        t = np.asarray(bundle.t, dtype=float).reshape(-1)
        brightness = np.full(t.shape, np.nan, dtype=float)
        any_frame = False
        for frame_idx in range(t.size):
            if not observed[frame_idx]:
                continue
            ok, frame = cap.read()
            if not ok:
                break
            value = float("nan")
            if frame_idx < bundle.radius.size and frame_idx < bundle.x.size and frame_idx < bundle.y.size:
                x = float(bundle.x[frame_idx])
                y = float(bundle.y[frame_idx])
                radius = float(bundle.radius[frame_idx])
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(radius) and radius > 0.0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    h, w = gray.shape[:2]
                    cx = float(np.clip(x, 0.0, max(0.0, float(w - 1))))
                    cy = float(np.clip(y, 0.0, max(0.0, float(h - 1))))
                    r = max(1.0, float(radius))
                    x0 = max(0, int(np.floor(cx - r)))
                    x1 = min(w, int(np.ceil(cx + r)) + 1)
                    y0 = max(0, int(np.floor(cy - r)))
                    y1 = min(h, int(np.ceil(cy + r)) + 1)
                    if x1 > x0 and y1 > y0:
                        yy, xx = np.ogrid[y0:y1, x0:x1]
                        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
                        region = gray[y0:y1, x0:x1]
                        if np.any(mask):
                            value = float(np.nanmean(region[mask]))
            brightness[frame_idx] = value
            any_frame = True
        cap.release()
        if not any_frame:
            return None, None
        return t, brightness

    def _migrate_manual_masks_to_session_timebase(
        self,
        exp_id: str,
        masks: list[tuple[float, float]],
        *,
        summary: SessionSummary | None = None,
    ) -> list[tuple[float, float]] | None:
        summary = summary or self.get_session_summary(exp_id)
        if summary is None or not summary.has_right_video:
            return None
        video_duration = float(summary.video_duration_sec or 0.0)
        if video_duration <= 0.0:
            return None
        try:
            bundle = self.load_session_bundle(exp_id)
        except Exception:
            bundle = None
        if bundle is not None and bundle.t.size:
            new_start = float(bundle.t[0])
            new_end = float(bundle.t[-1])
        else:
            new_start = 0.0
            new_end = float(summary.duration_sec)
        if not np.isfinite(new_end) or new_end <= new_start:
            return None
        span = float(new_end - new_start)
        scale = span / video_duration
        migrated = []
        for start, end in masks:
            if end <= start:
                continue
            migrated.append((float(new_start + start * scale), float(new_start + end * scale)))
        return migrated

    def load_manual_masks(self, exp_id: str) -> list[tuple[float, float]]:
        state = self.load_session_state(exp_id)
        masks = []
        for item in state.get("manual_masks", []):
            try:
                start, end = float(item[0]), float(item[1])
                if end > start:
                    masks.append((start, end))
            except Exception:
                continue
        timebase = state.get("manual_masks_timebase")
        if masks and timebase != "session" and bool(state.get("timebase_aligned", False)):
            migrated = self._migrate_manual_masks_to_session_timebase(exp_id, masks)
            if migrated is not None:
                state["manual_masks"] = [[float(s), float(e)] for s, e in migrated]
                state["manual_masks_timebase"] = "session"
                state["stats_dirty"] = True
                state["last_stats_signature"] = ""
                self.save_session_state(exp_id, state)
                self.mark_stats_dirty()
                return migrated
        return masks

    def save_manual_masks(self, exp_id: str, intervals: list[tuple[float, float]], *, mark_dirty: bool = True) -> None:
        state = self.load_session_state(exp_id)
        state["manual_masks"] = [[float(s), float(e)] for s, e in sorted(intervals)]
        state["manual_masks_timebase"] = "session"
        if mark_dirty:
            state["stats_dirty"] = True
            state["last_stats_signature"] = ""
        self.save_session_state(exp_id, state)
        self.mark_stats_dirty()

    def mark_stats_dirty(self) -> None:
        self.settings.setdefault("global", {})
        self.settings["global"]["stats_dirty"] = True
        self.save_settings()

    def clear_stats_dirty(self) -> None:
        self.settings.setdefault("global", {})
        self.settings["global"]["stats_dirty"] = False
        self.save_settings()

    def is_stats_dirty(self) -> bool:
        return bool(self.settings.get("global", {}).get("stats_dirty", True))

    def is_animal_dirty(self, animal_id: str) -> bool:
        data = self.get_animal_settings(animal_id)
        threshold_dirty = False
        if animal_id != "All":
            threshold_dirty = str(data.get("threshold_signature", "")) != self.pupil_percentile_signature()
        return bool(data.get("stats_dirty", True) or data.get("baseline_dirty", True) or threshold_dirty)

    def clear_animal_dirty(self, animal_id: str) -> None:
        data = self.get_animal_settings(animal_id)
        data["stats_dirty"] = False
        data["baseline_dirty"] = False
        self.set_animal_settings(animal_id, data)
        if not any(self.is_animal_dirty(a) for a in self.animals()):
            self.clear_stats_dirty()

    def mark_all_animals_stats_dirty(self) -> None:
        self.settings.setdefault("animals", {})
        for animal_id, data in list(self.settings["animals"].items()):
            if not isinstance(data, dict):
                data = {}
            data["stats_dirty"] = True
            self.settings["animals"][animal_id] = data
        self.settings.setdefault("global", {})
        self.settings["global"]["stats_dirty"] = True
        self.save_settings()

    def set_animal_settings(self, animal_id: str, data: dict) -> None:
        self.settings.setdefault("animals", {})
        self.settings["animals"][animal_id] = data
        self.save_settings()

    def get_animal_settings(self, animal_id: str) -> dict:
        self.settings.setdefault("animals", {})
        data = self.settings["animals"].get(animal_id, {})
        if not isinstance(data, dict):
            data = {}
        data.setdefault("threshold_signature", "")
        data.setdefault("threshold_values", [0.0, 0.5, 1.0])
        data.setdefault("zscore_mean", None)
        data.setdefault("zscore_std", None)
        data.setdefault("baseline_dirty", True)
        data.setdefault("stats_dirty", True)
        return data

    def update_animal_settings(self, animal_id: str, **kwargs) -> dict:
        data = self.get_animal_settings(animal_id)
        data.update(kwargs)
        self.set_animal_settings(animal_id, data)
        return data

    def invalidate_animal(self, animal_id: str) -> None:
        data = self.get_animal_settings(animal_id)
        data["baseline_dirty"] = True
        data["stats_dirty"] = True
        self.set_animal_settings(animal_id, data)
        self.mark_stats_dirty()

    def dataset_sessions(self) -> list[SessionSummary]:
        if self.index is None:
            self.load_index()
        assert self.index is not None
        return list(self.index.sessions)

    def animals(self) -> list[str]:
        animals = sorted({s.animal_id for s in self.dataset_sessions()})
        return animals

    def sessions_for_animal(self, animal_id: str) -> list[SessionSummary]:
        sessions = [s for s in self.dataset_sessions() if s.animal_id == animal_id]
        return sorted(sessions, key=lambda s: s.sort_key)

    def update_dataset(
        self,
        progress_cb: Callable[[float, str], None] | None = None,
        *,
        refresh_reference_sessions: bool = False,
    ) -> DatasetIndex:
        if progress_cb:
            progress_cb(0.0, "Scanning source tree")
        index = self.refresh_index()
        if refresh_reference_sessions:
            references = self.deeplabcut_reference_sessions()
            total = max(1, len(references))
            for idx, summary in enumerate(references):
                if progress_cb:
                    progress_cb(0.5 + 0.49 * (idx / total), f"Refreshing DeeplabCut reference cache {summary.exp_id}")
                try:
                    self.load_session_bundle(summary.exp_id, force_rebuild=True)
                except Exception:
                    pass
                try:
                    self.load_face_motion(summary.exp_id, force_recompute=True)
                except Exception:
                    pass
        if progress_cb:
            progress_cb(1.0, "Dataset index refreshed")
        self.mark_stats_dirty()
        return index

    def stats_signature(self, scope: str, animal_id: str, thresholds: dict, masks_hash: str, zscore_mean: float, zscore_std: float) -> str:
        payload = {
            "scope": scope,
            "animal_id": animal_id,
            "thresholds": thresholds,
            "masks_hash": masks_hash,
            "zscore_mean": float(zscore_mean),
            "zscore_std": float(zscore_std),
            "cache_version": self.source_version,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


    def statistics_cache_signature(self, scope: str, animal_id: str, thresholds: dict) -> str:
        if scope == "All" or animal_id == "All":
            sessions = self.dataset_sessions()
        else:
            sessions = self.sessions_for_animal(animal_id)
        analysis_sessions = [
            summary
            for summary in sessions
            if summary.has_right_pickle and not self.is_deeplabcut_reference_session(summary.exp_id) and not self.is_session_do_not_use(summary.exp_id)
        ]
        payload = {
            "scope": scope,
            "animal_id": animal_id,
            "thresholds": thresholds,
            "sessions": [
                {
                    "exp_id": summary.exp_id,
                    "source_signature": self._current_source_signature(summary),
                    "manual_masks_hash": self.session_masks_hash(summary.exp_id),
                    "analysis_cutoff_sec": self.session_analysis_cutoff_sec(summary.exp_id),
                    "do_not_use": self.is_session_do_not_use(summary.exp_id),
                    "deeplabcut_reference": self.is_deeplabcut_reference_session(summary.exp_id),
                    "extra_large_calibration": self.session_extra_large_calibration(summary.exp_id),
                }
                for summary in analysis_sessions
            ],
            "cache_version": self.source_version,
            "statistics_results_version": STATISTICS_RESULTS_VERSION,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def session_masks_hash(self, exp_id: str) -> str:
        masks = self.load_manual_masks(exp_id)
        payload = json.dumps(masks, sort_keys=True).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

