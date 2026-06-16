from __future__ import annotations
import json
import os
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from .data import HabituationStore, SessionSummary, analysis_cutoff_mask, apply_time_mask
from .plotting import style_axes
from .stats import (
    STATE_LABELS,
    animal_zscores,
    build_extra_large_mask,
    compute_animal_baseline,
    compute_statistics,
    learn_extra_large_calibration,
    learn_extra_large_reference_band,
    load_cached_statistics_outputs,
    percentile_from_value,
    percentile_threshold_values,
    save_statistics_outputs,
    statistics_panel_paths,
    suggest_extra_large_calibration,
)
from .widgets import DraggableHLine, TracePanZoomCanvas, VideoPlayerWidget
BOUNDARY_NAMES = ["small/medium", "medium/large", "large/extra-large"]
BOUNDARY_COLORS = ["tab:red", "tab:orange", "tab:green"]
@dataclass
class LoadedSession:
    summary: SessionSummary
    t: np.ndarray
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
    face_motion: np.ndarray
    face_motion_t: np.ndarray
    eye_lid_x: np.ndarray
    eye_lid_y: np.ndarray
    eyeX: np.ndarray
    eyeY: np.ndarray
    pupilX: np.ndarray
    pupilY: np.ndarray
    source_signature: dict
    has_pupil: bool
    @property
    def visible_mask_base(self) -> np.ndarray:
        if self.radius.size == 0:
            return np.zeros(0, dtype=bool)
        visible = np.isfinite(self.radius)
        if self.in_eye.ndim == 2 and self.in_eye.shape[0] == self.radius.shape[0]:
            visible = visible & np.all(self.in_eye.astype(bool), axis=1)
        if self.qc.shape == self.radius.shape:
            visible = visible & (self.qc == 0)
        return visible
    def video_overlay(self) -> dict[str, np.ndarray]:
        return {
            'x': self.x,
            'y': self.y,
            'radius': self.radius,
            'eye_lid_x': self.eye_lid_x,
            'eye_lid_y': self.eye_lid_y,
            'eyeX': self.eyeX,
            'eyeY': self.eyeY,
            'pupilX': self.pupilX,
            'pupilY': self.pupilY,
        }
class TaskThread(QtCore.QThread):
    progress = pyqtSignal(float, str)
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn
    def run(self):
        try:
            result = self._fn(self.progress.emit)
        except Exception:
            self.error.emit(traceback.format_exc())
            return
        self.result_ready.emit(result)
class ScrollableComboBox(QComboBox):
    def __init__(self, parent=None, *, visible_rows: int = 4):
        super().__init__(parent)
        self._visible_rows = max(1, int(visible_rows))
    def showPopup(self):
        super().showPopup()
        QtCore.QTimer.singleShot(0, self._fit_popup_to_window)
    def _fit_popup_to_window(self):
        view = self.view()
        popup = view.window()
        if popup is None or self.count() <= 0:
            return
        rows = min(self._visible_rows, self.count())
        row_height = max(view.sizeHintForRow(0), self.fontMetrics().height() + 8)
        frame = max(0, popup.frameGeometry().height() - view.geometry().height())
        desired_height = rows * row_height + frame + 6
        desired_width = max(self.width(), view.sizeHintForColumn(0) + 32)
        window = self.window()
        if window is not None:
            bounds = window.frameGeometry()
        else:
            screen = QtGui.QGuiApplication.screenAt(self.mapToGlobal(QtCore.QPoint(0, self.height())))
            if screen is None:
                screen = QtGui.QGuiApplication.primaryScreen()
            bounds = screen.availableGeometry() if screen is not None else QtCore.QRect()
        combo_top_left = self.mapToGlobal(QtCore.QPoint(0, 0))
        combo_bottom_left = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        space_below = max(0, bounds.bottom() - combo_bottom_left.y())
        space_above = max(0, combo_top_left.y() - bounds.top())
        height = min(desired_height, max(space_below, space_above))
        if height <= 0:
            return
        if space_below >= space_above:
            y = combo_bottom_left.y()
        else:
            y = combo_top_left.y() - height
        width = min(desired_width, max(1, bounds.width()))
        x = combo_top_left.x()
        max_x = bounds.right() - width + 1
        if x > max_x:
            x = max(bounds.left(), max_x)
        if x < bounds.left():
            x = bounds.left()
        if y + height > bounds.bottom() + 1:
            y = max(bounds.top(), bounds.bottom() - height + 1)
        if y < bounds.top():
            y = bounds.top()
        popup.setGeometry(x, y, width, height)
        popup.raise_()
class TraceCanvas(TracePanZoomCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
class LoadingWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.Window | Qt.FramelessWindowHint | Qt.Dialog | Qt.WindowStaysOnTopHint,
        )
        self.setObjectName("LoadingWindow")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowTitle("Habituation Analysis")
        self.setMinimumSize(460, 220)
        self.setStyleSheet(
            """
            QWidget#LoadingWindow {
                background: #1f232a;
                border: 1px solid #48515d;
                border-radius: 14px;
                color: #f5f7fa;
            }
            QLabel {
                color: #f5f7fa;
            }
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(10)
        title = QLabel("Habituation Analysis", self)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        self.message_label = QLabel("Loading cached dataset and restoring the last browser state...", self)
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("font-size: 13px;")
        self.detail_label = QLabel("", self)
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("color: #c7ced8; font-size: 11px;")
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(self.message_label)
        layout.addWidget(self.detail_label)
        layout.addStretch(1)
    def set_message(self, message: str, detail: str = ""):
        self.message_label.setText(message)
        self.detail_label.setText(detail)
        self.adjustSize()
def format_seconds(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    total = int(round(float(value)))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s))
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue
        merged[-1][1] = max(merged[-1][1], end)
    return [(float(s), float(e)) for s, e in merged]
def mask_to_intervals(times: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    times = np.asarray(times, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if times.size == 0 or mask.size == 0:
        return []
    n = min(times.size, mask.size)
    times = times[:n]
    mask = mask[:n]
    intervals: list[tuple[float, float]] = []
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return intervals
    split_points = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, split_points)
    if times.size > 1:
        step = float(np.nanmedian(np.diff(times[np.isfinite(times)]))) if np.sum(np.isfinite(times)) > 1 else 0.0
    else:
        step = 0.0
    if not np.isfinite(step) or step < 0.0:
        step = 0.0
    for segment in segments:
        if segment.size == 0:
            continue
        start = float(times[segment[0]])
        end = float(times[segment[-1]] + step)
        intervals.append((start, end))
    return merge_intervals(intervals)
class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, parent=None, *, expanded: bool = False):
        super().__init__(parent)
        self._content = content
        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.toggled.connect(self._on_toggled)
        self._content.setVisible(expanded)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._toggle)
        layout.addWidget(self._content)
    def _on_toggled(self, checked: bool):
        self._content.setVisible(bool(checked))
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
class ExperimentIndexTab(QWidget):
    sessionActivated = pyqtSignal(str, str)
    def __init__(self, store: HabituationStore, parent=None):
        super().__init__(parent)
        self.store = store
        self._selected_animal = "All"
        self._selected_exp_id = ""
        self._current_signature = self.store.pupil_percentile_signature()
        self._hint = QLabel("Double-click an expID to open it in the session browser.", self)
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #666666; font-size: 11px;")
        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Experiment", "Status"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tree.setUniformRowHeights(True)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout = QVBoxLayout(self)
        layout.addWidget(self._hint)
        layout.addWidget(self.tree, stretch=1)
        self.refresh()
    def refresh(self):
        self._current_signature = self.store.pupil_percentile_signature()
        previous_animal = self._selected_animal
        previous_exp_id = self._selected_exp_id
        self.tree.clear()
        animals = self.store.animals()
        for animal in animals:
            sessions = self.store.sessions_for_animal(animal)
            if not sessions:
                continue
            ready_count = 0
            stale_count = 0
            reference_count = 0
            do_not_use_count = 0
            considered_count = 0
            parent = QTreeWidgetItem([f"{animal}", ""])
            parent_font = parent.font(0)
            parent_font.setBold(True)
            parent.setFont(0, parent_font)
            parent.setFont(1, parent_font)
            parent.setData(0, Qt.UserRole, {"kind": "animal", "animal_id": animal})
            parent.setToolTip(0, f"{animal} sessions")
            for summary in sessions:
                state = self.store.load_session_state(summary.exp_id)
                preprocessed = bool(state.get("preprocessed", False))
                reference = bool(state.get("deeplabcut_reference", False))
                do_not_use = bool(state.get("do_not_use", False))
                stored_sig = str(state.get("threshold_signature", ""))
                checked = preprocessed and stored_sig == self._current_signature
                if reference:
                    status = "Add for DLC model"
                    reference_count += 1
                    color = QtGui.QColor("#5c6bc0")
                elif do_not_use:
                    status = "Do not use"
                    do_not_use_count += 1
                    color = QtGui.QColor("#b00020")
                else:
                    considered_count += 1
                    if checked:
                        status = "Pre-processed"
                        ready_count += 1
                        color = QtGui.QColor("#1b5e20")
                    elif preprocessed:
                        status = "Marked, thresholds changed"
                        stale_count += 1
                        color = QtGui.QColor("#b26a00")
                    else:
                        status = "Not pre-processed"
                        color = QtGui.QColor("#666666")
                child = QTreeWidgetItem([summary.exp_id, status])
                child.setData(0, Qt.UserRole, {"kind": "session", "animal_id": animal, "exp_id": summary.exp_id})
                cutoff = self.store.session_analysis_cutoff_sec(summary.exp_id)
                usable_duration = self.store.effective_session_duration_sec(summary.exp_id)
                tooltip_lines = [
                    summary.exp_id,
                    f"Status: {status}",
                    f"Usable duration: {format_seconds(usable_duration)}",
                ]
                if cutoff is not None:
                    tooltip_lines.append(f"Cutoff: {format_seconds(cutoff)}")
                tooltip_text = "\n".join(tooltip_lines)
                child.setToolTip(0, tooltip_text)
                child.setToolTip(1, tooltip_text)
                if checked and not reference and not do_not_use:
                    font = child.font(0)
                    font.setBold(True)
                    child.setFont(0, font)
                    child.setFont(1, font)
                child.setForeground(0, QtGui.QBrush(color))
                child.setForeground(1, QtGui.QBrush(color))
                parent.addChild(child)
            parent.setText(0, f"{animal} ({ready_count}/{considered_count} pre-processed)")
            status_bits = []
            if stale_count:
                status_bits.append(f"{stale_count} stale")
            if do_not_use_count:
                status_bits.append(f"{do_not_use_count} do-not-use")
            if reference_count:
                status_bits.append(f"{reference_count} refs excluded")
            if status_bits:
                parent.setText(1, ", ".join(status_bits))
            self.tree.addTopLevelItem(parent)
            parent.setExpanded(animal == previous_animal)
        self.tree.resizeColumnToContents(0)
        self.tree.resizeColumnToContents(1)
        self._select_current_item(previous_animal, previous_exp_id)
    def set_current_selection(self, animal_id: str, exp_id: str):
        self._selected_animal = animal_id or "All"
        self._selected_exp_id = exp_id or ""
        if self.tree.topLevelItemCount():
            self._select_current_item(self._selected_animal, self._selected_exp_id)
    def _select_current_item(self, animal_id: str, exp_id: str):
        found = None
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            data = parent.data(0, Qt.UserRole) or {}
            if data.get("animal_id") != animal_id:
                continue
            parent.setExpanded(True)
            if not exp_id:
                found = parent
                break
            for j in range(parent.childCount()):
                child = parent.child(j)
                child_data = child.data(0, Qt.UserRole) or {}
                if child_data.get("exp_id") == exp_id:
                    found = child
                    break
            if found is not None:
                break
        if found is not None:
            self.tree.setCurrentItem(found)
            self.tree.scrollToItem(found)
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        data = item.data(0, Qt.UserRole) or {}
        if data.get("kind") == "animal":
            item.setExpanded(not item.isExpanded())
            return
        if data.get("kind") != "session":
            return
        animal_id = str(data.get("animal_id", ""))
        exp_id = str(data.get("exp_id", ""))
        if animal_id and exp_id:
            self.sessionActivated.emit(animal_id, exp_id)
class DeeplabcutReferenceTab(QWidget):
    sessionActivated = pyqtSignal(str, str)
    def __init__(self, store: HabituationStore, parent=None):
        super().__init__(parent)
        self.store = store
        self._selected_animal = "All"
        self._selected_exp_id = ""
        self._copy_text = "No sessions for the DLC model found."
        self._hint = QLabel(
            "These expIDs are manually marked for the DLC model and are excluded from statistics.",
            self,
        )
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #666666; font-size: 11px;")
        self.copy_btn = QPushButton("Copy list", self)
        self.copy_btn.clicked.connect(self._copy_list)
        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Experiment", "Note"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tree.setUniformRowHeights(True)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout = QVBoxLayout(self)
        layout.addWidget(self._hint)
        layout.addWidget(self.copy_btn, alignment=Qt.AlignLeft)
        layout.addWidget(self.tree, stretch=1)
        self.refresh()
    def refresh(self):
        previous_animal = self._selected_animal
        previous_exp_id = self._selected_exp_id
        lines: list[str] = []
        self.tree.clear()
        for animal in self.store.animals():
            sessions = self.store.deeplabcut_reference_sessions(animal)
            if not sessions:
                continue
            parent = QTreeWidgetItem([animal, f"{len(sessions)} session{'s' if len(sessions) != 1 else ''}"])
            parent_font = parent.font(0)
            parent_font.setBold(True)
            parent.setFont(0, parent_font)
            parent.setFont(1, parent_font)
            parent.setData(0, Qt.UserRole, {"kind": "animal", "animal_id": animal})
            parent.setToolTip(0, f"{animal} sessions for the DLC model")
            lines.append(animal)
            for summary in sessions:
                child = QTreeWidgetItem([summary.exp_id, "Reference only"])
                child.setData(0, Qt.UserRole, {"kind": "session", "animal_id": animal, "exp_id": summary.exp_id})
                child.setToolTip(0, summary.exp_id)
                child.setToolTip(1, "Excluded from statistics")
                color = QtGui.QColor("#1b5e20")
                font = child.font(0)
                font.setBold(True)
                child.setFont(0, font)
                child.setFont(1, font)
                child.setForeground(0, QtGui.QBrush(color))
                child.setForeground(1, QtGui.QBrush(color))
                parent.addChild(child)
                lines.append(f"  {summary.exp_id}")
            parent.setExpanded(animal == previous_animal)
            self.tree.addTopLevelItem(parent)
        self._copy_text = "\n".join(lines) if lines else "No sessions for the DLC model found."
        self.tree.resizeColumnToContents(0)
        self.tree.resizeColumnToContents(1)
        self._select_current_item(previous_animal, previous_exp_id)
    def _copy_list(self):
        QApplication.clipboard().setText(self._copy_text)
    def _select_current_item(self, animal_id: str, exp_id: str):
        found = None
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            data = parent.data(0, Qt.UserRole) or {}
            if data.get("animal_id") != animal_id:
                continue
            parent.setExpanded(True)
            if not exp_id:
                found = parent
                break
            for j in range(parent.childCount()):
                child = parent.child(j)
                child_data = child.data(0, Qt.UserRole) or {}
                if child_data.get("exp_id") == exp_id:
                    found = child
                    break
            if found is not None:
                break
        if found is not None:
            self.tree.setCurrentItem(found)
            self.tree.scrollToItem(found)
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        data = item.data(0, Qt.UserRole) or {}
        if data.get("kind") == "animal":
            item.setExpanded(not item.isExpanded())
            return
        if data.get("kind") != "session":
            return
        animal_id = str(data.get("animal_id", ""))
        exp_id = str(data.get("exp_id", ""))
        if animal_id and exp_id:
            self._selected_animal = animal_id
            self._selected_exp_id = exp_id
            self.sessionActivated.emit(animal_id, exp_id)
class MetricsTab(QWidget):
    thresholds_changed = pyqtSignal()
    masks_changed = pyqtSignal()
    session_state_changed = pyqtSignal()
    def __init__(self, store: HabituationStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.animal_id = "All"
        self.exp_id = ""
        self.view_mode = "Overall"
        self._updating_controls = False
        self._threshold_lines: list[DraggableHLine] = []
        self._locomotion_threshold_line: DraggableHLine | None = None
        self._cursor_lines: list = []
        self._zscore_distribution = np.array([], dtype=float)
        self._scope_cache_key: str | None = None
        self._zscore_mean = 0.0
        self._zscore_std = 1.0
        self._percentile_cutoffs = self.store.global_pupil_percentile_cutoffs()
        self._threshold_values = [0.0, 0.5, 1.0]
        self._scope_sessions: list[SessionSummary] = []
        self._current_session: SessionSummary | None = None
        self._current_payload: LoadedSession | None = None
        self._pending_mask_start: float | None = None
        self._calibration_selection_start: float | None = None
        self._calibration_selection_end: float | None = None
        self._calibration_suggestion: dict | None = None
        self._reset_trace_view_pending = True
        self._dirty_label = QLabel("Statistics are clean")
        self._scope_label = QLabel("")
        self._scope_label.setWordWrap(True)
        self._baseline_label = QLabel("")
        self._baseline_label.setWordWrap(True)
        self._session_label = QLabel("")
        self._session_label.setWordWrap(True)
        self.timebase_warning = QLabel("")
        self.timebase_warning.setWordWrap(True)
        self.timebase_warning.setStyleSheet("color: #b26a00; font-weight: 600;")
        self.timebase_warning.setVisible(False)
        self.movie_locomotion_warning = QLabel("")
        self.movie_locomotion_warning.setWordWrap(True)
        self.movie_locomotion_warning.setStyleSheet("color: #7a4a00; font-weight: 600;")
        self.movie_locomotion_warning.setVisible(False)
        self.canvas = TraceCanvas(self)
        self.canvas.set_pan_blocker(self._should_block_trace_pan)
        self.canvas.pupil_ax.set_title("Pupil dynamics", pad=10)
        self.canvas.loc_ax.set_title("Locomotion", pad=10)
        self.zoom_in_btn = QPushButton("Zoom in", self)
        self.zoom_out_btn = QPushButton("Zoom out", self)
        self.reset_zoom_btn = QPushButton("Reset zoom", self)
        self.zoom_in_btn.clicked.connect(lambda: self.canvas.zoom(0.8))
        self.zoom_out_btn.clicked.connect(lambda: self.canvas.zoom(1.25))
        self.reset_zoom_btn.clicked.connect(self._reset_trace_zoom_view)
        self.video_widget = VideoPlayerWidget(self)
        self.video_widget.time_changed.connect(self._on_video_time_changed)
        self.video_time_label = QLabel("Current video time: --")
        self.video_time_label.setWordWrap(True)
        self.interval_list = QListWidget(self)
        self.start_btn = QPushButton("Set start", self)
        self.end_btn = QPushButton("Set end / add", self)
        self.delete_btn = QPushButton("Delete selected", self)
        self.clear_btn = QPushButton("Clear all", self)
        self.save_btn = QPushButton("Save intervals", self)
        self.start_btn.clicked.connect(self._set_interval_start)
        self.end_btn.clicked.connect(self._set_interval_end)
        self.delete_btn.clicked.connect(self._delete_selected_interval)
        self.clear_btn.clicked.connect(self._clear_intervals)
        self.save_btn.clicked.connect(self._save_intervals)
        self.calibration_group = QGroupBox("Extra-large calibration", self)
        calibration_layout = QVBoxLayout(self.calibration_group)
        self.calibration_status_label = QLabel("No calibration suggestion available yet.", self.calibration_group)
        self.calibration_status_label.setWordWrap(True)
        calibration_layout.addWidget(self.calibration_status_label)
        self.calibration_selection_label = QLabel("Selected period: none", self.calibration_group)
        self.calibration_selection_label.setWordWrap(True)
        self.calibration_selection_label.setStyleSheet("color: #444444; font-size: 11px;")
        calibration_layout.addWidget(self.calibration_selection_label)
        selection_row = QHBoxLayout()
        self.set_calibration_start_btn = QPushButton("Set calibration start", self.calibration_group)
        self.set_calibration_start_btn.clicked.connect(self._set_extra_large_calibration_start)
        self.set_calibration_end_btn = QPushButton("Set calibration end", self.calibration_group)
        self.set_calibration_end_btn.clicked.connect(self._set_extra_large_calibration_end)
        self.clear_calibration_selection_btn = QPushButton("Clear selection", self.calibration_group)
        self.clear_calibration_selection_btn.clicked.connect(self._clear_extra_large_calibration_selection)
        selection_row.addWidget(self.set_calibration_start_btn)
        selection_row.addWidget(self.set_calibration_end_btn)
        selection_row.addWidget(self.clear_calibration_selection_btn)
        calibration_layout.addLayout(selection_row)
        calibration_button_row = QHBoxLayout()
        self.goto_calibration_btn = QPushButton("Go to calibration", self.calibration_group)
        self.goto_calibration_btn.clicked.connect(self._go_to_extra_large_calibration)
        self.confirm_calibration_btn = QPushButton("Confirm calibration", self.calibration_group)
        self.confirm_calibration_btn.clicked.connect(self._confirm_extra_large_calibration)
        self.clear_calibration_btn = QPushButton("Clear calibration", self.calibration_group)
        self.clear_calibration_btn.clicked.connect(self._clear_extra_large_calibration)
        calibration_button_row.addWidget(self.goto_calibration_btn)
        calibration_button_row.addWidget(self.confirm_calibration_btn)
        calibration_button_row.addWidget(self.clear_calibration_btn)
        calibration_button_row.addStretch(1)
        calibration_layout.addLayout(calibration_button_row)
        calibration_hint = QLabel("The app suggests a calibration interval automatically from big detected pupil periods. You can also select a new calibration period from the video controls, then confirm it to activate the brightness-based extra-large rule.", self.calibration_group)
        calibration_hint.setWordWrap(True)
        calibration_hint.setStyleSheet("color: #666666; font-size: 11px;")
        calibration_layout.addWidget(calibration_hint)
        self.do_not_use_check = QCheckBox("Do not use", self)
        self.do_not_use_check.toggled.connect(self._on_do_not_use_toggled)
        self.do_not_use_check.setToolTip("Mark this expID as excluded from statistics.")
        self.preprocessed_check = QCheckBox("Pre-processed", self)
        self.preprocessed_check.toggled.connect(self._on_preprocessed_toggled)
        self.preprocessed_check.setToolTip("Mark this expID as pre-processed under the current shared pupil percentiles.")
        self.reference_check = QCheckBox("Add for DLC model", self)
        self.reference_check.toggled.connect(self._on_reference_toggled)
        self.reference_check.setToolTip("Mark this expID for DLC model use. This is independent of not-visible pupil intervals.")
        self.timebase_aligned_check = QCheckBox("Posteriorly aligned", self)
        self.timebase_aligned_check.toggled.connect(self._on_timebase_aligned_toggled)
        self.timebase_aligned_check.setToolTip(
            "Mark sessions whose timebase was aligned after acquisition. This changes the warning wording but does not hide it."
        )
        self.cutoff_btn = QPushButton("Set cutoff", self)
        self.cutoff_btn.clicked.connect(self._set_cutoff_from_current_time)
        self.clear_cutoff_btn = QPushButton("Clear cutoff", self)
        self.clear_cutoff_btn.clicked.connect(self._clear_cutoff)
        self.cutoff_label = QLabel("Cutoff: none", self)
        self.cutoff_label.setWordWrap(True)
        self.threshold_group = QGroupBox("Pupil thresholds (shared percentiles)", self)
        threshold_layout = QGridLayout(self.threshold_group)
        self.percentile_lock_check = QCheckBox("Unlock shared percentiles", self.threshold_group)
        self.percentile_lock_check.toggled.connect(self._on_percentile_lock_toggled)
        self.percentile_lock_check.setToolTip(
            "Enable direct typing in the shared percentile fields. Dragging threshold lines stays available either way."
        )
        threshold_layout.addWidget(self.percentile_lock_check, 0, 0, 1, 3)
        threshold_layout.addWidget(QLabel("Boundary"), 1, 0)
        threshold_layout.addWidget(QLabel("Shared percentile"), 1, 1)
        threshold_layout.addWidget(QLabel("Absolute z-score"), 1, 2)
        self.percentile_spins: list[QDoubleSpinBox] = []
        self.value_labels: list[QLabel] = []
        for idx, name in enumerate(BOUNDARY_NAMES):
            row = idx + 2
            threshold_layout.addWidget(QLabel(name), row, 0)
            pspin = QDoubleSpinBox(self.threshold_group)
            pspin.setRange(0.0, 100.0)
            pspin.setDecimals(1)
            pspin.setSingleStep(1.0)
            pspin.setKeyboardTracking(False)
            pspin.setEnabled(False)
            pspin.valueChanged.connect(lambda value, idx=idx: self._on_percentile_changed(idx, value))
            self.percentile_spins.append(pspin)
            threshold_layout.addWidget(pspin, row, 1)
            vlabel = QLabel("--", self.threshold_group)
            self.value_labels.append(vlabel)
            threshold_layout.addWidget(vlabel, row, 2)
        threshold_layout.addWidget(QLabel("Missing pupil buffer (s)", self.threshold_group), 5, 0)
        self.missing_buffer_spin = QDoubleSpinBox(self.threshold_group)
        self.missing_buffer_spin.setRange(0.0, 60.0)
        self.missing_buffer_spin.setDecimals(2)
        self.missing_buffer_spin.setSingleStep(0.1)
        self.missing_buffer_spin.setToolTip(
            "Gap length outside manual not-visible intervals before missing pupil detections count as extra-large."
        )
        self.missing_buffer_spin.valueChanged.connect(self._on_missing_buffer_changed)
        threshold_layout.addWidget(self.missing_buffer_spin, 5, 1)
        buffer_hint = QLabel("Sustained gaps only", self.threshold_group)
        buffer_hint.setStyleSheet("color: #666666; font-size: 11px;")
        threshold_layout.addWidget(buffer_hint, 5, 2)
        self.threshold_hint = QLabel("Unlock the shared percentile fields to type values. Dragging the threshold lines stays live.", self)
        self.threshold_hint.setWordWrap(True)
        self.threshold_hint.setStyleSheet("color: #666666; font-size: 11px;")
        self.locomotion_group = QGroupBox("Locomotion", self)
        loc_layout = QFormLayout(self.locomotion_group)
        self.locomotion_spin = QDoubleSpinBox(self.locomotion_group)
        self.locomotion_spin.setRange(0.0, 1e6)
        self.locomotion_spin.setDecimals(4)
        self.locomotion_spin.setSingleStep(0.01)
        self.locomotion_spin.valueChanged.connect(self._on_locomotion_threshold_changed)
        loc_layout.addRow("Threshold", self.locomotion_spin)
        left_panel = QWidget(self)
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        self.metrics_summary_panel = QWidget(self)
        metrics_summary_layout = QVBoxLayout(self.metrics_summary_panel)
        metrics_summary_layout.setContentsMargins(0, 0, 0, 0)
        metrics_summary_layout.addWidget(self._scope_label)
        metrics_summary_layout.addWidget(self._session_label)
        metrics_summary_layout.addWidget(self.timebase_warning)
        metrics_summary_layout.addWidget(self.movie_locomotion_warning)
        metrics_summary_layout.addWidget(self._baseline_label)
        metrics_summary_layout.addWidget(self._dirty_label)
        metrics_summary_layout.addWidget(self.threshold_group)
        metrics_summary_layout.addWidget(self.threshold_hint)
        metrics_summary_layout.addWidget(self.locomotion_group)
        metrics_summary_layout.addStretch(1)
        self.metrics_summary_section = CollapsibleSection("Metrics summary", self.metrics_summary_panel, self, expanded=False)
        left_layout.addWidget(self.metrics_summary_section)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(self.zoom_in_btn)
        zoom_row.addWidget(self.zoom_out_btn)
        zoom_row.addWidget(self.reset_zoom_btn)
        zoom_row.addStretch(1)
        left_layout.addLayout(zoom_row)
        left_layout.addWidget(self.canvas, stretch=1)
        self.experiment_index = ExperimentIndexTab(self.store, self)
        self.reference_sessions = DeeplabcutReferenceTab(self.store, self)
        self.reference_group = QGroupBox("Session markers", self)
        reference_layout = QVBoxLayout(self.reference_group)
        reference_layout.addWidget(self.do_not_use_check)
        reference_layout.addWidget(self.preprocessed_check)
        reference_layout.addWidget(self.reference_check)
        reference_layout.addWidget(self.timebase_aligned_check)
        cutoff_row = QHBoxLayout()
        cutoff_row.addWidget(self.cutoff_btn)
        cutoff_row.addWidget(self.clear_cutoff_btn)
        cutoff_row.addStretch(1)
        reference_layout.addLayout(cutoff_row)
        reference_layout.addWidget(self.cutoff_label)
        reference_hint = QLabel("Mark this session as excluded from statistics, cut it shorter, or flag it as a manual reference for DeeplabCut training.", self.reference_group)
        reference_hint.setWordWrap(True)
        reference_hint.setStyleSheet("color: #666666; font-size: 11px;")
        reference_layout.addWidget(reference_hint)
        self.mask_group = QGroupBox("Not visible pupil intervals", self)
        mask_layout = QVBoxLayout(self.mask_group)
        mask_layout.addWidget(self.video_widget)
        mask_layout.addWidget(self.video_time_label)
        button_row = QHBoxLayout()
        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.end_btn)
        button_row.addWidget(self.delete_btn)
        button_row.addWidget(self.clear_btn)
        mask_layout.addLayout(button_row)
        mask_layout.addWidget(self.interval_list, stretch=1)
        mask_layout.addWidget(self.save_btn)
        right_controls_panel = QWidget(self)
        right_controls_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        right_controls_layout = QVBoxLayout(right_controls_panel)
        right_controls_layout.setContentsMargins(0, 0, 0, 0)
        right_controls_layout.addWidget(self.calibration_group)
        right_controls_layout.addWidget(self.mask_group, stretch=1)
        index_group = QGroupBox("Experiment Index", self)
        index_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        index_layout = QVBoxLayout(index_group)
        index_layout.addWidget(self.experiment_index)
        reference_model_group = QGroupBox("Add for DLC model", self)
        reference_model_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        reference_model_layout = QVBoxLayout(reference_model_group)
        reference_model_layout.addWidget(self.reference_sessions)
        right_column_panel = QWidget(self)
        right_column_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        right_column_layout = QVBoxLayout(right_column_panel)
        right_column_layout.setContentsMargins(0, 0, 0, 0)
        right_column_layout.addWidget(self.reference_group)
        right_column_layout.addWidget(index_group)
        right_column_layout.addWidget(reference_model_group)
        right_column_layout.addStretch(1)
        right_panel = QSplitter(Qt.Horizontal, self)
        right_panel.setChildrenCollapsible(False)
        right_panel.setOpaqueResize(True)
        right_panel.setHandleWidth(6)
        right_panel.addWidget(right_controls_panel)
        right_panel.addWidget(right_column_panel)
        right_panel.setStretchFactor(0, 2)
        right_panel.setStretchFactor(1, 3)
        right_panel.setSizes([320, 760])
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setOpaqueResize(True)
        splitter.setHandleWidth(8)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([1000, 1100])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self._set_threshold_control_values(self._percentile_cutoffs, self._threshold_values)
        self._on_percentile_lock_toggled(self.percentile_lock_check.isChecked())
    def set_selection(self, animal_id: str, exp_id: str, view_mode: str):
        changed = (animal_id != self.animal_id) or (exp_id != self.exp_id) or (view_mode != self.view_mode)
        self.animal_id = animal_id
        self.exp_id = exp_id
        self.view_mode = view_mode
        if changed:
            self._calibration_selection_start = None
            self._calibration_selection_end = None
        self._reset_trace_view_pending = changed or not self._metrics_available()
        if changed or not self._metrics_available():
            self.refresh()
    def refresh(self):
        self._scope_sessions = self._sessions_for_scope()
        self.experiment_index.refresh()
        self.reference_sessions.refresh()
        if not self._metrics_available():
            self._show_unavailable_state()
            self._update_video()
            self._update_dirty_label()
            return
        self.threshold_group.setEnabled(True)
        self.locomotion_group.setEnabled(True)
        self.mask_group.setEnabled(True)
        self.zoom_in_btn.setEnabled(True)
        self.zoom_out_btn.setEnabled(True)
        self.reset_zoom_btn.setEnabled(True)
        self._load_animal_state(force_reload=False)
        self._apply_state_to_controls()
        self._refresh_interval_list()
        self._refresh_do_not_use_checkbox()
        self._refresh_preprocessed_checkbox()
        self._refresh_reference_checkbox()
        self._refresh_timebase_aligned_checkbox()
        self._refresh_cutoff_controls()
        self._draw_plots()
        self._update_video()
        self._update_dirty_label()
    def _sessions_for_scope(self) -> list[SessionSummary]:
        if self.animal_id == "All":
            return self.store.dataset_sessions()
        return self.store.sessions_for_animal(self.animal_id)
    def _animal_settings_key(self) -> str:
        return self.animal_id or "All"
    def _metrics_available(self) -> bool:
        return self.animal_id != "All"
    def _show_unavailable_state(self):
        message = "Metrics are not available for the All selection. Select a specific animal to view metrics."
        self._scope_label.setText("Scope: All")
        self._session_label.setText(message)
        self._baseline_label.setText("Choose a specific animal or expID to view metrics.")
        self._dirty_label.setText(message)
        self._dirty_label.setStyleSheet("color: #b00020; font-weight: 600;")
        self.threshold_group.setEnabled(False)
        self.locomotion_group.setEnabled(False)
        self.mask_group.setEnabled(False)
        self.zoom_in_btn.setEnabled(False)
        self.zoom_out_btn.setEnabled(False)
        self.reset_zoom_btn.setEnabled(False)
        if hasattr(self, "do_not_use_check"):
            self.do_not_use_check.setEnabled(False)
        if hasattr(self, "preprocessed_check"):
            self.preprocessed_check.setEnabled(False)
        if hasattr(self, "reference_check"):
            self.reference_check.setEnabled(False)
        if hasattr(self, "timebase_aligned_check"):
            self.timebase_aligned_check.setEnabled(False)
        if hasattr(self, "cutoff_btn"):
            self.cutoff_btn.setEnabled(False)
        if hasattr(self, "clear_cutoff_btn"):
            self.clear_cutoff_btn.setEnabled(False)
        if hasattr(self, "calibration_group"):
            self.calibration_group.setEnabled(False)
            self.calibration_group.setVisible(False)
        self._update_timebase_warning(None, None)
        self.movie_locomotion_warning.setText("")
        self.movie_locomotion_warning.setVisible(False)
        self._current_session = None
        self._current_payload = None
        self._calibration_suggestion = None
        self._clear_threshold_lines()
        self._clear_locomotion_threshold_line()
        self._clear_cursor_lines()
        pupil_ax = self.canvas.pupil_ax
        loc_ax = self.canvas.loc_ax
        face_ax = self.canvas.face_ax
        lock_ax = self.canvas.lock_ax
        pupil_ax.clear()
        loc_ax.clear()
        face_ax.clear()
        lock_ax.clear()
        style_axes(pupil_ax, title="Pupil dynamics", ylabel="z-scored radius")
        style_axes(loc_ax, title="Locomotion", ylabel="Wheel speed")
        style_axes(face_ax, title="Face motion", ylabel="Motion")
        style_axes(lock_ax, title="Lock state", xlabel="Time (s)", ylabel="Lock state")
        for ax in (pupil_ax, loc_ax, face_ax, lock_ax):
            ax.text(
                0.5,
                0.5,
                message,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="600",
            )
        self.canvas.draw_idle()
    def _load_animal_state(self, force_reload: bool = False):
        settings = self.store.get_animal_settings(self._animal_settings_key())
        cache_miss = force_reload or self._scope_cache_key != self._animal_settings_key()
        percentiles = self.store.global_pupil_percentile_cutoffs()
        global_signature = self.store.pupil_percentile_signature(percentiles)
        threshold_values = settings.get("threshold_values", [])
        threshold_signature = str(settings.get("threshold_signature", ""))
        needs_recompute = (
            cache_miss
            or settings.get("baseline_dirty", True)
            or settings.get("zscore_mean") is None
            or settings.get("zscore_std") is None
            or not isinstance(threshold_values, list)
            or len(threshold_values) != 3
            or threshold_signature != global_signature
        )
        if needs_recompute:
            mean, std, distribution, threshold_values = self._recompute_animal_threshold_state(self._animal_settings_key())
            settings = self.store.get_animal_settings(self._animal_settings_key())
            self._scope_cache_key = self._animal_settings_key()
            self._zscore_distribution = distribution
        self._zscore_mean = float(settings.get("zscore_mean", 0.0))
        self._zscore_std = float(settings.get("zscore_std", 1.0)) or 1.0
        if not np.isfinite(self._zscore_std) or self._zscore_std <= 0:
            self._zscore_std = 1.0
        if not needs_recompute:
            if cache_miss:
                self._scope_cache_key = self._animal_settings_key()
            self._zscore_distribution = self._zscore_distribution if self._scope_cache_key == self._animal_settings_key() else np.array([], dtype=float)
        self._percentile_cutoffs = [float(v) for v in percentiles]
        self._threshold_values = [float(v) for v in threshold_values]
        self._sort_thresholds_in_place(update_store=False)
        self._update_baseline_label()
    def _update_baseline_label(self):
        n_sessions = sum(1 for s in self._scope_sessions if s.has_right_pickle)
        self._scope_label.setText(
            f"Scope: {self.animal_id} | View: {self.view_mode} | Sessions with pupil data: {n_sessions}/{len(self._scope_sessions)}"
        )
        self._baseline_label.setText(
            f"Shared pupil percentiles: {', '.join(f'{v:.1f}%' for v in self._percentile_cutoffs)} | "
            f"Missing pupil buffer={self.store.global_pupil_missing_buffer_sec():.2f} s | "
            f"Pupil baseline mean={self._zscore_mean:.3f}, std={self._zscore_std:.3f}"
        )
        if self.view_mode == "Session" and self.exp_id:
            cutoff = self.store.session_analysis_cutoff_sec(self.exp_id)
            label_bits = [f"Focused expID: {self.exp_id}"]
            if cutoff is not None:
                label_bits.append(f"Cutoff: {format_seconds(cutoff)}")
            if self.store.is_session_timebase_aligned(self.exp_id):
                label_bits.append("Posteriorly aligned")
            self._session_label.setText(" | ".join(label_bits))
        elif self.animal_id == "All":
            self._session_label.setText("Cohort-wide overview")
        else:
            self._session_label.setText("Overall view for the selected animal")
    def _apply_state_to_controls(self):
        self._updating_controls = True
        try:
            for spin, pct in zip(self.percentile_spins, self._percentile_cutoffs):
                spin.setValue(float(pct))
            self.locomotion_spin.setValue(float(self.store.settings.get("global", {}).get("locomotion_threshold", 0.35)))
            self.missing_buffer_spin.setValue(float(self.store.global_pupil_missing_buffer_sec()))
            self._refresh_threshold_value_labels()
            self._update_threshold_lines()
        finally:
            self._updating_controls = False
    def _refresh_threshold_value_labels(self):
        for idx, label in enumerate(self.value_labels):
            if idx < len(self._threshold_values):
                percentile = percentile_from_value(self._zscore_distribution, self._threshold_values[idx]) if self._zscore_distribution.size else 0.0
                label.setText(f"{self._threshold_values[idx]:.3f}  ({percentile:.1f}%)")
            else:
                label.setText("--")
    def _on_percentile_lock_toggled(self, checked: bool):
        editable = bool(checked)
        for spin in self.percentile_spins:
            spin.setEnabled(editable)
            spin.setKeyboardTracking(False)
        if editable:
            self.percentile_lock_check.setToolTip(
                "Shared percentiles are unlocked. Type the final values, or drag threshold lines for immediate updates."
            )
            self.threshold_hint.setText("Shared percentile fields are unlocked for typing. Dragging the threshold lines still works.")
        else:
            self.percentile_lock_check.setToolTip(
                "Enable direct typing in the shared percentile fields. Dragging threshold lines stays available either way."
            )
            self.threshold_hint.setText("Unlock the shared percentile fields to type values. Dragging the threshold lines stays live.")
    def _set_threshold_control_values(self, percentiles: list[float], values: list[float]):
        self._updating_controls = True
        try:
            for spin, pct in zip(self.percentile_spins, percentiles):
                spin.setValue(float(pct))
            self._threshold_values = [float(v) for v in values]
            self._refresh_threshold_value_labels()
            self._update_threshold_lines()
        finally:
            self._updating_controls = False
    def _update_threshold_lines(self):
        if not self._threshold_lines:
            return
        for line, value in zip(self._threshold_lines, self._threshold_values):
            try:
                line.set_y(float(value), trigger_callback=False)
            except Exception:
                pass
        self.canvas.draw_idle()
    def _clear_threshold_lines(self):
        for line in self._threshold_lines:
            try:
                line.disconnect()
            except Exception:
                pass
        self._threshold_lines = []
    def _clear_locomotion_threshold_line(self):
        if self._locomotion_threshold_line is None:
            return
        try:
            self._locomotion_threshold_line.disconnect()
        except Exception:
            pass
        self._locomotion_threshold_line = None
    def _make_threshold_lines(self, ax):
        self._threshold_lines = []
        for idx, value in enumerate(self._threshold_values):
            line = DraggableHLine(
                ax,
                value,
                color=BOUNDARY_COLORS[idx],
                linestyle="--",
                linewidth=2.0,
                on_changed=lambda y, i=idx: self._on_threshold_line_moved(i, y),
            )
            self._threshold_lines.append(line)
    def _make_locomotion_threshold_line(self, ax):
        self._locomotion_threshold_line = DraggableHLine(
            ax,
            float(self.locomotion_spin.value()),
            color="tab:red",
            linestyle="--",
            linewidth=2.0,
            label="threshold",
            tolerance_px=10,
            on_changed=self._on_locomotion_threshold_line_moved,
        )
    def _sort_thresholds_in_place(self, update_store: bool = True):
        paired = sorted(zip(self._percentile_cutoffs, self._threshold_values), key=lambda item: item[1])
        if paired:
            self._percentile_cutoffs = [float(p) for p, _ in paired]
            self._threshold_values = [float(v) for _, v in paired]
        if update_store:
            self._persist_threshold_state(mark_baseline_dirty=False)
    def _persist_threshold_state(self, mark_baseline_dirty: bool = False):
        percentiles = self.store.set_global_pupil_percentile_cutoffs(self._percentile_cutoffs)
        self._percentile_cutoffs = [float(v) for v in percentiles]
        current_animal = self._animal_settings_key()
        for animal_id in ["All", *self.store.animals()]:
            mean, std, distribution, threshold_values = self._recompute_animal_threshold_state(animal_id)
            if animal_id == current_animal:
                self._zscore_mean = float(mean)
                self._zscore_std = float(std)
                self._zscore_distribution = distribution
                self._threshold_values = [float(v) for v in threshold_values]
        settings = self.store.get_animal_settings(current_animal)
        if mark_baseline_dirty:
            settings["baseline_dirty"] = True
        settings["stats_dirty"] = True
        self.store.set_animal_settings(current_animal, settings)
        self.thresholds_changed.emit()
        self._refresh_preprocessed_checkbox()
        self._update_dirty_label()
        if self.view_mode == "Session" and self._current_session is not None and self._current_payload is not None:
            self._refresh_extra_large_calibration_controls(self._current_session, self._current_payload)
    def _persist_locomotion_threshold(self):
        self.store.settings.setdefault("global", {})
        self.store.settings["global"]["locomotion_threshold"] = float(self.locomotion_spin.value())
        self.store.save_settings()
        self.store.mark_stats_dirty()
        self.thresholds_changed.emit()
        self._update_dirty_label()
    def _persist_missing_buffer(self):
        self.store.set_global_pupil_missing_buffer_sec(self.missing_buffer_spin.value())
        self.thresholds_changed.emit()
        self._update_dirty_label()
    def _update_dirty_label(self):
        dirty = self.store.is_animal_dirty(self._animal_settings_key()) or self.store.is_stats_dirty()
        self._dirty_label.setText("Statistics need to be rerun for this scope." if dirty else "Statistics are clean")
        self._dirty_label.setStyleSheet("color: #b00020; font-weight: 600;" if dirty else "color: #1b5e20; font-weight: 600;")
    def _update_timebase_warning(self, summary: SessionSummary | None, payload: LoadedSession | None):
        if not hasattr(self, "timebase_warning"):
            return
        if self.view_mode != "Session" or summary is None or payload is None:
            self.timebase_warning.setText("")
            self.timebase_warning.setVisible(False)
            return
        video_duration = float(summary.video_duration_sec or 0.0)
        session_duration = float(payload.t[-1]) if payload.t.size else 0.0
        if not summary.has_right_video or video_duration <= 0.0 or session_duration <= 0.0:
            self.timebase_warning.setText("")
            self.timebase_warning.setVisible(False)
            return
        diff = abs(video_duration - session_duration)
        if diff > 5.0:
            aligned = self.store.is_session_timebase_aligned(summary.exp_id)
            if aligned:
                self.timebase_warning.setText(
                    f"Posteriorly aligned timebase mismatch: video {video_duration:.1f} s vs session clock {session_duration:.1f} s. "
                    f"This expID was resampled after acquisition, so the session clock is kept even though the raw video duration differs."
                )
            else:
                shorter = "video" if video_duration < session_duration else "CSV session clock"
                self.timebase_warning.setText(
                    f"Timebase mismatch: video {video_duration:.1f} s vs session clock {session_duration:.1f} s. "
                    f"The {shorter} is shorter; the CSV session clock is kept."
                )
            self.timebase_warning.setVisible(True)
        else:
            self.timebase_warning.setText("")
            self.timebase_warning.setVisible(False)
    def _update_movie_locomotion_warning(self, summary: SessionSummary | None, payload: LoadedSession | None):
        if not hasattr(self, "movie_locomotion_warning"):
            return
        if self.view_mode != "Session" or summary is None or payload is None:
            self.movie_locomotion_warning.setText("")
            self.movie_locomotion_warning.setVisible(False)
            return
        video_duration = float(summary.video_duration_sec or 0.0)
        locomotion_duration = float(payload.locomotion_t[-1]) if payload.locomotion_t.size else 0.0
        if not summary.has_right_video or video_duration <= 0.0 or locomotion_duration <= 0.0:
            self.movie_locomotion_warning.setText("")
            self.movie_locomotion_warning.setVisible(False)
            return
        diff = abs(video_duration - locomotion_duration)
        if diff > 5.0:
            if self.store.is_session_timebase_aligned(summary.exp_id):
                self.movie_locomotion_warning.setText(
                    f"Movie / locomotion mismatch: the right video ends at {video_duration:.1f} s, while the locomotion trace spans {locomotion_duration:.1f} s. "
                    f"This session was aligned after acquisition, so the movie is shown on the session clock and the full locomotion trace is kept."
                )
            else:
                longer = "locomotion trace" if locomotion_duration > video_duration else "movie"
                self.movie_locomotion_warning.setText(
                    f"Movie / locomotion mismatch: the right video ends at {video_duration:.1f} s, while the locomotion trace spans {locomotion_duration:.1f} s. "
                    f"The {longer} is longer; the raw movie and locomotion timeline are not the same length."
                )
            self.movie_locomotion_warning.setVisible(True)
        else:
            self.movie_locomotion_warning.setText("")
            self.movie_locomotion_warning.setVisible(False)
    def _on_percentile_changed(self, idx: int, value: float):
        if self._updating_controls:
            return
        self._percentile_cutoffs[idx] = float(value)
        if self._zscore_distribution.size:
            values = percentile_threshold_values(self._zscore_distribution, self._percentile_cutoffs)
        else:
            values = [float(v) for v in self._threshold_values]
        self._threshold_values = [float(v) for v in values]
        self._sort_thresholds_in_place(update_store=False)
        self._updating_controls = True
        try:
            self._set_threshold_control_values(self._percentile_cutoffs, self._threshold_values)
        finally:
            self._updating_controls = False
        self._persist_threshold_state(mark_baseline_dirty=False)
    def _on_threshold_line_moved(self, idx: int, value: float):
        if self._updating_controls:
            return
        self._threshold_values[idx] = float(value)
        if self._zscore_distribution.size:
            self._percentile_cutoffs = [
                float(percentile_from_value(self._zscore_distribution, v)) for v in self._threshold_values
            ]
        self._sort_thresholds_in_place(update_store=False)
        self._updating_controls = True
        try:
            self._set_threshold_control_values(self._percentile_cutoffs, self._threshold_values)
        finally:
            self._updating_controls = False
        self._persist_threshold_state(mark_baseline_dirty=False)
    def _on_locomotion_threshold_changed(self, value: float):
        if self._updating_controls:
            return
        self._persist_locomotion_threshold()
        self._draw_plots()
    def _on_locomotion_threshold_line_moved(self, value: float):
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            self.locomotion_spin.setValue(float(value))
        finally:
            self._updating_controls = False
        self._persist_locomotion_threshold()
        self._draw_plots()
    def _on_missing_buffer_changed(self, value: float):
        if self._updating_controls:
            return
        self._persist_missing_buffer()
    def _current_percentile_signature(self) -> str:
        return self.store.pupil_percentile_signature(self._percentile_cutoffs)
    def _recompute_animal_threshold_state(self, animal_id: str) -> tuple[float, float, np.ndarray, list[float]]:
        scope = "All" if animal_id == "All" else None
        mean, std = compute_animal_baseline(self.store, animal_id, scope=scope)
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        zmap = animal_zscores(self.store, animal_id, mean=float(mean), std=float(std))
        pooled = [values[np.isfinite(values)] for values in zmap.values() if np.any(np.isfinite(values))]
        distribution = np.concatenate(pooled) if pooled else np.array([], dtype=float)
        threshold_values = percentile_threshold_values(distribution, self._percentile_cutoffs) if distribution.size else [0.0, 0.0, 0.0]
        settings = self.store.get_animal_settings(animal_id)
        settings["zscore_mean"] = float(mean)
        settings["zscore_std"] = float(std)
        settings["threshold_values"] = [float(v) for v in threshold_values]
        settings["threshold_signature"] = self._current_percentile_signature()
        settings["baseline_dirty"] = False
        settings["stats_dirty"] = True
        self.store.set_animal_settings(animal_id, settings)
        return float(mean), float(std), distribution.astype(float), [float(v) for v in threshold_values]
    def _refresh_preprocessed_checkbox(self):
        if not hasattr(self, "preprocessed_check"):
            return
        if self.view_mode != "Session" or not self.exp_id:
            self.preprocessed_check.blockSignals(True)
            self.preprocessed_check.setChecked(False)
            self.preprocessed_check.blockSignals(False)
            self.preprocessed_check.setEnabled(False)
            self.preprocessed_check.setToolTip("Available only for a specific expID.")
            return
        state = self.store.load_session_state(self.exp_id)
        current_sig = self._current_percentile_signature()
        stored_sig = str(state.get("threshold_signature", ""))
        preprocessed = bool(state.get("preprocessed", False))
        checked = preprocessed and stored_sig == current_sig
        self.preprocessed_check.blockSignals(True)
        self.preprocessed_check.setChecked(checked)
        self.preprocessed_check.blockSignals(False)
        self.preprocessed_check.setEnabled(True)
        if checked:
            tip = "Marked pre-processed under the current shared pupil percentiles."
        elif preprocessed and stored_sig and stored_sig != current_sig:
            tip = "This expID was marked pre-processed, but the shared pupil percentiles changed."
        else:
            tip = "Mark this expID as pre-processed for the current shared pupil percentiles."
        self.preprocessed_check.setToolTip(tip)
    def _refresh_do_not_use_checkbox(self):
        if not hasattr(self, "do_not_use_check"):
            return
        if self.view_mode != "Session" or not self.exp_id:
            self.do_not_use_check.blockSignals(True)
            self.do_not_use_check.setChecked(False)
            self.do_not_use_check.blockSignals(False)
            self.do_not_use_check.setEnabled(False)
            self.do_not_use_check.setToolTip("Available only for a specific expID.")
            return
        state = self.store.load_session_state(self.exp_id)
        do_not_use = bool(state.get("do_not_use", False))
        forced_do_not_use = self.store.is_session_forced_do_not_use(self.exp_id)
        self.do_not_use_check.blockSignals(True)
        self.do_not_use_check.setChecked(do_not_use or forced_do_not_use)
        self.do_not_use_check.blockSignals(False)
        self.do_not_use_check.setEnabled(not forced_do_not_use)
        if forced_do_not_use:
            tip = "Sessions under 30 minutes and TEST sessions are always excluded from statistics."
        elif do_not_use:
            tip = "This expID is excluded from statistics."
        else:
            tip = "Mark this expID as excluded from statistics."
        self.do_not_use_check.setToolTip(tip)
    def _refresh_reference_checkbox(self):
        if not hasattr(self, "reference_check"):
            return
        if self.view_mode != "Session" or not self.exp_id:
            self.reference_check.blockSignals(True)
            self.reference_check.setChecked(False)
            self.reference_check.blockSignals(False)
            self.reference_check.setEnabled(False)
            self.reference_check.setToolTip("Available only for a specific expID.")
            return
        state = self.store.load_session_state(self.exp_id)
        reference = bool(state.get("deeplabcut_reference", False))
        self.reference_check.blockSignals(True)
        self.reference_check.setChecked(reference)
        self.reference_check.blockSignals(False)
        self.reference_check.setEnabled(True)
        if reference:
            tip = "This expID is marked for DLC model use."
        else:
            tip = "Mark this expID for DLC model use."
        self.reference_check.setToolTip(tip)
    def _refresh_timebase_aligned_checkbox(self):
        if not hasattr(self, "timebase_aligned_check"):
            return
        if self.view_mode != "Session" or not self.exp_id:
            self.timebase_aligned_check.blockSignals(True)
            self.timebase_aligned_check.setChecked(False)
            self.timebase_aligned_check.blockSignals(False)
            self.timebase_aligned_check.setEnabled(False)
            self.timebase_aligned_check.setToolTip("Available only for a specific expID.")
            return
        aligned = self.store.is_session_timebase_aligned(self.exp_id)
        self.timebase_aligned_check.blockSignals(True)
        self.timebase_aligned_check.setChecked(aligned)
        self.timebase_aligned_check.blockSignals(False)
        self.timebase_aligned_check.setEnabled(True)
        if aligned:
            tip = "This expID was aligned after acquisition, so the warning uses the posteriorly aligned wording."
        else:
            tip = "Mark this expID if its timebase was aligned after acquisition."
        self.timebase_aligned_check.setToolTip(tip)
    def _on_do_not_use_toggled(self, checked: bool):
        if self._updating_controls or self.view_mode != "Session" or not self.exp_id:
            return
        self.store.set_session_do_not_use(self.exp_id, bool(checked))
        self._refresh_do_not_use_checkbox()
        self.session_state_changed.emit()
    def _on_preprocessed_toggled(self, checked: bool):
        if self._updating_controls or self.view_mode != "Session" or not self.exp_id:
            return
        self.store.set_session_preprocessed(self.exp_id, bool(checked), threshold_signature=self._current_percentile_signature())
        self._refresh_preprocessed_checkbox()
        self.session_state_changed.emit()
    def _on_reference_toggled(self, checked: bool):
        if self._updating_controls or self.view_mode != "Session" or not self.exp_id:
            return
        self.store.set_session_deeplabcut_reference(self.exp_id, bool(checked))
        self._refresh_reference_checkbox()
        self.session_state_changed.emit()
    def _on_timebase_aligned_toggled(self, checked: bool):
        if self._updating_controls or self.view_mode != "Session" or not self.exp_id:
            return
        self.store.set_session_timebase_aligned(self.exp_id, bool(checked))
        self._refresh_timebase_aligned_checkbox()
        self._update_timebase_warning(self._current_session, self._current_payload)
        self.session_state_changed.emit()
    def _session_cutoff_text(self, cutoff_sec: float | None) -> str:
        if cutoff_sec is None:
            return "Cutoff: none"
        return f"Cutoff: {format_seconds(cutoff_sec)}"
    def _refresh_cutoff_controls(self):
        if not hasattr(self, "cutoff_btn"):
            return
        if self.view_mode != "Session" or not self.exp_id:
            self.cutoff_btn.setEnabled(False)
            self.clear_cutoff_btn.setEnabled(False)
            self.cutoff_label.setText("Cutoff: unavailable")
            self.cutoff_btn.setToolTip("Available only for a specific expID with video data.")
            self.clear_cutoff_btn.setToolTip("Available only for a specific expID.")
            return
        summary = self.store.get_session_summary(self.exp_id)
        cutoff = self.store.session_analysis_cutoff_sec(self.exp_id)
        has_video = bool(summary and summary.has_right_video and summary.right_video and Path(summary.right_video).exists())
        self.cutoff_btn.setEnabled(has_video)
        self.clear_cutoff_btn.setEnabled(cutoff is not None)
        self.cutoff_label.setText(self._session_cutoff_text(cutoff))
        if has_video:
            self.cutoff_btn.setToolTip("Store the current video time as the analysis cutoff for this session.")
        else:
            self.cutoff_btn.setToolTip("This session has no right-eye video available to use as a cutoff source.")
        if cutoff is None:
            self.clear_cutoff_btn.setToolTip("No cutoff is set for this session.")
        else:
            self.clear_cutoff_btn.setToolTip("Remove the analysis cutoff and restore the full session.")
    def _set_cutoff_from_current_time(self):
        if self._updating_controls or self.view_mode != "Session" or not self.exp_id:
            return
        cutoff = self.video_widget.current_time()
        self.store.set_session_analysis_cutoff(self.exp_id, cutoff)
        self.refresh()
    def _clear_cutoff(self):
        if self._updating_controls or self.view_mode != "Session" or not self.exp_id:
            return
        self.store.set_session_analysis_cutoff(self.exp_id, None)
        self.refresh()
    def _analysis_payload(self, payload: LoadedSession, summary: SessionSummary | None = None) -> LoadedSession:
        summary = summary or payload.summary
        cutoff = self.store.session_analysis_cutoff_sec(summary.exp_id)
        if cutoff is None:
            return payload
        frame_mask = analysis_cutoff_mask(payload.t, cutoff)
        locomotion_mask = analysis_cutoff_mask(payload.locomotion_t, cutoff)
        brake_mask = analysis_cutoff_mask(payload.brake_t, cutoff)
        return replace(
            payload,
            t=apply_time_mask(payload.t, frame_mask),
            radius=apply_time_mask(payload.radius, frame_mask),
            x=apply_time_mask(payload.x, frame_mask),
            y=apply_time_mask(payload.y, frame_mask),
            velocity=apply_time_mask(payload.velocity, frame_mask),
            qc=apply_time_mask(payload.qc, frame_mask),
            in_eye=apply_time_mask(payload.in_eye, frame_mask),
            brake_raw=apply_time_mask(payload.brake_raw, brake_mask),
            brake_t=apply_time_mask(payload.brake_t, brake_mask),
            wheel_pos=apply_time_mask(payload.wheel_pos, locomotion_mask),
            locomotion=apply_time_mask(payload.locomotion, locomotion_mask),
            locomotion_t=apply_time_mask(payload.locomotion_t, locomotion_mask),
            face_motion=apply_time_mask(payload.face_motion, frame_mask),
            face_motion_t=apply_time_mask(payload.face_motion_t, frame_mask),
            eye_lid_x=apply_time_mask(payload.eye_lid_x, frame_mask),
            eye_lid_y=apply_time_mask(payload.eye_lid_y, frame_mask),
            eyeX=apply_time_mask(payload.eyeX, frame_mask),
            eyeY=apply_time_mask(payload.eyeY, frame_mask),
            pupilX=apply_time_mask(payload.pupilX, frame_mask),
            pupilY=apply_time_mask(payload.pupilY, frame_mask),
        )
    def _reset_trace_zoom_view(self):
        self._reset_trace_view_pending = True
        self.canvas.reset_zoom()
    def _should_block_trace_pan(self, event) -> bool:
        if self.view_mode != "Session":
            return False
        for line in getattr(self, "_threshold_lines", []):
            try:
                if line._is_near_line(event):
                    return True
            except Exception:
                pass
        line = getattr(self, "_locomotion_threshold_line", None)
        if line is not None:
            try:
                if line._is_near_line(event):
                    return True
            except Exception:
                pass
        return False
    def _trace_bounds_from_arrays(self, *arrays: np.ndarray) -> tuple[float, float] | None:
        finite_parts = []
        for arr in arrays:
            if arr is None:
                continue
            values = np.asarray(arr, dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size:
                finite_parts.append(values)
        if not finite_parts:
            return None
        combined = np.concatenate(finite_parts)
        if combined.size == 0:
            return None
        xmin = float(np.nanmin(combined))
        xmax = float(np.nanmax(combined))
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            return None
        if xmax <= xmin:
            xmax = xmin + 1.0
        return xmin, xmax
    def _apply_trace_bounds(self, bounds: tuple[float, float] | None):
        if bounds is None:
            self.canvas.clear_limits()
            self._reset_trace_view_pending = False
            return
        xmin, xmax = bounds
        self.canvas.set_x_bounds(xmin, xmax, reset_view=self._reset_trace_view_pending)
        self.canvas.apply_view()
        self._reset_trace_view_pending = False
    def _video_time_axis(self, summary: SessionSummary, frame_count: int | None = None) -> np.ndarray:
        timestamps = self.store.load_frame_timestamps(summary.exp_id)
        if timestamps.size:
            return np.asarray(timestamps, dtype=float)
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
    def _load_session_payload(self, summary: SessionSummary) -> LoadedSession:
        try:
            bundle = self.store.load_session_bundle(summary.exp_id)
            face_motion_t, face_motion = self.store.load_face_motion(summary.exp_id)
            if face_motion_t is None or face_motion is None:
                face_motion_t = np.array([], dtype=float)
                face_motion = np.array([], dtype=float)
            return LoadedSession(
                summary=summary,
                t=np.asarray(bundle.t, dtype=float),
                radius=np.asarray(bundle.radius, dtype=float),
                x=np.asarray(bundle.x, dtype=float),
                y=np.asarray(bundle.y, dtype=float),
                velocity=np.asarray(bundle.velocity, dtype=float),
                qc=np.asarray(bundle.qc, dtype=float),
                in_eye=np.asarray(bundle.in_eye, dtype=bool),
                brake_raw=np.asarray(bundle.brake_raw, dtype=float),
                brake_t=np.asarray(bundle.brake_t, dtype=float),
                wheel_pos=np.asarray(bundle.wheel_pos, dtype=float),
                locomotion=np.asarray(bundle.locomotion, dtype=float),
                locomotion_t=np.asarray(bundle.locomotion_t, dtype=float),
                face_motion=np.asarray(face_motion, dtype=float),
                face_motion_t=np.asarray(face_motion_t, dtype=float),
                eye_lid_x=np.asarray(bundle.eye_lid_x, dtype=float),
                eye_lid_y=np.asarray(bundle.eye_lid_y, dtype=float),
                eyeX=np.asarray(bundle.eyeX, dtype=float),
                eyeY=np.asarray(bundle.eyeY, dtype=float),
                pupilX=np.asarray(bundle.pupilX, dtype=float),
                pupilY=np.asarray(bundle.pupilY, dtype=float),
                source_signature=dict(bundle.source_signature),
                has_pupil=True,
            )
        except Exception:
            try:
                brake_t, brake_raw = self.store._load_lock_trace(summary.animal_id, summary.exp_id)
            except Exception:
                brake_t = np.array([], dtype=float)
                brake_raw = np.array([], dtype=float)
            try:
                locomotion_t, _, wheel_pos, locomotion = self.store._load_locomotion_trace(summary.animal_id, summary.exp_id)
            except Exception:
                locomotion_t = np.array([], dtype=float)
                wheel_pos = np.array([], dtype=float)
                locomotion = np.array([], dtype=float)
            frame_t = self._video_time_axis(summary, frame_count=summary.video_frame_count)
            if frame_t.size == 0 and locomotion_t.size:
                frame_t = np.asarray(locomotion_t, dtype=float)
            face_motion_t = np.array([], dtype=float)
            face_motion = np.array([], dtype=float)
            return LoadedSession(
                summary=summary,
                t=np.asarray(frame_t, dtype=float),
                radius=np.array([], dtype=float),
                x=np.array([], dtype=float),
                y=np.array([], dtype=float),
                velocity=np.array([], dtype=float),
                qc=np.array([], dtype=float),
                in_eye=np.zeros((0, 1), dtype=bool),
                brake_raw=np.asarray(brake_raw, dtype=float),
                brake_t=np.asarray(brake_t, dtype=float),
                wheel_pos=np.asarray(wheel_pos, dtype=float),
                locomotion=np.asarray(locomotion, dtype=float),
                locomotion_t=np.asarray(locomotion_t, dtype=float),
                face_motion=np.array([], dtype=float),
                face_motion_t=np.array([], dtype=float),
                eye_lid_x=np.array([], dtype=float),
                eye_lid_y=np.array([], dtype=float),
                eyeX=np.array([], dtype=float),
                eyeY=np.array([], dtype=float),
                pupilX=np.array([], dtype=float),
                pupilY=np.array([], dtype=float),
                source_signature={},
                has_pupil=False,
            )
    def _build_scope_trace(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float, str]], list[str]]:
        if not self._scope_sessions:
            empty = np.array([], dtype=float)
            return empty, empty, empty, empty, empty, empty, empty, empty, [], []
        pupil_t_parts: list[np.ndarray] = []
        z_parts: list[np.ndarray] = []
        loc_t_parts: list[np.ndarray] = []
        loc_parts: list[np.ndarray] = []
        face_t_parts: list[np.ndarray] = []
        face_parts: list[np.ndarray] = []
        lock_t_parts: list[np.ndarray] = []
        lock_parts: list[np.ndarray] = []
        spans: list[tuple[float, float, str]] = []
        labels: list[str] = []
        offset = 0.0
        for summary in self._scope_sessions:
            payload = self._analysis_payload(self._load_session_payload(summary), summary)
            labels.append(summary.exp_id)
            if payload.t.size == 0 and payload.locomotion_t.size == 0 and payload.brake_t.size == 0:
                continue
            pupil_t = np.asarray(payload.t, dtype=float)
            loc_t = np.asarray(payload.locomotion_t, dtype=float)
            lock_t = np.asarray(payload.brake_t, dtype=float)
            radius = np.asarray(payload.radius, dtype=float)
            locomotion = np.asarray(payload.locomotion, dtype=float)
            face_t = np.asarray(payload.face_motion_t, dtype=float)
            face_motion = np.asarray(payload.face_motion, dtype=float)
            lock_state = np.asarray(payload.brake_raw, dtype=float)
            if radius.size:
                n = min(pupil_t.size, radius.size)
                pupil_t = pupil_t[:n]
                radius = radius[:n]
                z = (radius.astype(float) - self._zscore_mean) / self._zscore_std
                visible = payload.visible_mask_base[:n]
                if visible.size == z.size:
                    z = z.copy()
                    z[~visible] = np.nan
                z[~np.isfinite(z)] = np.nan
            else:
                z = np.full(pupil_t.shape, np.nan, dtype=float)
            if locomotion.size and loc_t.size:
                m = min(loc_t.size, locomotion.size)
                loc_t = loc_t[:m]
                locomotion = locomotion[:m]
            else:
                loc_t = np.array([], dtype=float)
                locomotion = np.array([], dtype=float)
            if face_motion.size and face_t.size:
                f = min(face_t.size, face_motion.size)
                face_t = face_t[:f]
                face_motion = face_motion[:f]
            else:
                face_t = np.array([], dtype=float)
                face_motion = np.array([], dtype=float)
            if lock_state.size and lock_t.size:
                k = min(lock_t.size, lock_state.size)
                lock_t = lock_t[:k]
                lock_state = lock_state[:k]
            else:
                lock_t = np.array([], dtype=float)
                lock_state = np.array([], dtype=float)
            pupil_t_shift = pupil_t + offset
            loc_t_shift = loc_t + offset
            face_t_shift = face_t + offset
            lock_t_shift = lock_t + offset
            pupil_t_parts.append(pupil_t_shift)
            z_parts.append(z)
            if loc_t_shift.size:
                loc_t_parts.append(loc_t_shift)
                loc_parts.append(locomotion)
            if face_t_shift.size:
                face_t_parts.append(face_t_shift)
                face_parts.append(face_motion)
            if lock_t_shift.size:
                lock_t_parts.append(lock_t_shift)
                lock_parts.append(lock_state)
            session_duration = float(self.store.effective_session_duration_sec(summary.exp_id))
            if session_duration <= 0.0:
                session_duration = max(
                    float(pupil_t[-1]) if pupil_t.size else 0.0,
                    float(loc_t[-1]) if loc_t.size else 0.0,
                    float(lock_t[-1]) if lock_t.size else 0.0,
                )
            spans.append((float(offset), float(offset + session_duration), summary.exp_id))
            offset += session_duration + 30.0
        if not pupil_t_parts and not loc_t_parts and not face_t_parts and not lock_t_parts:
            empty = np.array([], dtype=float)
            return empty, empty, empty, empty, empty, empty, empty, empty, spans, labels
        pupil_t_out = np.concatenate(pupil_t_parts) if pupil_t_parts else np.array([], dtype=float)
        z_out = np.concatenate(z_parts) if z_parts else np.array([], dtype=float)
        loc_t_out = np.concatenate(loc_t_parts) if loc_t_parts else np.array([], dtype=float)
        loc_out = np.concatenate(loc_parts) if loc_parts else np.array([], dtype=float)
        face_t_out = np.concatenate(face_t_parts) if face_t_parts else np.array([], dtype=float)
        face_out = np.concatenate(face_parts) if face_parts else np.array([], dtype=float)
        lock_t_out = np.concatenate(lock_t_parts) if lock_t_parts else np.array([], dtype=float)
        lock_out = np.concatenate(lock_parts) if lock_parts else np.array([], dtype=float)
        return pupil_t_out, z_out, loc_t_out, loc_out, face_t_out, face_out, lock_t_out, lock_out, spans, labels
    def _clear_cursor_lines(self):
        for line in self._cursor_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._cursor_lines = []
    def _make_cursor_lines(self, *axes):
        self._clear_cursor_lines()
        for ax in axes:
            line = ax.axvline(0.0, color="tab:cyan", linestyle="-", linewidth=1.6, alpha=0.85, zorder=30)
            self._cursor_lines.append(line)
    def _set_cursor_position(self, value: float | None):
        if not self._cursor_lines or value is None or not np.isfinite(value):
            return
        for line in self._cursor_lines:
            try:
                line.set_xdata([float(value), float(value)])
            except Exception:
                pass
        self.canvas.draw_idle()
    def _plot_lock_state(self, ax, lock_t: np.ndarray, lock_state: np.ndarray, empty_message: str = "No lock state available") -> bool:
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["Locked", "Unlocked"])
        if lock_t.size and lock_state.size:
            n = min(lock_t.size, lock_state.size)
            lock_t = np.asarray(lock_t[:n], dtype=float)
            lock_state = np.asarray(lock_state[:n], dtype=float)
            finite = np.isfinite(lock_t) & np.isfinite(lock_state)
            if np.any(finite):
                values = np.where(lock_state[finite] > 0.5, 1.0, 0.0)
                ax.step(lock_t[finite], values, where="post", color="tab:purple", linewidth=1.3, label="lock state")
                return True
        ax.text(0.5, 0.5, empty_message, transform=ax.transAxes, ha="center", va="center")
        return False
    def _draw_plots(self):
        self._clear_threshold_lines()
        self._clear_locomotion_threshold_line()
        self._clear_cursor_lines()
        pupil_ax = self.canvas.pupil_ax
        loc_ax = self.canvas.loc_ax
        face_ax = self.canvas.face_ax
        lock_ax = self.canvas.lock_ax
        pupil_ax.clear()
        loc_ax.clear()
        face_ax.clear()
        lock_ax.clear()
        style_axes(pupil_ax, title="Pupil dynamics", ylabel="z-scored radius")
        style_axes(loc_ax, title="Locomotion", ylabel="Wheel speed")
        style_axes(face_ax, title="Face motion", ylabel="Motion")
        style_axes(lock_ax, title="Lock state", xlabel="Time (s)", ylabel="Lock state")
        if not self._metrics_available():
            self._show_unavailable_state()
            return
        bounds = None
        if self.view_mode == "Session":
            summary = self.store.get_session_summary(self.exp_id)
            if summary is None:
                pupil_ax.text(0.5, 0.5, "Unknown expID", transform=pupil_ax.transAxes, ha="center", va="center")
                self.canvas.draw_idle()
                return
            self._current_session = summary
            raw_payload = self._load_session_payload(summary)
            self._current_payload = raw_payload
            payload = self._analysis_payload(raw_payload, summary)
            if payload.has_pupil and payload.radius.size:
                z = (payload.radius.astype(float) - self._zscore_mean) / self._zscore_std
                visible = payload.visible_mask_base
                z_plot = z.copy()
                if visible.size == z_plot.size:
                    z_plot[~visible] = np.nan
                z_plot[~np.isfinite(z_plot)] = np.nan
                pupil_ax.plot(payload.t[: z_plot.size], z_plot, color="black", linewidth=1.5, label="pupil z-score")
                manual_masks = self.store.load_manual_masks(summary.exp_id)
                for start_t, end_t in manual_masks:
                    pupil_ax.axvspan(start_t, end_t, color="tab:red", alpha=0.15)
                calibration = self.store.session_extra_large_calibration(summary.exp_id)
                extra_large_mask = build_extra_large_mask(
                    self.store,
                    payload,
                    calibration,
                    manual_masks,
                    manual_buffer_sec=self.store.global_pupil_missing_buffer_sec(),
                )
                extra_large_intervals = mask_to_intervals(payload.t, extra_large_mask)
                for idx, (start_t, end_t) in enumerate(extra_large_intervals):
                    pupil_ax.axvspan(
                        start_t,
                        end_t,
                        facecolor="gold",
                        alpha=0.20,
                        hatch="//",
                        edgecolor="darkgoldenrod",
                        linewidth=0.0,
                        label="inferred extra-large missing" if idx == 0 else None,
                    )
                self._refresh_extra_large_calibration_controls(summary, payload, pupil_ax)
            else:
                self._refresh_extra_large_calibration_controls(summary, payload, pupil_ax)
                pupil_ax.text(
                    0.5,
                    0.5,
                    "No right-eye pupil data available",
                    transform=pupil_ax.transAxes,
                    ha="center",
                    va="center",
                )
            if payload.locomotion.size:
                loc_t = np.asarray(payload.locomotion_t, dtype=float)
                loc_n = min(loc_t.size, payload.locomotion.size)
                if loc_n:
                    loc_ax.plot(loc_t[:loc_n], payload.locomotion[:loc_n], color="tab:blue", linewidth=1.5, label="locomotion")
                loc_thr = float(self.store.settings.get("global", {}).get("locomotion_threshold", 0.35))
                loc_ax.axhline(loc_thr, color="tab:red", linestyle="--", linewidth=1.5, label="threshold")
            else:
                loc_ax.text(0.5, 0.5, "No locomotion data", transform=loc_ax.transAxes, ha="center", va="center")
            if payload.face_motion.size:
                face_t = np.asarray(payload.face_motion_t, dtype=float)
                face_n = min(face_t.size, payload.face_motion.size)
                if face_n:
                    face_ax.plot(face_t[:face_n], payload.face_motion[:face_n], color="tab:purple", linewidth=1.2, label="face motion")
            else:
                face_ax.text(0.5, 0.5, "No face motion data", transform=face_ax.transAxes, ha="center", va="center")
            self._plot_lock_state(lock_ax, np.asarray(payload.brake_t, dtype=float), np.asarray(payload.brake_raw, dtype=float))
            pupil_ax.set_title(f"Pupil dynamics - {summary.exp_id}", pad=10)
            loc_ax.set_title(f"Locomotion - {summary.exp_id}", pad=10)
            face_ax.set_title(f"Face motion - {summary.exp_id}", pad=10)
            lock_ax.set_title(f"Lock state - {summary.exp_id}", pad=10)
            self._update_timebase_warning(summary, payload)
            self._update_movie_locomotion_warning(summary, payload)
            if self.timebase_warning.isVisible() and payload.t.size:
                video_end = float(payload.t[-1])
                for ax in (pupil_ax, loc_ax, face_ax, lock_ax):
                    ax.axvline(video_end, color="0.45", linestyle="--", linewidth=1.1, alpha=0.8, label="session end")
            if payload.t.size:
                self._make_cursor_lines(pupil_ax, loc_ax, face_ax, lock_ax)
                self._set_cursor_position(float(payload.t[0]))
            if payload.has_pupil and payload.radius.size:
                self._make_threshold_lines(pupil_ax)
            else:
                self._threshold_lines = []
            bounds = self._trace_bounds_from_arrays(payload.t, payload.locomotion_t, payload.brake_t)
        else:
            pupil_t, z, loc_t, loc, face_t, face, lock_t, lock, spans, labels = self._build_scope_trace()
            if pupil_t.size and z.size:
                pupil_ax.plot(pupil_t[: z.size], z, color="black", linewidth=1.2)
            else:
                pupil_ax.text(0.5, 0.5, "No pupil data to display", transform=pupil_ax.transAxes, ha="center", va="center")
            if loc_t.size and loc.size:
                loc_ax.plot(loc_t[: loc.size], loc, color="tab:blue", linewidth=1.2)
            else:
                loc_ax.text(0.5, 0.5, "No locomotion data to display", transform=loc_ax.transAxes, ha="center", va="center")
            if face_t.size and face.size:
                face_ax.plot(face_t[: face.size], face, color="tab:purple", linewidth=1.1)
            else:
                face_ax.text(0.5, 0.5, "No face motion data to display", transform=face_ax.transAxes, ha="center", va="center")
            self._plot_lock_state(lock_ax, lock_t, lock)
            for start_t, end_t, label in spans:
                pupil_ax.axvline(start_t, color="0.65", linestyle=":", linewidth=0.8)
                loc_ax.axvline(start_t, color="0.65", linestyle=":", linewidth=0.8)
                lock_ax.axvline(start_t, color="0.65", linestyle=":", linewidth=0.8)
                if len(spans) <= 20:
                    pupil_ax.text(start_t, 0.98, label, transform=pupil_ax.get_xaxis_transform(), rotation=90, fontsize=8, va="top")
            pupil_ax.set_title(f"Pupil dynamics - {self.animal_id} overall", pad=10)
            loc_ax.set_title(f"Locomotion - {self.animal_id} overall", pad=10)
            face_ax.set_title(f"Face motion - {self.animal_id} overall", pad=10)
            lock_ax.set_title(f"Lock state - {self.animal_id} overall", pad=10)
            self._update_timebase_warning(None, None)
            self._update_movie_locomotion_warning(None, None)
            self._make_threshold_lines(pupil_ax)
            bounds = self._trace_bounds_from_arrays(pupil_t, loc_t, face_t, lock_t)
        self._refresh_axis_legend(pupil_ax)
        self._refresh_axis_legend(loc_ax)
        self._refresh_axis_legend(face_ax)
        self._refresh_axis_legend(lock_ax)
        self._apply_trace_bounds(bounds)
        self.canvas.draw_idle()
    def _refresh_axis_legend(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", frameon=False)
    def _update_video(self):
        show_video = self.view_mode == "Session"
        self.mask_group.setVisible(show_video)
        self.calibration_group.setVisible(show_video)
        if not show_video:
            self.video_widget.pause()
            self.video_widget.set_video(None, None)
            self.interval_list.clear()
            self.video_time_label.setText("Current video time: --")
            self._clear_cursor_lines()
            self._update_movie_locomotion_warning(None, None)
            return
        summary = self.store.get_session_summary(self.exp_id)
        if summary is None:
            self.video_widget.set_video(None, None)
            self.mask_group.setEnabled(False)
            self._clear_cursor_lines()
            self._update_movie_locomotion_warning(None, None)
            return
        raw_payload = self._current_payload if self._current_payload and self._current_payload.summary.exp_id == summary.exp_id else self._load_session_payload(summary)
        self._current_payload = raw_payload
        payload = self._analysis_payload(raw_payload, summary)
        if summary.has_right_video and summary.right_video and Path(summary.right_video).exists():
            self.video_widget.set_video(summary.right_video, payload.t, overlay=payload.video_overlay())
            self.mask_group.setEnabled(bool(payload.has_pupil))
            if payload.t.size:
                self._set_cursor_position(float(self.video_widget.current_time()))
        else:
            self.video_widget.set_video(None, None)
            self.mask_group.setEnabled(False)
            self._clear_cursor_lines()
        self._update_movie_locomotion_warning(summary, payload)
    def _refresh_interval_list(self):
        self.interval_list.clear()
        if self.view_mode != "Session" or not self.exp_id:
            return
        for start, end in self.store.load_manual_masks(self.exp_id):
            self._add_interval_item(start, end)
        self._pending_mask_start = None
    def _add_interval_item(self, start: float, end: float):
        label = f"{format_seconds(start)}  ->  {format_seconds(end)}"
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, (float(start), float(end)))
        self.interval_list.addItem(item)
    def _set_interval_start(self):
        if self.view_mode != "Session" or self.video_widget.current_time() is None:
            return
        self._pending_mask_start = float(self.video_widget.current_time())
        self.video_time_label.setText(f"Start set at {format_seconds(self._pending_mask_start)}")
    def _set_interval_end(self):
        if self.view_mode != "Session" or self._pending_mask_start is None:
            return
        end = float(self.video_widget.current_time())
        start = float(self._pending_mask_start)
        if end < start:
            start, end = end, start
        self._add_interval_item(start, end)
        self._pending_mask_start = None
        self.video_time_label.setText(f"Added interval {format_seconds(start)} -> {format_seconds(end)}")
    def _delete_selected_interval(self):
        row = self.interval_list.currentRow()
        if row < 0:
            return
        self.interval_list.takeItem(row)
    def _clear_intervals(self):
        self.interval_list.clear()
        self._pending_mask_start = None
    def _save_intervals(self):
        if self.view_mode != "Session" or not self.exp_id:
            return
        intervals = []
        for row in range(self.interval_list.count()):
            item = self.interval_list.item(row)
            start, end = item.data(Qt.UserRole)
            intervals.append((float(start), float(end)))
        intervals = merge_intervals(intervals)
        self.store.save_manual_masks(self.exp_id, intervals, mark_dirty=True)
        self._refresh_interval_list()
        self.masks_changed.emit()
        self.refresh()
    def _format_extra_large_calibration_text(self, calibration: dict | None) -> str:
        if calibration is None:
            return "No confirmed calibration. Select a period or use the suggestion, then confirm to train the extra-large similarity rule."
        start, end = calibration["interval"]
        parts = [f"Confirmed: {format_seconds(float(start))} -> {format_seconds(float(end))}"]
        details = self._extra_large_calibration_details(calibration)
        if details:
            parts.append(details)
        if calibration.get("legacy", False):
            parts.append("legacy calibration, please reconfirm")
        return " | ".join(parts)
    def _calibration_selection_interval(self) -> tuple[float, float] | None:
        if self._calibration_selection_start is None or self._calibration_selection_end is None:
            return None
        start = float(self._calibration_selection_start)
        end = float(self._calibration_selection_end)
        if not np.isfinite(start) or not np.isfinite(end) or end == start:
            return None
        if end < start:
            start, end = end, start
        return (start, end)
    def _extra_large_calibration_details(self, calibration: dict | None) -> str:
        if calibration is None:
            return ""
        parts: list[str] = []
        band = calibration.get("reference_band")
        if band is not None and len(band) >= 2:
            parts.append(f"pupil band={float(band[0]):.2f}..{float(band[1]):.2f}")
        ref_mean = calibration.get("reference_mean")
        ref_std = calibration.get("reference_std")
        if ref_mean is not None and ref_std is not None:
            parts.append(f"reference mean±spread={float(ref_mean):.2f}±{float(ref_std):.2f}")
        similarity_cutoff = calibration.get("similarity_cutoff")
        if similarity_cutoff is not None:
            parts.append(f"similarity cutoff={float(similarity_cutoff):.3f}")
        selected_frames = calibration.get("selected_frames")
        if selected_frames is not None:
            parts.append(f"{int(selected_frames)} frames")
        return " | ".join(parts)
    def _calibration_from_interval(
        self,
        summary: SessionSummary,
        payload: LoadedSession,
        interval: tuple[float, float],
    ) -> dict | None:
        brightness_t, brightness = self.store.load_pupil_brightness(summary.exp_id)
        if brightness_t is None or brightness is None or not brightness.size:
            return None
        reference = learn_extra_large_reference_band(np.asarray(brightness_t, dtype=float), np.asarray(brightness, dtype=float), interval)
        if reference is None:
            return None
        similarity_t, similarity = self.store.load_eye_similarity(summary.exp_id, reference["reference_band"])
        if similarity_t is None or similarity is None or not similarity.size:
            return None
        n = min(int(np.asarray(payload.t, dtype=float).size), int(np.asarray(brightness, dtype=float).size), int(np.asarray(similarity, dtype=float).size))
        if n <= 0:
            return None
        calibration = learn_extra_large_calibration(
            np.asarray(brightness_t, dtype=float)[:n],
            np.asarray(brightness, dtype=float)[:n],
            np.asarray(similarity, dtype=float)[:n],
            interval,
        )
        if calibration is None:
            return None
        calibration["selected_exp_id"] = summary.exp_id
        return calibration
    def _resolve_extra_large_calibration(
        self,
        calibration: dict | None,
        summary: SessionSummary,
        payload: LoadedSession,
    ) -> dict | None:
        if calibration is None:
            return None
        if calibration.get("reference_band") is not None and calibration.get("similarity_cutoff") is not None:
            return calibration
        interval = calibration.get("interval")
        if interval is None:
            return calibration
        resolved = self._calibration_from_interval(summary, payload, interval)
        if resolved is None:
            return calibration
        resolved["confirmed"] = bool(calibration.get("confirmed", False))
        if calibration.get("legacy", False):
            resolved["legacy"] = True
        if calibration.get("selected_exp_id") is not None:
            resolved["selected_exp_id"] = calibration["selected_exp_id"]
        if calibration.get("manual_selection") is not None:
            resolved["manual_selection"] = calibration["manual_selection"]
        if calibration.get("candidate_source") is not None:
            resolved["candidate_source"] = calibration["candidate_source"]
        return resolved
    def _selected_extra_large_calibration(self) -> dict | None:
        if self.view_mode != "Session" or self.exp_id == "":
            return None
        interval = self._calibration_selection_interval()
        if interval is None:
            return None
        summary = self._current_session or self.store.get_session_summary(self.exp_id)
        if summary is None:
            return None
        raw_payload = self._current_payload if self._current_payload is not None and self._current_payload.summary.exp_id == summary.exp_id else self._load_session_payload(summary)
        payload = self._analysis_payload(raw_payload, summary)
        calibration = self._calibration_from_interval(summary, payload, interval)
        if calibration is None:
            return None
        calibration["selected_exp_id"] = summary.exp_id
        calibration["manual_selection"] = True
        return calibration
    def _active_extra_large_calibration(self) -> dict | None:
        if self.view_mode != "Session" or self.exp_id == "":
            return None
        summary = self._current_session or self.store.get_session_summary(self.exp_id)
        if summary is None:
            return None
        raw_payload = self._current_payload if self._current_payload is not None and self._current_payload.summary.exp_id == summary.exp_id else self._load_session_payload(summary)
        payload = self._analysis_payload(raw_payload, summary)
        selected = self._selected_extra_large_calibration()
        if selected is not None:
            return selected
        confirmed = self.store.session_extra_large_calibration(self.exp_id)
        confirmed = self._resolve_extra_large_calibration(confirmed, summary, payload) if confirmed is not None else None
        if confirmed is not None:
            return confirmed
        suggestion = self._calibration_suggestion
        if suggestion is not None:
            suggestion = self._resolve_extra_large_calibration(suggestion, summary, payload)
        return suggestion
    def _set_extra_large_calibration_start(self):
        if self.view_mode != "Session" or self.exp_id == "":
            return
        current_time = self.video_widget.current_time()
        if current_time is None or not np.isfinite(current_time):
            return
        self._calibration_selection_start = float(current_time)
        self.refresh()
    def _set_extra_large_calibration_end(self):
        if self.view_mode != "Session" or self.exp_id == "":
            return
        current_time = self.video_widget.current_time()
        if current_time is None or not np.isfinite(current_time):
            return
        self._calibration_selection_end = float(current_time)
        self.refresh()
    def _clear_extra_large_calibration_selection(self):
        if self.view_mode != "Session" or self.exp_id == "":
            return
        self._calibration_selection_start = None
        self._calibration_selection_end = None
        self.refresh()
    def _go_to_extra_large_calibration(self):
        calibration = self._active_extra_large_calibration()
        if calibration is None:
            return
        interval = calibration.get("interval")
        if not interval:
            return
        self.video_widget.seek_time(float(interval[0]))
    def _refresh_extra_large_calibration_controls(
        self,
        summary: SessionSummary | None,
        payload: LoadedSession | None,
        pupil_ax=None,
    ):
        if not hasattr(self, "calibration_group"):
            return
        if self.view_mode != "Session" or summary is None or payload is None or not payload.has_pupil or not payload.radius.size:
            self._calibration_suggestion = None
            self.calibration_status_label.setText("No calibration suggestion available for this session.")
            self.calibration_selection_label.setText("Selected period: none")
            self.set_calibration_start_btn.setEnabled(False)
            self.set_calibration_end_btn.setEnabled(False)
            self.clear_calibration_selection_btn.setEnabled(False)
            self.goto_calibration_btn.setEnabled(False)
            self.confirm_calibration_btn.setEnabled(False)
            self.clear_calibration_btn.setEnabled(False)
            return
        confirmed_raw = self.store.session_extra_large_calibration(summary.exp_id)
        confirmed = self._resolve_extra_large_calibration(confirmed_raw, summary, payload) if confirmed_raw is not None else None
        brightness_t, brightness = self.store.load_pupil_brightness(summary.exp_id)
        raw_suggestion = None
        if brightness_t is not None and brightness is not None and brightness.size:
            raw_suggestion = suggest_extra_large_calibration(
                np.asarray(payload.t, dtype=float),
                np.asarray(payload.radius, dtype=float),
                np.asarray(payload.visible_mask_base, dtype=bool),
                np.asarray(brightness, dtype=float),
                self._zscore_mean,
                self._zscore_std,
                self._threshold_values,
            )
        suggestion = self._resolve_extra_large_calibration(raw_suggestion, summary, payload) if raw_suggestion is not None else None
        self._calibration_suggestion = suggestion
        selection_interval = self._calibration_selection_interval()
        selection = self._selected_extra_large_calibration()
        status_lines = []
        status_lines.append(self._format_extra_large_calibration_text(confirmed))
        if selection is not None:
            start, end = selection["interval"]
            details = self._extra_large_calibration_details(selection)
            status_lines.append(
                f"Selected: {format_seconds(float(start))} -> {format_seconds(float(end))}" + (f" | {details}" if details else "")
            )
            status_lines.append("This selection is not active until you confirm it.")
        elif suggestion is None:
            status_lines.append("Suggested: no suitable big-detected interval was found.")
        else:
            start, end = suggestion["interval"]
            details = self._extra_large_calibration_details(suggestion)
            status_lines.append(
                f"Suggested: {format_seconds(float(start))} -> {format_seconds(float(end))}" + (f" | {details}" if details else "")
            )
            if confirmed is not None and tuple(map(float, confirmed["interval"])) != tuple(map(float, suggestion["interval"])):
                status_lines.append("The suggestion differs from the confirmed calibration. Confirm again to update it.")
            elif confirmed is None:
                status_lines.append("This suggestion is not active until you confirm it.")
        confirm_enabled = selection is not None or suggestion is not None
        self.calibration_status_label.setText("\n".join(status_lines))
        if selection is not None:
            details = self._extra_large_calibration_details(selection)
            self.calibration_selection_label.setText(
                f"Selected period: {format_seconds(float(selection['interval'][0]))} -> {format_seconds(float(selection['interval'][1]))}" + (f" | {details}" if details else "")
            )
        elif self._calibration_selection_start is not None and self._calibration_selection_end is None:
            self.calibration_selection_label.setText(
                f"Selected period: start at {format_seconds(float(self._calibration_selection_start))}; end not set yet"
            )
        elif self._calibration_selection_start is None and self._calibration_selection_end is not None:
            self.calibration_selection_label.setText(
                f"Selected period: end at {format_seconds(float(self._calibration_selection_end))}; start not set yet"
            )
        elif selection_interval is not None:
            self.calibration_selection_label.setText(
                f"Selected period: {format_seconds(float(selection_interval[0]))} -> {format_seconds(float(selection_interval[1]))} | ready to confirm similarity rule"
            )
        else:
            self.calibration_selection_label.setText("Selected period: none")
        self.set_calibration_start_btn.setEnabled(True)
        self.set_calibration_end_btn.setEnabled(True)
        self.clear_calibration_selection_btn.setEnabled(
            self._calibration_selection_start is not None or self._calibration_selection_end is not None
        )
        self.goto_calibration_btn.setEnabled(confirmed is not None or selection is not None or suggestion is not None)
        if selection is not None:
            self.goto_calibration_btn.setToolTip(
                f"Jump the video player to the selected calibration start at {format_seconds(float(selection['interval'][0]))}."
            )
        elif confirmed is not None:
            self.goto_calibration_btn.setToolTip(
                f"Jump the video player to the confirmed calibration start at {format_seconds(float(confirmed['interval'][0]))}."
            )
        elif suggestion is not None:
            self.goto_calibration_btn.setToolTip(
                f"Jump the video player to the suggested calibration start at {format_seconds(float(suggestion['interval'][0]))}."
            )
        else:
            self.goto_calibration_btn.setToolTip("No calibration interval is available for this session.")
        self.confirm_calibration_btn.setEnabled(confirm_enabled)
        self.clear_calibration_btn.setEnabled(confirmed is not None)
        if pupil_ax is not None and summary is not None and payload is not None and payload.has_pupil and payload.radius.size:
            if confirmed is not None:
                start, end = confirmed["interval"]
                pupil_ax.axvspan(start, end, color="tab:blue", alpha=0.12, label="confirmed calibration")
            if selection is not None:
                start, end = selection["interval"]
                pupil_ax.axvspan(start, end, color="tab:orange", alpha=0.14, label="selected calibration")
            if suggestion is not None:
                start, end = suggestion["interval"]
                label = "suggested calibration" if confirmed is None else "suggested calibration (pending)"
                pupil_ax.axvspan(start, end, color="tab:cyan", alpha=0.12, label=label)
    def _confirm_extra_large_calibration(self):
        if self.view_mode != "Session" or self.exp_id == "":
            return
        calibration = self._selected_extra_large_calibration()
        if calibration is None:
            calibration = self._calibration_suggestion
        if not calibration:
            return
        self.store.set_session_extra_large_calibration(self.exp_id, calibration, confirmed=True)
        self.session_state_changed.emit()
        self.refresh()
    def _clear_extra_large_calibration(self):
        if self.view_mode != "Session" or self.exp_id == "":
            return
        self._calibration_selection_start = None
        self._calibration_selection_end = None
        self.store.clear_session_extra_large_calibration(self.exp_id)
        self.session_state_changed.emit()
        self.refresh()
    def _on_video_time_changed(self, value: float):
        self.video_time_label.setText(f"Current video time: {format_seconds(value)}")
        self._set_cursor_position(value)
class StatisticsTab(QWidget):
    def __init__(self, store: HabituationStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.animal_id = "All"
        self.view_mode = "Overall"
        self._worker: TaskThread | None = None
        self._prompted_for_entry = False
        self._current_result = None
        self._current_paths: tuple[Path, Path, Path] | None = None
        self._plot_pages: list[tuple[str, Path]] = []
        self._plot_index = 0
        self._scope_label = QLabel("Statistics are not running yet.", self)
        self._scope_label.setWordWrap(True)
        self._status_label = QLabel("Open this tab to run or review statistics.", self)
        self._status_label.setWordWrap(True)
        self._paths_label = QLabel("", self)
        self._paths_label.setWordWrap(True)
        self._current_figure_pixmap = QtGui.QPixmap()
        self.log_toggle = QCheckBox("Show log window", self)
        self.log_toggle.setChecked(True)
        self.log_toggle.toggled.connect(self._set_log_visible)
        self.log_toggle.setToolTip("Show or hide the statistics log and file paths.")
        self.summary_edit = QPlainTextEdit(self)
        self.summary_edit.setReadOnly(True)
        self.summary_edit.setPlaceholderText("Statistics output will appear here after the analysis runs.")
        self.summary_edit.setMinimumHeight(220)
        self.log_panel = QWidget(self)
        log_layout = QVBoxLayout(self.log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(self._paths_label)
        log_layout.addWidget(self.summary_edit, stretch=1)
        self._plot_title_label = QLabel("No plot selected.", self)
        self._plot_title_label.setAlignment(Qt.AlignCenter)
        self._plot_title_label.setWordWrap(True)
        self.figure_label = QLabel("No statistics figure yet.", self)
        self.figure_label.setAlignment(Qt.AlignCenter)
        self.figure_label.setMinimumSize(0, 0)
        self.figure_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.figure_scroll = QScrollArea(self)
        self.figure_scroll.setWidgetResizable(True)
        self.figure_scroll.setWidget(self.figure_label)
        self._plot_prev_button = QToolButton(self)
        self._plot_prev_button.setArrowType(Qt.LeftArrow)
        self._plot_prev_button.clicked.connect(lambda: self._step_plot(-1))
        self._plot_next_button = QToolButton(self)
        self._plot_next_button.setArrowType(Qt.RightArrow)
        self._plot_next_button.clicked.connect(lambda: self._step_plot(1))
        self._plot_counter_label = QLabel("", self)
        self._plot_counter_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.run_button = QPushButton("Run statistics now", self)
        self.run_button.clicked.connect(self.run_current_scope)
        self.run_all_button = QPushButton("Run all animals", self)
        self.run_all_button.clicked.connect(self.run_all_animals)
        self.run_all_button.setToolTip("Run statistics separately for every animal in the dataset.")
        button_row = QHBoxLayout()
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.run_all_button)
        button_row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(self._scope_label)
        layout.addWidget(self._status_label)
        layout.addLayout(button_row)
        layout.addWidget(self.log_toggle, alignment=Qt.AlignLeft)
        layout.addWidget(self.log_panel, stretch=1)
        layout.addWidget(self._plot_title_label)
        layout.addWidget(self.figure_scroll, stretch=2)
        plot_nav = QHBoxLayout()
        plot_nav.addWidget(self._plot_prev_button)
        plot_nav.addWidget(self._plot_next_button)
        plot_nav.addStretch(1)
        plot_nav.addWidget(self._plot_counter_label)
        layout.addLayout(plot_nav)
    def set_context(self, animal_id: str, view_mode: str):
        changed = animal_id != self.animal_id or view_mode != self.view_mode
        self.animal_id = animal_id
        self.view_mode = view_mode
        self._scope_label.setText(f"Statistics scope: {animal_id} | View: {view_mode}")
        if changed:
            self._prompted_for_entry = False
        self._reload_current_context(changed=changed)
    def refresh(self):
        self._reload_current_context(changed=True)
    def _reload_current_context(self, *, changed: bool):
        if self._worker is not None and self._worker.isRunning():
            return
        if not changed:
            return
        self._set_placeholder(f"Loading statistics for {self.animal_id}...")
        QApplication.processEvents()
        if self._show_cached_statistics_for_context():
            self._prompted_for_entry = True
        else:
            self._set_placeholder(f"Statistics for {self.animal_id} are not running yet.")
    def _set_placeholder(self, message: str):
        self._status_label.setText(message)
        self.summary_edit.setPlainText(message)
        self._paths_label.setText("")
        self._current_result = None
        self._current_paths = None
        self._set_plot_pages([])
    def _set_log_visible(self, visible: bool):
        self.log_panel.setVisible(bool(visible))
    def _set_plot_pages(self, pages: list[tuple[str, Path]]):
        self._plot_pages = [(title, Path(path)) for title, path in pages if Path(path).exists()]
        self._plot_index = 0
        self._show_current_plot_page()
    def _load_plot_pages_from_output_dir(self, output_dir: Path):
        pages: list[tuple[str, Path]] = []
        for title, path in statistics_panel_paths(output_dir):
            path = Path(path)
            if path.exists():
                pages.append((title, path))
        if not pages:
            summary_path = Path(output_dir) / "statistics_summary.png"
            if summary_path.exists():
                pages.append(("Statistics summary", summary_path))
        self._set_plot_pages(pages)
    def _show_current_plot_page(self):
        if not self._plot_pages:
            self._plot_title_label.setText("No plot selected.")
            self._plot_counter_label.setText("")
            self.figure_label.setText("No statistics figure yet.")
            self.figure_label.setPixmap(QtGui.QPixmap())
            self._current_figure_pixmap = QtGui.QPixmap()
            self._update_plot_controls()
            return
        title, path = self._plot_pages[self._plot_index]
        pixmap = QtGui.QPixmap(str(path))
        if pixmap.isNull():
            self._plot_title_label.setText(title)
            self._plot_counter_label.setText("")
            self.figure_label.setText("Statistics figure could not be loaded.")
            self.figure_label.setPixmap(QtGui.QPixmap())
            self._current_figure_pixmap = QtGui.QPixmap()
            self._update_plot_controls()
            return
        self._plot_title_label.setText(title)
        self._current_figure_pixmap = QtGui.QPixmap(pixmap)
        self.figure_label.setText("")
        self._refresh_figure_preview()
        self._update_plot_controls()
    def _update_plot_controls(self):
        enabled = len(self._plot_pages) > 1
        self._plot_prev_button.setEnabled(enabled)
        self._plot_next_button.setEnabled(enabled)
        if self._plot_pages:
            self._plot_counter_label.setText(f"{self._plot_index + 1}/{len(self._plot_pages)}")
        else:
            self._plot_counter_label.setText("")
    def _step_plot(self, delta: int):
        if not self._plot_pages:
            return
        self._plot_index = (self._plot_index + int(delta)) % len(self._plot_pages)
        self._show_current_plot_page()
    def _display_statistics_payload(self, payload: dict, *, status_prefix: str | None = None):
        result = payload["result"]
        result_path, svg_path, png_path = payload["paths"]
        self._current_result = result
        self._current_paths = (result_path, svg_path, png_path)
        self.store.clear_animal_dirty(result.animal_id)
        prefix = status_prefix or ("Loaded cached statistics for" if payload.get("cached") else "Statistics complete for")
        self._status_label.setText(f"{prefix} {result.scope} / {result.animal_id}")
        self._paths_label.setText(f"Saved: {result_path}\nSVG: {svg_path}\nPNG: {png_path}")
        self.summary_edit.setPlainText(self._format_result_text(result))
        self._load_plot_pages_from_output_dir(Path(result_path).parent)
    def _show_cached_statistics_for_context(self) -> bool:
        if self._worker is not None and self._worker.isRunning():
            return False
        scope = self.animal_id
        thresholds = self._threshold_payload(*self._threshold_inputs_for_animal(scope))
        cached = load_cached_statistics_outputs(
            self.store,
            scope=scope,
            animal_id=scope,
            thresholds=thresholds,
        )
        if cached is None:
            return False
        result, result_paths = cached
        self._display_statistics_payload({"result": result, "paths": result_paths, "cached": True})
        return True
    def _set_figure_preview(self, pixmap: QtGui.QPixmap):
        self._current_figure_pixmap = QtGui.QPixmap(pixmap)
        self._refresh_figure_preview()
    def _refresh_figure_preview(self):
        if self._current_figure_pixmap.isNull():
            return
        viewport = self.figure_scroll.viewport()
        if viewport is None:
            self.figure_label.setPixmap(self._current_figure_pixmap)
            return
        target = viewport.size()
        if target.width() <= 0 or target.height() <= 0:
            self.figure_label.setPixmap(self._current_figure_pixmap)
            return
        scaled = self._current_figure_pixmap.scaled(
            target,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.figure_label.setPixmap(scaled)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_figure_preview()
    def maybe_prompt_and_run(self):
        if self._worker is not None and self._worker.isRunning():
            return
        if self._prompted_for_entry:
            return
        self._prompted_for_entry = True
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            self.run_current_scope()
            return
        answer = QMessageBox.question(
            self,
            "Run statistics?",
            "Do you want to run the statistical analysis for the current scope?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self.run_current_scope()
    def _set_statistics_running(self, message: str):
        self._status_label.setText(message)
        self.run_button.setEnabled(False)
        self.run_all_button.setEnabled(False)
    def _finish_statistics_run(self):
        self.run_button.setEnabled(True)
        self.run_all_button.setEnabled(True)
        self._worker = None
    def _start_statistics_worker(self, job, on_result):
        self._worker = TaskThread(job, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()
    def _ask_all_animals_mode(self) -> str | None:
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return "all"
        box = QMessageBox(self)
        box.setWindowTitle("Run all animals")
        box.setText("Do you want to run statistics for all animals or only the ones without saved stats?")
        all_button = box.addButton("All animals", QMessageBox.AcceptRole)
        missing_button = box.addButton("Only missing stats", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(all_button)
        box.exec_()
        clicked = box.clickedButton()
        if clicked == all_button:
            return "all"
        if clicked == missing_button:
            return "missing"
        return None
    def run_current_scope(self):
        if self._worker is not None and self._worker.isRunning():
            return
        scope = self.animal_id
        self._set_statistics_running(f"Running statistics for {scope}...")
        self._start_statistics_worker(self._build_statistics_job(scope), self._on_result)
    def run_all_animals(self):
        if self._worker is not None and self._worker.isRunning():
            return
        animals = self._analysis_animals()
        if not animals:
            self._status_label.setText("No animals with analysis sessions were found.")
            return
        if "All" not in animals:
            animals.append("All")
        mode = self._ask_all_animals_mode()
        if mode is None:
            return
        mode_label = "all animals + overall" if mode == "all" else "animals without saved stats + overall"
        self._set_statistics_running(f"Running statistics for {len(animals)} scopes ({mode_label})...")
        self._start_statistics_worker(self._build_all_statistics_job(animals, mode), self._on_batch_result)
    @staticmethod
    def _threshold_payload(percentiles, threshold_values, locomotion_threshold, missing_buffer_sec) -> dict:
        return {
            "percentiles": [float(v) for v in percentiles],
            "threshold_values": [float(v) for v in threshold_values],
            "locomotion_threshold": float(locomotion_threshold),
            "missing_buffer_sec": float(missing_buffer_sec),
        }
    def _build_statistics_job(self, scope: str):
        percentiles, threshold_values, locomotion_threshold, missing_buffer_sec = self._threshold_inputs_for_animal(scope)
        thresholds = self._threshold_payload(percentiles, threshold_values, locomotion_threshold, missing_buffer_sec)
        def job(progress_cb):
            cached = load_cached_statistics_outputs(
                self.store,
                scope=scope,
                animal_id=scope,
                thresholds=thresholds,
            )
            if cached is not None:
                result, result_paths = cached
                return {"result": result, "paths": result_paths, "cached": True}
            result = compute_statistics(
                self.store,
                scope=scope,
                animal_id=scope,
                percentiles=percentiles,
                threshold_values=threshold_values,
                locomotion_threshold=locomotion_threshold,
                missing_buffer_sec=missing_buffer_sec,
                progress_cb=progress_cb,
            )
            result_paths = save_statistics_outputs(self.store, result)
            return {"result": result, "paths": result_paths, "cached": False}
        return job
    def _build_all_statistics_job(self, animals: list[str], mode: str):
        threshold_inputs = {animal: self._threshold_inputs_for_animal(animal) for animal in animals}
        def job(progress_cb):
            outputs = []
            skipped = []
            total = max(1, len(animals))
            for idx, animal in enumerate(animals):
                percentiles, threshold_values, locomotion_threshold, missing_buffer_sec = threshold_inputs[animal]
                thresholds = self._threshold_payload(percentiles, threshold_values, locomotion_threshold, missing_buffer_sec)
                if animal != "All":
                    cached = load_cached_statistics_outputs(
                        self.store,
                        scope=animal,
                        animal_id=animal,
                        thresholds=thresholds,
                    )
                    if cached is not None and mode == "missing":
                        result, result_paths = cached
                        skipped.append({"animal_id": animal, "result": result, "paths": result_paths, "cached": True})
                        progress_cb((idx + 1) / total, f"{animal}: cached statistics already exist")
                        continue
                    if cached is not None:
                        result, result_paths = cached
                        outputs.append({"animal_id": animal, "result": result, "paths": result_paths, "cached": True})
                        progress_cb((idx + 1) / total, f"{animal}: loaded cached statistics")
                        continue
                def animal_progress(fraction: float, message: str, *, idx=idx, animal=animal):
                    overall = (idx + float(fraction)) / total
                    progress_cb(overall, f"{animal}: {message}")
                result = compute_statistics(
                    self.store,
                    scope=animal,
                    animal_id=animal,
                    percentiles=percentiles,
                    threshold_values=threshold_values,
                    locomotion_threshold=locomotion_threshold,
                    missing_buffer_sec=missing_buffer_sec,
                    progress_cb=animal_progress,
                )
                result_paths = save_statistics_outputs(self.store, result)
                outputs.append({"animal_id": animal, "result": result, "paths": result_paths, "cached": False})
            return {"mode": "all_animals", "selection_mode": mode, "results": outputs, "skipped": skipped}
        return job
    def _analysis_animals(self) -> list[str]:
        animals = []
        for animal in self.store.animals():
            if any(
                s.has_right_pickle and not self.store.is_deeplabcut_reference_session(s.exp_id) and not self.store.is_session_do_not_use(s.exp_id)
                for s in self.store.sessions_for_animal(animal)
            ):
                animals.append(animal)
        return animals
    def _threshold_inputs_for_animal(self, animal_id: str):
        global_percentiles = self.store.global_pupil_percentile_cutoffs()
        locomotion_threshold = float(self.store.settings.get("global", {}).get("locomotion_threshold", 0.35))
        missing_buffer_sec = float(self.store.global_pupil_missing_buffer_sec())
        if self.parent() is not None:
            main_window = self.window()
            metrics = getattr(main_window, "metrics_tab", None)
            if (
                metrics is not None
                and metrics._metrics_available()
                and metrics.animal_id == animal_id
                and metrics._percentile_cutoffs
                and metrics._threshold_values
            ):
                return (
                    list(metrics._percentile_cutoffs),
                    list(metrics._threshold_values),
                    locomotion_threshold,
                    float(metrics.missing_buffer_spin.value()),
                )
        if animal_id == "All":
            mean, std = compute_animal_baseline(self.store, "All", scope="All")
            zmap = animal_zscores(self.store, "All", mean=mean, std=std)
            pooled = [values[np.isfinite(values)] for values in zmap.values() if np.any(np.isfinite(values))]
            distribution = np.concatenate(pooled) if pooled else np.array([], dtype=float)
            threshold_values = percentile_threshold_values(distribution, global_percentiles) if distribution.size else [0.0, 0.0, 0.0]
            return list(map(float, global_percentiles)), list(map(float, threshold_values)), locomotion_threshold, missing_buffer_sec
        settings = self.store.get_animal_settings(animal_id)
        threshold_values = settings.get("threshold_values", [0.0, 0.5, 1.0])
        threshold_signature = str(settings.get("threshold_signature", ""))
        global_signature = self.store.pupil_percentile_signature(global_percentiles)
        if not isinstance(threshold_values, list) or len(threshold_values) != 3 or threshold_signature != global_signature:
            mean, std = compute_animal_baseline(self.store, animal_id)
            zmap = animal_zscores(self.store, animal_id, mean=mean, std=std)
            pooled = [values[np.isfinite(values)] for values in zmap.values() if np.any(np.isfinite(values))]
            distribution = np.concatenate(pooled) if pooled else np.array([], dtype=float)
            threshold_values = percentile_threshold_values(distribution, global_percentiles) if distribution.size else [0.0, 0.0, 0.0]
            settings["threshold_values"] = [float(v) for v in threshold_values]
            settings["threshold_signature"] = global_signature
            self.store.set_animal_settings(animal_id, settings)
        return list(map(float, global_percentiles)), list(map(float, threshold_values)), locomotion_threshold, missing_buffer_sec
    def _current_threshold_inputs(self):
        return self._threshold_inputs_for_animal(self.animal_id)
    def _on_progress(self, fraction: float, message: str):
        self._status_label.setText(f"{message} ({fraction * 100.0:.0f}%)")
    def _on_error(self, message: str):
        self._finish_statistics_run()
        self._status_label.setText("Statistics failed.")
        self._set_message_box_error(message)
    def _set_message_box_error(self, message: str):
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            print(message)
            return
        QMessageBox.critical(self, "Statistics error", message)
    def _on_result(self, payload: dict):
        self._finish_statistics_run()
        self._display_statistics_payload(payload)
    def _on_batch_result(self, payload: dict):
        self._finish_statistics_run()
        results = payload.get("results", [])
        skipped = payload.get("skipped", [])
        if not results and not skipped:
            self._status_label.setText("No batch statistics were generated.")
            self.summary_edit.setPlainText("No batch statistics were generated.")
            self._paths_label.setText("")
            self._set_plot_pages([])
            self._current_result = None
            self._current_paths = None
            return
        for item in results + skipped:
            self.store.clear_animal_dirty(item["animal_id"])
        if results:
            loaded_count = sum(1 for item in results if item.get("cached"))
            computed_count = len(results) - loaded_count
            skipped_count = len(skipped)
            status_bits = []
            if computed_count:
                status_bits.append(f"{computed_count} computed")
            if loaded_count:
                status_bits.append(f"{loaded_count} loaded from cache")
            if skipped_count:
                status_bits.append(f"{skipped_count} skipped because cached stats already exist")
            extra = f" ({'; '.join(status_bits)})" if status_bits else ""
            self._status_label.setText(f"Batch statistics complete for {len(results)} animals{extra}.")
        else:
            self._status_label.setText(f"No animals needed rerun; {len(skipped)} cached stats were already up to date.")
        active_items = results + skipped
        matching_item = next((item for item in active_items if item["animal_id"] == self.animal_id), None)
        if matching_item is None and self.animal_id == "All":
            matching_item = next((item for item in active_items if item["result"].scope == "All"), None)
        if matching_item is None and active_items:
            matching_item = active_items[0]
        if matching_item is None:
            return
        self._display_statistics_payload(
            {"result": matching_item["result"], "paths": matching_item["paths"], "cached": matching_item.get("cached", False)}
        )
        if len(active_items) > 1:
            batch_text = self._format_batch_result_text(payload, active_items, matching_item)
            self.summary_edit.setPlainText(batch_text)
    def _format_result_text(self, result) -> str:
        lines = [
            f"Scope: {result.scope}",
            f"Animal: {result.animal_id}",
            f"Generated at: {result.generated_at}",
            f"Sessions: {len(result.session_ids)}",
            f"Eligible sessions (>30 min usable duration): {len(result.eligible_session_ids)}",
            f"Baseline mean/std: {result.zscore_mean:.4f} / {result.zscore_std:.4f}",
            f"Thresholds: percentiles={result.thresholds.get('percentiles', [])} | values={result.thresholds.get('threshold_values', [])}",
            f"Locomotion threshold: {result.thresholds.get('locomotion_threshold', float('nan'))}",
            f"Missing pupil buffer: {result.thresholds.get('missing_buffer_sec', float('nan')):.2f} s",
            "",
            "Session-wise locomotion fraction:",
        ]
        for session in result.session_labels:
            lines.append(f"  {session}: {result.locomotion_pct_by_session.get(session, float('nan')):.4f}")
        lines.append("")
        lines.append("Session-wise face motion fraction:")
        for session in result.session_labels:
            lines.append(f"  {session}: {result.face_motion_pct_by_session.get(session, float('nan')):.4f}")
        lines.append("")
        lines.append("Session-wise pupil state fractions:")
        for session in result.session_labels:
            state_text = ", ".join(
                f"{label}={result.pupil_pct_by_session.get(session, {}).get(label, float('nan')):.4f}" for label in STATE_LABELS
            )
            lines.append(f"  {session}: {state_text}")
        return "\n".join(lines)
    def _format_batch_result_text(self, payload: dict, active_items: list[dict], selected_item: dict) -> str:
        results = payload.get("results", [])
        skipped = payload.get("skipped", [])
        skipped_ids = {str(item.get("animal_id", "")) for item in skipped}
        computed_ids = {str(item.get("animal_id", "")) for item in results if not item.get("cached")}
        loaded_ids = {str(item.get("animal_id", "")) for item in results if item.get("cached")}
        selected_id = str(selected_item.get("animal_id", ""))
        lines = [
            f"Batch mode: {payload.get('selection_mode', 'all')}",
            f"Animals processed: {len(active_items)}",
            "",
            "Animal IDs:",
        ]
        for item in active_items:
            animal_id = str(item.get("animal_id", ""))
            if animal_id in skipped_ids:
                status = "cached, skipped rerun"
            elif animal_id in loaded_ids:
                status = "loaded from cache"
            elif animal_id in computed_ids:
                status = "computed"
            else:
                status = "processed"
            lines.append(f"  - {animal_id}: {status}")
        lines.extend([
            "",
            f"Showing detailed statistics for: {selected_id}",
            "",
            self._format_result_text(selected_item["result"]),
        ])
        return "\n".join(lines)
class HabituationMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Habituation Analysis")
        self.resize(1720, 1080)
        self.setMinimumSize(1280, 820)
        self.store = HabituationStore()
        self.index = self.store.load_index(prefer_cache=True)
        self.app_state = self.store.load_app_state()
        self._last_session_exp_id = self.app_state.get("selected_exp_id", "")
        self._selection_refresh_in_progress = False
        if self._is_placeholder_saved_state():
            default_animal, default_exp = self._first_available_session()
            if default_animal != "All" and default_exp:
                self.app_state["selected_animal"] = default_animal
                self.app_state["selected_exp_id"] = default_exp
                self.app_state["view_mode"] = "Session"
                self._last_session_exp_id = default_exp
        self._updating_browser = False
        self._initializing = True
        self._worker: TaskThread | None = None
        self._stats_prompted_for_entry = False
        self.animal_combo = ScrollableComboBox(self, visible_rows=4)
        self.animal_combo.setMinimumWidth(240)
        self.animal_combo.setMinimumContentsLength(12)
        self.animal_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        animal_view = self.animal_combo.view()
        animal_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        animal_view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerItem)
        self.exp_combo = ScrollableComboBox(self, visible_rows=6)
        self.prev_btn = QPushButton("<", self)
        self.next_btn = QPushButton(">", self)
        self.update_btn = QPushButton("Update dataset", self)
        self.status_label = QLabel("", self)
        self.status_label.setWordWrap(True)
        self.animal_combo.currentIndexChanged.connect(self._on_browser_changed)
        self.exp_combo.currentIndexChanged.connect(self._on_browser_changed)
        self.prev_btn.clicked.connect(self._go_prev_session)
        self.next_btn.clicked.connect(self._go_next_session)
        self.update_btn.clicked.connect(self._update_dataset)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Animal", self))
        top_bar.addWidget(self.animal_combo)
        top_bar.addWidget(QLabel("ExpID", self))
        top_bar.addWidget(self.prev_btn)
        top_bar.addWidget(self.exp_combo, stretch=1)
        top_bar.addWidget(self.next_btn)
        top_bar.addWidget(self.update_btn)
        top_widget = QWidget(self)
        top_widget.setLayout(top_bar)
        self.tabs = QtWidgets.QTabWidget(self)
        self.metrics_tab = MetricsTab(self.store, self)
        self.statistics_tab = StatisticsTab(self.store, self)
        self.tabs.addTab(self.metrics_tab, "Metrics")
        self.tabs.addTab(self.statistics_tab, "Statistics")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        main_layout = QVBoxLayout()
        main_layout.addWidget(top_widget)
        main_layout.addWidget(self.tabs, stretch=1)
        main_layout.addWidget(self.status_label)
        central = QWidget(self)
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.metrics_tab.experiment_index.sessionActivated.connect(self._open_index_session)
        self.metrics_tab.reference_sessions.sessionActivated.connect(self._open_index_session)
        self.metrics_tab.thresholds_changed.connect(self._on_scope_changed)
        self.metrics_tab.thresholds_changed.connect(self.metrics_tab.experiment_index.refresh)
        self.metrics_tab.thresholds_changed.connect(self.metrics_tab.reference_sessions.refresh)
        self.metrics_tab.masks_changed.connect(self._on_scope_changed)
        self.metrics_tab.masks_changed.connect(self.metrics_tab.reference_sessions.refresh)
        self.metrics_tab.session_state_changed.connect(self.metrics_tab.experiment_index.refresh)
        self.metrics_tab.session_state_changed.connect(self.metrics_tab.reference_sessions.refresh)
        self.metrics_tab.session_state_changed.connect(self.statistics_tab.refresh)
        self._populate_browser(initial=True)
        self._apply_saved_state()
        self._initializing = False
        self._sync_browser_state()
        self.statusBar().showMessage(f"Loaded {len(self.index.sessions)} sessions from cache/source trees")
    def _animals_for_combo(self) -> list[str]:
        return ["All"] + self.store.animals()
    def _sessions_for_scope(self, animal_id: str) -> list[SessionSummary]:
        if animal_id == "All":
            return self.store.dataset_sessions()
        return self.store.sessions_for_animal(animal_id)
    def _populate_browser(self, initial: bool = False):
        animal = self.animal_combo.currentText() if self.animal_combo.count() else self.app_state.get("selected_animal", "All")
        exp = self.app_state.get("selected_exp_id", "")
        view = self.app_state.get("view_mode", "Session")
        blocker = QtCore.QSignalBlocker(self.animal_combo)
        self.animal_combo.clear()
        self.animal_combo.addItems(self._animals_for_combo())
        del blocker
        if animal not in [self.animal_combo.itemText(i) for i in range(self.animal_combo.count())]:
            animal = self.app_state.get("selected_animal", "All")
            if animal not in [self.animal_combo.itemText(i) for i in range(self.animal_combo.count())]:
                animal = "All"
        self.animal_combo.setCurrentText(animal)
        self._populate_exp_combo(animal, exp, prefer_overall=view == "Overall" or animal == "All")
        self._sync_browser_state()
        if initial and self.tabs.currentIndex() != 0:
            self.tabs.setCurrentIndex(0)
    def _apply_saved_state(self):
        animal = self.app_state.get("selected_animal", "All")
        exp = self.app_state.get("selected_exp_id", "")
        view = self.app_state.get("view_mode", "Session")
        if animal in [self.animal_combo.itemText(i) for i in range(self.animal_combo.count())]:
            self.animal_combo.setCurrentText(animal)
        self._populate_exp_combo(self.animal_combo.currentText(), exp, prefer_overall=view == "Overall" or animal == "All")
        self._sync_browser_state()
    def _populate_exp_combo(self, animal: str, preferred: str | None = None, *, prefer_overall: bool = False):
        sessions = self._sessions_for_scope(animal)
        blocker = QtCore.QSignalBlocker(self.exp_combo)
        self.exp_combo.clear()
        self._summary_by_exp = {s.exp_id: s for s in sessions}
        self.exp_combo.addItem("Overall", {"kind": "overall"})
        overall_tooltip = (
            "Cohort-wide aggregate view for all animals."
            if animal == "All"
            else f"Full-session overview for {animal}."
        )
        self.exp_combo.setItemData(0, overall_tooltip, Qt.ToolTipRole)
        if animal != "All":
            for summary in sessions:
                state = self.store.load_session_state(summary.exp_id)
                label = summary.exp_id
                item_brush = None
                if not summary.has_right_pickle:
                    label += " [no pupil]"
                elif not summary.has_right_video:
                    label += " [no video]"
                if bool(state.get("do_not_use", False)):
                    label += " [do not use]"
                    item_brush = QtGui.QBrush(QtGui.QColor("#b00020"))
                self.exp_combo.addItem(label, {"kind": "session", "exp_id": summary.exp_id})
                idx = self.exp_combo.count() - 1
                if item_brush is not None:
                    self.exp_combo.setItemData(idx, item_brush, Qt.ForegroundRole)
                usable_duration = self.store.effective_session_duration_sec(summary.exp_id)
                cutoff = self.store.session_analysis_cutoff_sec(summary.exp_id)
                tooltip_lines = [
                    summary.exp_id,
                    f"Animal: {summary.animal_id}",
                    f"Duration: {format_seconds(usable_duration)}",
                    f"Right video: {'yes' if summary.has_right_video else 'no'}",
                    f"Right pupil pickle: {'yes' if summary.has_right_pickle else 'no'}",
                    f"Locomotion CSV: {'yes' if summary.has_locomotion_csv else 'no'}",
                ]
                if cutoff is not None:
                    tooltip_lines.append(f"Cutoff: {format_seconds(cutoff)}")
                tooltip = "\n".join(tooltip_lines)
                self.exp_combo.setItemData(idx, tooltip, Qt.ToolTipRole)
        del blocker
        if prefer_overall or animal == "All":
            self.exp_combo.setCurrentIndex(0)
        elif preferred and preferred in self._summary_by_exp:
            for i in range(self.exp_combo.count()):
                data = self.exp_combo.itemData(i)
                if isinstance(data, dict) and data.get("kind") == "session" and data.get("exp_id") == preferred:
                    self.exp_combo.setCurrentIndex(i)
                    break
            else:
                self.exp_combo.setCurrentIndex(1 if self.exp_combo.count() > 1 else 0)
        elif self.exp_combo.count() > 1:
            self.exp_combo.setCurrentIndex(1)
        elif self.exp_combo.count():
            self.exp_combo.setCurrentIndex(0)
    def _current_animal(self) -> str:
        return self.animal_combo.currentText() or "All"
    def _current_exp_entry(self) -> dict:
        if self.exp_combo.count() == 0:
            return {"kind": "overall" if self._current_animal() == "All" else "session", "exp_id": ""}
        data = self.exp_combo.currentData()
        if isinstance(data, dict) and data.get("kind"):
            return data
        text = self.exp_combo.currentText().split(" [", 1)[0]
        if text == "Overall":
            return {"kind": "overall", "exp_id": ""}
        return {"kind": "session", "exp_id": text}
    def _current_view_mode(self) -> str:
        return "Overall" if self._current_exp_entry().get("kind") == "overall" else "Session"
    def _current_exp_id(self) -> str:
        entry = self._current_exp_entry()
        if entry.get("kind") == "session":
            return str(entry.get("exp_id", ""))
        return ""
    def _current_summary(self) -> SessionSummary | None:
        exp_id = self._current_exp_id()
        if not exp_id:
            return None
        return self.store.get_session_summary(exp_id)
    def _current_selection_label(self) -> str:
        entry = self._current_exp_entry()
        if entry.get("kind") == "overall":
            return "Overall"
        return self._current_exp_id() or "--"
    def _is_placeholder_saved_state(self) -> bool:
        return (
            self.app_state.get("selected_animal", "All") == "All"
            and not self.app_state.get("selected_exp_id", "")
            and self.app_state.get("view_mode", "Session") == "Session"
        )
    def _first_available_session(self) -> tuple[str, str]:
        for animal in self.store.animals():
            sessions = self.store.sessions_for_animal(animal)
            if sessions:
                return animal, sessions[0].exp_id
        return "All", ""
    def _sync_browser_state(self):
        if getattr(self, "_initializing", False):
            return
        animal = self._current_animal()
        entry = self._current_exp_entry()
        is_overall = entry.get("kind") == "overall"
        has_session_items = self.exp_combo.count() > 1
        self.exp_combo.setEnabled(True)
        self.prev_btn.setEnabled(not is_overall and has_session_items and self.exp_combo.currentIndex() > 1)
        self.next_btn.setEnabled(not is_overall and has_session_items and self.exp_combo.currentIndex() < self.exp_combo.count() - 1)
        if entry.get("kind") == "session" and entry.get("exp_id"):
            self._last_session_exp_id = str(entry.get("exp_id"))
        self.app_state["selected_animal"] = animal
        self.app_state["selected_exp_id"] = self._last_session_exp_id if is_overall and self._last_session_exp_id else self._current_exp_id()
        self.app_state["view_mode"] = self._current_view_mode()
        self.store.save_app_state(self.app_state)
        self._refresh_contexts()
    def _refresh_contexts(self):
        if getattr(self, "_initializing", False):
            return
        if not hasattr(self, "metrics_tab") or not hasattr(self, "statistics_tab"):
            return
        animal = self._current_animal()
        view = self._current_view_mode()
        exp_id = self._current_exp_id()
        self.metrics_tab.set_selection(animal, exp_id, view)
        self.statistics_tab.set_context(animal, view)
        self.metrics_tab.experiment_index.set_current_selection(animal, exp_id)
        self.status_label.setText(
            f"Selected scope: {animal} | Selection: {self._current_selection_label()} | View: {view} | Output: {self.store.source_root / 'gui_output'}"
        )
    def _on_browser_changed(self):
        if self._updating_browser or getattr(self, "_initializing", False):
            return
        self._updating_browser = True
        try:
            animal = self._current_animal()
            current_exp = self._current_exp_id() or self._last_session_exp_id or self.app_state.get("selected_exp_id", "")
            view = self.app_state.get("view_mode", "Session")
            prefer_overall = view == "Overall" or animal == "All"
            sessions = self._sessions_for_scope(animal)
            preferred = current_exp if current_exp in {s.exp_id for s in sessions} else (sessions[0].exp_id if sessions else "")
            self._populate_exp_combo(animal, preferred, prefer_overall=prefer_overall)
            self._sync_browser_state()
        finally:
            self._updating_browser = False
    def _open_index_session(self, animal_id: str, exp_id: str):
        self._updating_browser = True
        try:
            if animal_id in [self.animal_combo.itemText(i) for i in range(self.animal_combo.count())]:
                self.animal_combo.setCurrentText(animal_id)
            self._last_session_exp_id = exp_id
            self.app_state["selected_animal"] = animal_id
            self.app_state["selected_exp_id"] = exp_id
            self.app_state["view_mode"] = "Session"
            self._populate_exp_combo(animal_id, exp_id, prefer_overall=False)
            self._sync_browser_state()
            self.tabs.setCurrentIndex(0)
        finally:
            self._updating_browser = False
    def _go_prev_session(self):
        if self.exp_combo.count() <= 1 or self._current_view_mode() != "Session":
            return
        idx = self.exp_combo.currentIndex()
        if idx <= 1:
            return
        self.exp_combo.setCurrentIndex(idx - 1)
    def _go_next_session(self):
        if self.exp_combo.count() <= 1 or self._current_view_mode() != "Session":
            return
        idx = self.exp_combo.currentIndex()
        if idx >= self.exp_combo.count() - 1:
            return
        self.exp_combo.setCurrentIndex(idx + 1)
    def _on_scope_changed(self):
        self._refresh_contexts()
        self.statusBar().showMessage("Current scope changed; statistics are marked stale.")
    def _update_dataset(self):
        if self._worker is not None and self._worker.isRunning():
            return
        self.update_btn.setEnabled(False)
        self.statusBar().showMessage("Refreshing dataset index...")
        box = QMessageBox(self)
        box.setWindowTitle("Refresh dataset")
        box.setText("Refresh the dataset index?")
        box.setInformativeText(
            "You can also rebuild DLC model caches from source instead of reusing the cached data."
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Yes)
        refresh_reference_checkbox = QCheckBox("Also refresh Add for DLC model caches", box)
        box.setCheckBox(refresh_reference_checkbox)
        if box.exec_() != QMessageBox.Yes:
            return
        refresh_reference_sessions = refresh_reference_checkbox.isChecked()
        def job(progress_cb):
            return self.store.update_dataset(progress_cb, refresh_reference_sessions=refresh_reference_sessions)
        self._worker = TaskThread(job, self)
        self._worker.progress.connect(self._on_dataset_progress)
        self._worker.result_ready.connect(self._on_dataset_result)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()
    def _on_dataset_progress(self, fraction: float, message: str):
        self.statusBar().showMessage(f"{message} ({fraction * 100.0:.0f}%)")
    def _on_dataset_result(self, index):
        self.index = index
        self.update_btn.setEnabled(True)
        self._populate_browser(initial=False)
        self._sync_browser_state()
        self.metrics_tab.refresh()
        self.statusBar().showMessage(f"Dataset refreshed at {index.generated_at}")
    def _on_worker_error(self, message: str):
        self.update_btn.setEnabled(True)
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            print(message)
        else:
            QMessageBox.critical(self, "Habituation Analysis", message)
        self.statusBar().showMessage("Operation failed")
    def _on_tab_changed(self, index: int):
        widget = self.tabs.widget(index)
        if widget is self.statistics_tab:
            self.statistics_tab.set_context(self._current_animal(), self._current_view_mode())
            if not self._stats_prompted_for_entry:
                self._stats_prompted_for_entry = True
                self.statistics_tab.maybe_prompt_and_run()
        else:
            self._stats_prompted_for_entry = False
    def closeEvent(self, event: QtGui.QCloseEvent):
        self.app_state["selected_animal"] = self._current_animal()
        self.app_state["selected_exp_id"] = self._current_exp_id() or self._last_session_exp_id
        self.app_state["view_mode"] = self._current_view_mode()
        self.store.save_app_state(self.app_state)
        super().closeEvent(event)
def main() -> int:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Habituation Analysis")
    loading = LoadingWindow()
    loading.show()
    app.processEvents()
    try:
        loading.set_message("Loading cached dataset and restoring the last browser state...")
        app.processEvents()
        window = HabituationMainWindow()
        loading.set_message("Finishing startup...")
        window.show()
        app.processEvents()
    except Exception:
        loading.close()
        message = traceback.format_exc()
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            print(message)
        else:
            QMessageBox.critical(None, "Habituation Analysis", message)
        return 1
    loading.close()
    return app.exec_()
