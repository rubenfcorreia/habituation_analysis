from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QLabel, QPushButton, QSlider, QComboBox, QHBoxLayout, QVBoxLayout, QWidget


class DraggableHLine:
    """A draggable horizontal matplotlib line."""

    def __init__(
        self,
        ax,
        y0: float,
        *,
        color: str = "tab:red",
        linestyle: str = "--",
        linewidth: float = 2.0,
        label: str | None = None,
        tolerance_px: int = 8,
        on_changed: Callable[[float], None] | None = None,
        is_enabled_fn: Callable[[], bool] | None = None,
    ):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.on_changed = on_changed
        self.is_enabled_fn = is_enabled_fn
        self.tolerance_px = tolerance_px
        self.line = ax.axhline(
            y0, color=color, linestyle=linestyle, linewidth=linewidth, zorder=20
        )
        if label is not None:
            self.line.set_label(label)
        self._dragging = False
        self._cid_press = self.canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_motion = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._cid_release = self.canvas.mpl_connect("button_release_event", self._on_release)

    def disconnect(self):
        for cid in (self._cid_press, self._cid_motion, self._cid_release):
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        try:
            self.line.remove()
        except Exception:
            pass

    def get_y(self) -> float:
        return float(self.line.get_ydata()[0])

    def set_y(self, y: float, *, trigger_callback: bool = True):
        self.line.set_ydata([y, y])
        self.canvas.draw_idle()
        if trigger_callback and self.on_changed is not None:
            self.on_changed(float(y))

    def _enabled(self) -> bool:
        if self.is_enabled_fn is None:
            return True
        try:
            return bool(self.is_enabled_fn())
        except Exception:
            return True

    def _is_near_line(self, event) -> bool:
        if event.inaxes is not self.ax or event.ydata is None:
            return False
        y_line = self.get_y()
        p_event = self.ax.transData.transform((0.0, event.ydata))
        p_line = self.ax.transData.transform((0.0, y_line))
        return abs(p_event[1] - p_line[1]) <= self.tolerance_px

    def _on_press(self, event):
        if not self._enabled() or event.button != 1:
            return
        if self._is_near_line(event):
            self._dragging = True

    def _on_motion(self, event):
        if not self._enabled() or not self._dragging:
            return
        if event.inaxes is not self.ax or event.ydata is None:
            return
        self.set_y(float(event.ydata), trigger_callback=True)

    def _on_release(self, event):
        self._dragging = False


class VideoPlayerWidget(QWidget):
    """Simple OpenCV-backed video player for the right eye video."""

    time_changed = pyqtSignal(float)
    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(360, 240)
        self.video_label.setStyleSheet("background: #111; color: #ddd;")
        self.video_label.setScaledContents(False)
        self.video_path: str | None = None
        self.timestamps = np.array([], dtype=float)
        self._overlay = None
        self._capture: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._playing = False
        self._fps = 10.0
        self._playback_speed = 1.0
        self._speed_options = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._toggle_play)
        self.prev_btn = QPushButton("<")
        self.prev_btn.setAutoRepeat(True)
        self.prev_btn.setAutoRepeatDelay(300)
        self.prev_btn.setAutoRepeatInterval(50)
        self.prev_btn.clicked.connect(lambda: self.seek(self._frame_index - 1))
        self.next_btn = QPushButton(">")
        self.next_btn.setAutoRepeat(True)
        self.next_btn.setAutoRepeatDelay(300)
        self.next_btn.setAutoRepeatInterval(50)
        self.next_btn.clicked.connect(lambda: self.seek(self._frame_index + 1))
        self.speed_combo = QComboBox()
        for speed in self._speed_options:
            self.speed_combo.addItem(f"{speed:g}x", speed)
        self.speed_combo.setCurrentIndex(self._speed_options.index(1.0))
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._slider_changed)
        self.current_label = QLabel("0.0 s")
        self.total_label = QLabel("/ 0.0 s")

        controls = QHBoxLayout()
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.next_btn)
        controls.addWidget(QLabel("Speed"))
        controls.addWidget(self.speed_combo)
        controls.addWidget(self.current_label)
        controls.addWidget(self.total_label)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.slider)
        layout.addLayout(controls)

    def close_video(self):
        self.pause()
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
        self._capture = None
        self.video_path = None
        self.timestamps = np.array([], dtype=float)
        self._overlay = None
        self._frame_index = 0
        self.slider.setEnabled(False)
        self.video_label.setText("No video loaded")
        self.current_label.setText("0.0 s")
        self.total_label.setText("/ 0.0 s")
        self.pause()

    def set_video(self, video_path: str | None, timestamps: np.ndarray | None, overlay=None):
        self.close_video()
        self._overlay = overlay
        if not video_path:
            self.video_label.setText("No right video available")
            return
        self.video_path = video_path
        self.timestamps = np.asarray(timestamps if timestamps is not None else [], dtype=float)
        self._capture = cv2.VideoCapture(video_path)
        if not self._capture.isOpened():
            self.video_label.setText("Unable to open video")
            return
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps > 0:
            self._fps = fps
        self._apply_playback_speed()
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0 and self.timestamps.size:
            frame_count = int(self.timestamps.size)
        self.slider.setEnabled(True)
        self.slider.setMaximum(max(0, frame_count - 1))
        self._frame_index = 0
        self._show_frame(0)
        if self.timestamps.size:
            self.total_label.setText(f"/ {float(self.timestamps[-1]):.1f} s")
        else:
            self.total_label.setText(f"/ {frame_count / self._fps:.1f} s")

    def current_time(self) -> float:
        if self.timestamps.size and 0 <= self._frame_index < self.timestamps.size:
            return float(self.timestamps[self._frame_index])
        return float(self._frame_index / max(self._fps, 1e-6))

    def seek(self, frame_index: int):
        if self._capture is None:
            return
        max_index = int(self.slider.maximum())
        frame_index = max(0, min(int(frame_index), max_index))
        self._frame_index = frame_index
        self.slider.blockSignals(True)
        self.slider.setValue(frame_index)
        self.slider.blockSignals(False)
        self._show_frame(frame_index)
        self.frame_changed.emit(frame_index)
        self.time_changed.emit(self.current_time())

    def pause(self):
        self._playing = False
        self.play_btn.blockSignals(True)
        self.play_btn.setChecked(False)
        self.play_btn.blockSignals(False)
        self._timer.stop()

    def _apply_playback_speed(self):
        if self._playing:
            interval = int(round(1000.0 / max(self._fps * self._playback_speed, 1e-6)))
            self._timer.start(max(20, interval))

    def _toggle_play(self, checked: bool):
        self._playing = bool(checked)
        if checked:
            self._apply_playback_speed()
            self.play_btn.setText("Pause")
        else:
            self._timer.stop()
            self.play_btn.setText("Play")

    def _advance(self):
        if self._capture is None:
            self.pause()
            return
        next_idx = self._frame_index + 1
        if next_idx > int(self.slider.maximum()):
            self.pause()
            return
        self.seek(next_idx)

    def _on_speed_changed(self, index: int):
        speed = self.speed_combo.itemData(index)
        try:
            self._playback_speed = float(speed)
        except Exception:
            self._playback_speed = 1.0
        self._apply_playback_speed()

    def _slider_changed(self, value: int):
        if value != self._frame_index:
            self.seek(int(value))

    def _draw_series(self, frame, xs, ys, color, *, point_radius=4, thickness=2, connect=False, closed=False):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.ndim == 0 or ys.ndim == 0:
            return frame
        n = min(xs.shape[0], ys.shape[0])
        pts = []
        for x, y in zip(xs[:n].ravel(), ys[:n].ravel()):
            if np.isfinite(x) and np.isfinite(y):
                pts.append((int(round(x)), int(round(y))))
        if not pts:
            return frame
        if connect and len(pts) > 1:
            poly = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [poly], isClosed=closed, color=color, thickness=thickness)
        if not connect or not closed:
            for pt in pts:
                cv2.circle(frame, pt, point_radius, color, -1)
        return frame

    def _overlay_value(self, overlay, key: str, default=None):
        if overlay is None:
            return default
        if isinstance(overlay, dict):
            return overlay.get(key, default)
        return getattr(overlay, key, default)

    def _draw_overlay(self, frame, frame_index: int):
        overlay = self._overlay
        if overlay is None:
            return frame
        x = np.asarray(self._overlay_value(overlay, "x", []), dtype=float).reshape(-1)
        y = np.asarray(self._overlay_value(overlay, "y", []), dtype=float).reshape(-1)
        radius = np.asarray(self._overlay_value(overlay, "radius", []), dtype=float).reshape(-1)
        if frame_index < x.size and frame_index < y.size and frame_index < radius.size:
            if np.isfinite(x[frame_index]) and np.isfinite(y[frame_index]) and np.isfinite(radius[frame_index]):
                center = (int(round(x[frame_index])), int(round(y[frame_index])))
                rad = max(1, int(round(radius[frame_index])))
                cv2.circle(frame, center, rad, (0, 0, 255), 2)
        eye_lid_x = np.asarray(self._overlay_value(overlay, "eye_lid_x", []), dtype=float)
        eye_lid_y = np.asarray(self._overlay_value(overlay, "eye_lid_y", []), dtype=float)
        if eye_lid_x.ndim >= 2 and eye_lid_y.ndim >= 2 and frame_index < eye_lid_x.shape[0] and frame_index < eye_lid_y.shape[0]:
            frame = self._draw_series(frame, eye_lid_x[frame_index], eye_lid_y[frame_index], (0, 255, 0), thickness=2, connect=True, closed=True)
        eyeX = np.asarray(self._overlay_value(overlay, "eyeX", []), dtype=float)
        eyeY = np.asarray(self._overlay_value(overlay, "eyeY", []), dtype=float)
        if eyeX.ndim >= 2 and eyeY.ndim >= 2 and frame_index < eyeX.shape[0] and frame_index < eyeY.shape[0]:
            frame = self._draw_series(frame, eyeX[frame_index], eyeY[frame_index], (255, 128, 0), point_radius=3, connect=False)
        pupilX = np.asarray(self._overlay_value(overlay, "pupilX", []), dtype=float)
        pupilY = np.asarray(self._overlay_value(overlay, "pupilY", []), dtype=float)
        if pupilX.ndim >= 2 and pupilY.ndim >= 2 and frame_index < pupilX.shape[0] and frame_index < pupilY.shape[0]:
            frame = self._draw_series(frame, pupilX[frame_index], pupilY[frame_index], (255, 255, 0), point_radius=3, connect=False)
        return frame

    def _show_frame(self, frame_index: int):
        if self._capture is None:
            return
        try:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self._capture.read()
        except Exception:
            ok = False
            frame = None
        if not ok or frame is None:
            self.video_label.setText("Frame unavailable")
            return
        frame = self._draw_overlay(frame, frame_index)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)
        if self.timestamps.size and frame_index < self.timestamps.size:
            t = float(self.timestamps[frame_index])
        else:
            t = frame_index / max(self._fps, 1e-6)
        self.current_label.setText(f"{t:.1f} s")

