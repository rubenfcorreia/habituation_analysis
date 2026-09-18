from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
from habituation_analysis.widgets import TRACE_RIGHT_PADDING_FRACTION, TracePanZoomCanvas


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def canvas(qapp):
    widget = TracePanZoomCanvas()
    yield widget
    widget.close()
    widget.deleteLater()


def _limits(canvas):
    return [axis.get_xlim() for axis in canvas._trace_axes]


def test_initial_view_has_right_padding_on_all_trace_axes(canvas):
    canvas.set_x_bounds(10.0, 110.0, reset_view=True)
    canvas.apply_view()

    expected_right = 110.0 + 100.0 * TRACE_RIGHT_PADDING_FRACTION
    assert canvas._data_xlim == (10.0, 110.0)
    assert all(np.allclose(limit, (10.0, expected_right)) for limit in _limits(canvas))


def test_reset_zoom_restores_padded_full_view(canvas):
    canvas.set_x_bounds(0.0, 200.0, reset_view=True)
    canvas.zoom(0.5)
    assert canvas._view_xlim[1] - canvas._view_xlim[0] < 200.0 * (1.0 + TRACE_RIGHT_PADDING_FRACTION)

    canvas.reset_zoom()

    assert np.allclose(canvas._view_xlim, (0.0, 200.0 * (1.0 + TRACE_RIGHT_PADDING_FRACTION)))


def test_zoom_narrows_view_and_pan_stays_within_padded_bounds(canvas):
    canvas.set_x_bounds(0.0, 100.0, reset_view=True)
    full_right = 100.0 * (1.0 + TRACE_RIGHT_PADDING_FRACTION)
    canvas.zoom(0.5)
    zoomed_span = canvas._view_xlim[1] - canvas._view_xlim[0]
    assert zoomed_span < full_right

    canvas.pan(10_000.0)
    assert canvas._view_xlim[1] <= full_right
    canvas.pan(-10_000.0)
    assert canvas._view_xlim[0] >= 0.0


def test_empty_and_degenerate_bounds_are_safe(canvas):
    canvas.clear_limits()
    canvas.apply_view()
    canvas.reset_zoom()

    canvas.set_x_bounds(4.0, 4.0, reset_view=True)
    canvas.apply_view()

    assert canvas._data_xlim == (4.0, 5.0)
    assert np.allclose(canvas._view_xlim, (4.0, 5.03))

