"""Launch Control Widget — live i_cmd(t) curve preview.

Mirrors /vesc/simple_mux's dynamic_reconfigure values, recomputes the same
i_cmd(t) profile that simple_mux._launch_i_cmd uses, and plots it inside rqt
using matplotlib's Qt5Agg backend. Slider changes update the line in-place
(line.set_data + canvas.draw_idle) so axis state / zoom is preserved.
"""

import math
import threading

import numpy as np
import rospy
from dynamic_reconfigure.client import Client as DynRecClient

from python_qt_binding.QtCore import Signal, Qt
from python_qt_binding.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QStatusBar,
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure


# Default values (mirror SimpleMux.cfg defaults). Live-updated by dyn callback.
_DEFAULTS = {
    'launch_profile_mode': 1,
    'launch_current_A':    15.0,
    'launch_I_back':       30.0,
    'launch_I_peak':       75.0,
    'launch_t_back':       0.05,
    'launch_t_rise_end':   0.30,
    'launch_tau':          0.08,
    'launch_t_total':      1.00,
}

_PHASE_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
_PHASE_LABELS = {1: 'P1 backlash', 2: 'P2 exp rise', 3: 'P3 plateau'}


class LaunchControlWidget(QWidget):
    """Main widget: matplotlib plot + dyn_target field + reconnect button."""

    # ROS callback thread -> Qt main thread. Pass a snapshot dict so the
    # widget never reads dyn state from the ROS thread.
    config_signal = Signal(dict)
    status_signal = Signal(str)

    def __init__(self, context=None):
        super().__init__()
        self.setWindowTitle('Launch Control')

        self._cfg = dict(_DEFAULTS)
        self._cfg_lock = threading.Lock()
        self._client = None
        self._dyn_target = '/vesc/simple_mux'

        self._build_ui()
        self._connect_signals()
        self._connect_dyn(self._dyn_target)

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        root = QVBoxLayout(self)

        # row 1 — dyn_target text field + reconnect
        bar = QHBoxLayout()
        bar.addWidget(QLabel('dyn target:'))
        self.target_edit = QLineEdit(self._dyn_target)
        self.target_edit.setMinimumWidth(220)
        bar.addWidget(self.target_edit)
        self.reconnect_btn = QPushButton('Reconnect')
        bar.addWidget(self.reconnect_btn)
        bar.addStretch()
        root.addLayout(bar)

        # matplotlib canvas + nav toolbar (zoom / pan / save)
        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavToolbar(self.canvas, self)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)

        # Long-lived artists — created once, data swapped each redraw.
        # set_data on line / set_xdata on guides; fills must be re-built each time.
        self._line, = self.ax.plot([], [], 'k-', linewidth=2)
        self._fill_artists = []   # phase fill_between handles (cleared every redraw)
        self._peak_line = self.ax.axhline(0.0, color='red', linestyle=':', alpha=0.5)
        self._guide_t_back     = self.ax.axvline(0.0, color='gray',  linestyle='--', alpha=0.55)
        self._guide_t_rise_end = self.ax.axvline(0.0, color='gray',  linestyle='--', alpha=0.55)
        self._guide_t_total    = self.ax.axvline(0.0, color='black', linestyle='-',  alpha=0.4)

        self.ax.set_xlabel('t [s]')
        self.ax.set_ylabel('i_cmd [A]')
        self.ax.grid(True, alpha=0.3)

        self.status_bar = QStatusBar()
        self.status_bar.showMessage('Waiting for dyn_reconfigure …')
        root.addWidget(self.status_bar)

    def _connect_signals(self):
        self.config_signal.connect(self._on_config_main)
        self.status_signal.connect(self._on_status)
        self.reconnect_btn.clicked.connect(self._on_reconnect)
        # Pressing Enter in the target field also reconnects.
        self.target_edit.returnPressed.connect(self._on_reconnect)

    # ------------------------------------------------------------------ #
    #  dyn_reconfigure plumbing
    # ------------------------------------------------------------------ #
    def _connect_dyn(self, target):
        # Tear down old client (if any)
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._dyn_target = target
        try:
            self._client = DynRecClient(
                target, config_callback=self._on_config_ros, timeout=5.0)
            self.status_signal.emit('connected: %s' % target)
        except rospy.ROSException:
            self.status_signal.emit(
                'dyn server not found: %s (showing defaults)' % target)
        # Even without a client, render once so the user sees defaults.
        self._redraw()

    def _on_config_ros(self, config):
        # ROS callback thread — do not touch Qt widgets here.
        snap = {k: config.get(k, _DEFAULTS[k]) for k in _DEFAULTS}
        with self._cfg_lock:
            self._cfg.update(snap)
        self.config_signal.emit(snap)

    def _on_config_main(self, _snap):
        # Qt main thread — safe to redraw.
        self._redraw()

    def _on_status(self, text):
        self.status_bar.showMessage(text)

    def _on_reconnect(self):
        target = self.target_edit.text().strip() or '/vesc/simple_mux'
        self._connect_dyn(target)

    # ------------------------------------------------------------------ #
    #  i_cmd(t) — mirror of simple_mux._launch_i_cmd
    # ------------------------------------------------------------------ #
    @staticmethod
    def _i_cmd(t, cfg):
        mode = int(cfg['launch_profile_mode'])
        t_total = float(cfg['launch_t_total'])
        if mode == 0:
            inside = (0.0 <= t < t_total)
            return (float(cfg['launch_current_A']) if inside else 0.0,
                    1 if inside else 0)
        if t < 0.0:
            return 0.0, 0
        t_back     = float(cfg['launch_t_back'])
        t_rise_end = float(cfg['launch_t_rise_end'])
        tau        = float(cfg['launch_tau'])
        I_back     = float(cfg['launch_I_back'])
        I_peak     = float(cfg['launch_I_peak'])
        if t < t_back:
            return I_back * (t / max(t_back, 1e-6)), 1
        if t < t_rise_end:
            return (I_back + (I_peak - I_back)
                    * (1.0 - math.exp(-(t - t_back) / max(tau, 1e-6))), 2)
        if t < t_total:
            return I_peak, 3
        return 0.0, 0

    # ------------------------------------------------------------------ #
    #  Redraw (Qt main thread only)
    # ------------------------------------------------------------------ #
    def _redraw(self):
        with self._cfg_lock:
            cfg = dict(self._cfg)

        mode = int(cfg['launch_profile_mode'])
        t_total = max(float(cfg['launch_t_total']), 1e-3)
        I_peak = float(cfg['launch_I_peak'])
        I_const = float(cfg['launch_current_A'])
        peak_val = I_const if mode == 0 else I_peak

        ts = np.linspace(0.0, t_total, 600)
        ys = np.empty_like(ts)
        phases = np.empty_like(ts, dtype=int)
        for k, t in enumerate(ts):
            y, ph = self._i_cmd(float(t), cfg)
            ys[k] = y
            phases[k] = ph

        # 1. line — pure in-place
        self._line.set_data(ts, ys)

        # 2. fill_between is a PolyCollection — cannot in-place. remove + re-add.
        for art in self._fill_artists:
            try:
                art.remove()
            except (ValueError, AttributeError):
                pass
        self._fill_artists = []
        for ph in (1, 2, 3):
            mask = phases == ph
            if mask.any():
                art = self.ax.fill_between(
                    ts, 0.0, ys, where=mask,
                    color=_PHASE_COLORS[ph], alpha=0.25,
                    label=_PHASE_LABELS[ph])
                self._fill_artists.append(art)

        # 3. guides
        t_back     = float(cfg['launch_t_back'])
        t_rise_end = float(cfg['launch_t_rise_end'])
        if mode == 1:
            self._guide_t_back.set_xdata([t_back, t_back])
            self._guide_t_back.set_visible(0.0 < t_back < t_total)
            self._guide_t_rise_end.set_xdata([t_rise_end, t_rise_end])
            self._guide_t_rise_end.set_visible(0.0 < t_rise_end < t_total)
        else:
            self._guide_t_back.set_visible(False)
            self._guide_t_rise_end.set_visible(False)
        self._guide_t_total.set_xdata([t_total, t_total])
        self._peak_line.set_ydata([peak_val, peak_val])

        # 4. axes + title
        y_top = max(I_peak, I_const, 1.0) * 1.15 + 1.0
        self.ax.set_xlim(0.0, t_total)
        self.ax.set_ylim(0.0, y_top)
        mode_txt = 'CONSTANT' if mode == 0 else 'CURVE'
        if mode == 1:
            title = ('launch i_cmd(t) — %s   I_back=%.1f  I_peak=%.1f  '
                     'tau=%.3f  t_back=%.3f  t_rise_end=%.3f  t_total=%.2f'
                     % (mode_txt, float(cfg['launch_I_back']), I_peak,
                        float(cfg['launch_tau']), t_back, t_rise_end, t_total))
        else:
            title = ('launch i_cmd(t) — %s   I_const=%.1f  t_total=%.2f'
                     % (mode_txt, I_const, t_total))
        self.ax.set_title(title, fontsize=9)

        # legend: rebuild since fill_between artists changed
        if mode == 1:
            self.ax.legend(loc='lower right', fontsize=8)
        else:
            leg = self.ax.get_legend()
            if leg is not None:
                leg.remove()

        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #  Shutdown
    # ------------------------------------------------------------------ #
    def shutdown(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
