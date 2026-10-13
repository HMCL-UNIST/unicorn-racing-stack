"""GG Viewer Widget — matplotlib + table + multi-snapshot overlay."""

import json
import os
import copy
from datetime import datetime

import numpy as np
import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from python_qt_binding.QtCore import Qt, Signal
from python_qt_binding.QtGui import QColor, QFont
from python_qt_binding.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QComboBox, QCheckBox, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QStatusBar, QGroupBox,
    QListWidget, QListWidgetItem, QMessageBox,
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator   # 0.5-step grid


# color cycle for plotted versions (current=blue, baseline=gray, vers cycle)
_VERSION_COLORS = ['#FF9800', '#4CAF50', '#9C27B0', '#795548',
                   '#00BCD4', '#E91E63', '#8BC34A', '#FF5722']


class GGViewerWidget(QWidget):
    """Main widget: diamond plot + summary table + multi-snapshot overlay."""

    # Thread-safe signals (ROS callback thread → Qt main thread)
    results_signal = Signal(str)
    status_signal = Signal(str)

    def __init__(self, context=None):
        super().__init__()
        self.setWindowTitle('GG Viewer')

        self.current_data = None         # updated by /gg_results topic; pre-seeded from _latest on disk
        self.baseline_data = None        # set by Load Baseline button
        # multi-snapshot overlay
        self.snapshots = {}              # disk-load cache: {snap_name: payload}
        self.plotted_versions = set()    # currently checked snapshot names

        # disk scan dir for snapshot list
        _this_file = os.path.abspath(__file__)
        _race_stack_root = os.path.normpath(
            os.path.join(os.path.dirname(_this_file), '..', '..', '..'))
        self.gg_diagrams_dir = os.path.join(
            _race_stack_root, 'planner', '3d_gb_optimizer',
            'global_line', 'data', 'gg_diagrams')

        self._build_ui()
        self._connect_signals()
        self._subscribe_ros()
        self._refresh_snapshot_lists()   # initial disk scan
        self._load_latest_from_disk()    # pre-seed current_data before ROS topic arrives
        if self.current_data is not None:
            self._populate_combos()
            self._update_display()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        root = QVBoxLayout(self)

        # row 1 — baseline selector + save/refresh
        toolbar1 = QHBoxLayout()
        toolbar1.addWidget(QLabel('Baseline:'))
        self.baseline_combo = QComboBox()
        self.baseline_combo.setMinimumWidth(180)
        toolbar1.addWidget(self.baseline_combo)
        self.load_baseline_btn = QPushButton('Load')
        toolbar1.addWidget(self.load_baseline_btn)
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.setToolTip('Re-scan gg_diagrams/ directory')
        toolbar1.addWidget(self.refresh_btn)
        # restore baseline → _latest (overwrites current)
        self.restore_latest_btn = QPushButton('Restore Baseline → Latest')
        self.restore_latest_btn.setToolTip(
            'Copies the selected baseline (v_N) into _latest on disk; '
            'velopt picks it up on next Apply / cold-start.')
        toolbar1.addWidget(self.restore_latest_btn)
        toolbar1.addSpacing(20)
        self.save_version_btn = QPushButton('Save Current as Version')
        self.save_version_btn.setToolTip(
            'Calls /gg_tuner/save_version → snapshots latest to v_N on disk')
        toolbar1.addWidget(self.save_version_btn)
        toolbar1.addStretch()
        root.addLayout(toolbar1)

        # row 2 — versions multi-select + V/g + show baseline toggle
        toolbar2 = QHBoxLayout()
        toolbar2.addWidget(QLabel('Plotted Versions:'))
        self.versions_list = QListWidget()
        self.versions_list.setMaximumHeight(80)
        self.versions_list.setMinimumWidth(300)
        toolbar2.addWidget(self.versions_list)
        toolbar2.addSpacing(20)
        toolbar2.addWidget(QLabel('V:'))
        self.v_combo = QComboBox()
        self.v_combo.setMinimumWidth(80)
        toolbar2.addWidget(self.v_combo)
        toolbar2.addWidget(QLabel('g:'))
        self.g_combo = QComboBox()
        self.g_combo.setMinimumWidth(80)
        toolbar2.addWidget(self.g_combo)
        self.show_baseline_check = QCheckBox('Show baseline')
        self.show_baseline_check.setChecked(True)
        toolbar2.addWidget(self.show_baseline_check)
        toolbar2.addStretch()
        root.addLayout(toolbar2)

        # --- tabs ---
        self.tabs = QTabWidget()

        # Tab 1: Diamond plot + summary table
        main_tab = QWidget()
        main_layout = QHBoxLayout(main_tab)
        splitter = QSplitter(Qt.Horizontal)

        # left: matplotlib diamond plot
        plot_group = QGroupBox('GG Diamond')
        plot_layout = QVBoxLayout(plot_group)
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax_plot = self.fig.add_subplot(111)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_group)

        # right: summary table
        table_group = QGroupBox('Diamond Summary (velocity_frame)')
        table_layout = QVBoxLayout(table_group)
        self.table = QTableWidget()
        self.table.setFont(QFont('Monospace', 9))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.table)
        splitter.addWidget(table_group)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)
        self.tabs.addTab(main_tab, 'Diamond + Table')

        # Tab 2: Tuning params
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(3)
        self.params_table.setHorizontalHeaderLabels(['Param', 'Current', 'Baseline'])
        self.params_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        params_layout.addWidget(self.params_table)
        self.tabs.addTab(params_tab, 'Tuning Params')

        root.addWidget(self.tabs)

        # --- status bar ---
        self.status_bar = QStatusBar()
        self.status_bar.showMessage('Waiting for /gg_results ...')
        root.addWidget(self.status_bar)

    def _connect_signals(self):
        self.results_signal.connect(self._handle_results)
        self.status_signal.connect(self._handle_status)
        self.v_combo.currentIndexChanged.connect(self._on_combo_changed)
        self.g_combo.currentIndexChanged.connect(self._on_combo_changed)
        self.show_baseline_check.stateChanged.connect(self._on_combo_changed)
        # new buttons + multi-version list
        self.load_baseline_btn.clicked.connect(self._on_baseline_load)
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.save_version_btn.clicked.connect(self._on_save_version)
        self.restore_latest_btn.clicked.connect(self._on_restore_to_latest)
        self.versions_list.itemChanged.connect(self._on_version_check_changed)

    # ------------------------------------------------------------------ #
    #  ROS subscriptions
    # ------------------------------------------------------------------ #
    def _subscribe_ros(self):
        self._sub_results = rospy.Subscriber(
            '/gg_results', String, self._on_results_cb, queue_size=1)
        self._sub_status = rospy.Subscriber(
            '/gg_compute_status', String, self._on_status_cb, queue_size=5)

    def _on_results_cb(self, msg):
        self.results_signal.emit(msg.data)

    def _on_status_cb(self, msg):
        self.status_signal.emit(msg.data)

    # ------------------------------------------------------------------ #
    #  Qt slot handlers (main thread)
    # ------------------------------------------------------------------ #
    def _handle_results(self, json_str):
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            rospy.logwarn(f'[GGViewer] JSON parse error: {e}')
            return
        self.current_data = data

        # auto-discard incompatible baseline when v_list/g_list change
        baseline_reset_msg = None
        if self.baseline_data is not None:
            ok, reason = self._baseline_compatible(data, self.baseline_data)
            if not ok:
                self.baseline_data = None
                baseline_reset_msg = f'Baseline reset (incompatible: {reason})'

        self._populate_combos()
        self._update_display()
        if baseline_reset_msg is not None:
            self.status_bar.showMessage(baseline_reset_msg)
        else:
            self.status_bar.showMessage(
                f"Received: {data.get('vehicle_name', '?')} "
                f"@ {data.get('timestamp', '?')}")

    def _handle_status(self, text):
        self.status_bar.showMessage(text)

    def _on_combo_changed(self, _=None):
        self._update_display()

    # ------------------------------------------------------------------ #
    #  Snapshot disk scan + load (multi-version overlay)
    # ------------------------------------------------------------------ #
    # scan gg_diagrams/ for snapshot dirs (excl. *_latest — current slot)
    def _scan_snapshots(self):
        if not os.path.isdir(self.gg_diagrams_dir):
            return []
        snaps = []
        for name in sorted(os.listdir(self.gg_diagrams_dir)):
            full = os.path.join(self.gg_diagrams_dir, name)
            if not os.path.isdir(full):
                continue
            if name.endswith('_latest'):
                continue
            # require velocity_frame npy presence
            if not os.path.isfile(os.path.join(full, 'velocity_frame', 'v_list.npy')):
                continue
            snaps.append(name)
        return snaps

    def _load_latest_from_disk(self):
        """Pre-seed current_data from *_latest dir so the plot shows on cold start."""
        if not os.path.isdir(self.gg_diagrams_dir):
            return
        for name in sorted(os.listdir(self.gg_diagrams_dir)):
            if not name.endswith('_latest'):
                continue
            full = os.path.join(self.gg_diagrams_dir, name)
            if not os.path.isdir(full):
                continue
            if not os.path.isfile(os.path.join(full, 'velocity_frame', 'v_list.npy')):
                continue
            data = self._load_snapshot_from_disk(name)
            if data is not None:
                self.current_data = data
                rospy.loginfo(f'[GGViewer] pre-seeded current from disk: {name}')
                return

    # load one snapshot from disk → /gg_results-style payload dict
    def _load_snapshot_from_disk(self, snap_name):
        if snap_name in self.snapshots:
            return self.snapshots[snap_name]
        vf = os.path.join(self.gg_diagrams_dir, snap_name, 'velocity_frame')
        try:
            v_list = np.load(os.path.join(vf, 'v_list.npy')).tolist()
            g_list = np.load(os.path.join(vf, 'g_list.npy')).tolist()
            ax_max = np.load(os.path.join(vf, 'ax_max.npy')).tolist()
            ax_min = np.load(os.path.join(vf, 'ax_min.npy')).tolist()
            ay_max = np.load(os.path.join(vf, 'ay_max.npy')).tolist()
            gg_exp = np.load(os.path.join(vf, 'gg_exponent.npy')).tolist()
        except (FileNotFoundError, OSError) as e:
            rospy.logwarn(f'[GGViewer] load {snap_name} failed: {e}')
            return None
        # tuning_params from params_used.json if present
        tuning_params = {}
        meta_path = os.path.join(self.gg_diagrams_dir, snap_name, 'params_used.json')
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                tuning_params = meta.get('tuning', {})
            except (OSError, json.JSONDecodeError):
                pass
        payload = {
            'vehicle_name': snap_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'v_list': v_list,
            'g_list': g_list,
            'diamond': {
                'ax_max': ax_max, 'ax_min': ax_min,
                'ay_max': ay_max, 'gg_exponent': gg_exp,
            },
            'tuning_params': tuning_params,
        }
        self.snapshots[snap_name] = payload
        return payload

    # refresh baseline_combo + versions_list while preserving state
    def _refresh_snapshot_lists(self):
        snaps = self._scan_snapshots()
        # Baseline combo: keep current selection if still present
        prev_baseline = self.baseline_combo.currentText() if self.baseline_combo.count() else ''
        self.baseline_combo.blockSignals(True)
        self.baseline_combo.clear()
        self.baseline_combo.addItem('(none)')
        for s in snaps:
            self.baseline_combo.addItem(s)
        idx = self.baseline_combo.findText(prev_baseline)
        if idx >= 0:
            self.baseline_combo.setCurrentIndex(idx)
        self.baseline_combo.blockSignals(False)
        # Versions list: keep checked state for surviving names
        prev_checked = set(self.plotted_versions)
        self.versions_list.blockSignals(True)
        self.versions_list.clear()
        for s in snaps:
            item = QListWidgetItem(s)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if s in prev_checked else Qt.Unchecked)
            self.versions_list.addItem(item)
        self.versions_list.blockSignals(False)
        # Drop checked names that disappeared from disk
        self.plotted_versions = self.plotted_versions & set(snaps)

    # button handlers
    def _on_baseline_load(self):
        snap = self.baseline_combo.currentText()
        if not snap or snap == '(none)':
            self.baseline_data = None
            self.status_bar.showMessage('Baseline cleared')
        else:
            data = self._load_snapshot_from_disk(snap)
            if data is None:
                self.status_bar.showMessage(f'Load failed: {snap}')
                return
            self.baseline_data = data
            self.status_bar.showMessage(f'Baseline = {snap}')
        self._populate_combos()
        self._update_display()

    def _on_version_check_changed(self, item):
        snap = item.text()
        if item.checkState() == Qt.Checked:
            data = self._load_snapshot_from_disk(snap)
            if data is None:
                item.setCheckState(Qt.Unchecked)
                return
            self.plotted_versions.add(snap)
        else:
            self.plotted_versions.discard(snap)
        self._populate_combos()
        self._update_display()

    def _on_save_version(self):
        try:
            rospy.wait_for_service('/gg_tuner/save_version', timeout=2.0)
            srv = rospy.ServiceProxy('/gg_tuner/save_version', Trigger)
            resp = srv()
        except (rospy.ROSException, rospy.ServiceException) as e:
            self.status_bar.showMessage(f'Save failed: {e}')
            return
        if resp.success:
            self.status_bar.showMessage(f'Saved as {resp.message}')
            self._refresh_snapshot_lists()
        else:
            self.status_bar.showMessage(f'Save failed: {resp.message}')

    # copy selected baseline (v_N) into _latest
    def _on_restore_to_latest(self):
        snap = self.baseline_combo.currentText()
        if not snap or snap == '(none)':
            self.status_bar.showMessage('Restore: select a baseline first')
            return
        reply = QMessageBox.question(
            self, 'Restore to Latest',
            f'Overwrite _latest with "{snap}"?\n'
            f'(GGV diagrams + params yml on disk; velopt reloads on next Apply)',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            rospy.set_param('/gg_tuner/restore_target', snap)
            rospy.wait_for_service('/gg_tuner/restore_version', timeout=2.0)
            srv = rospy.ServiceProxy('/gg_tuner/restore_version', Trigger)
            resp = srv()
        except (rospy.ROSException, rospy.ServiceException) as e:
            self.status_bar.showMessage(f'Restore failed: {e}')
            return
        if resp.success:
            self.status_bar.showMessage(
                f'Restored {resp.message} (Apply to reload velopt)')
            self._refresh_snapshot_lists()
        else:
            self.status_bar.showMessage(f'Restore failed: {resp.message}')

    def _on_refresh(self):
        self._refresh_snapshot_lists()
        self.status_bar.showMessage(
            f'Snapshots refreshed ({self.versions_list.count()} found)')

    # ------------------------------------------------------------------ #
    #  Populate combo boxes
    # ------------------------------------------------------------------ #
    def _populate_combos(self):
        # union all V/g across current + baseline + cached snapshots
        all_v = set()
        all_g = set()
        if self.current_data is not None:
            all_v |= set(self.current_data.get('v_list', []))
            all_g |= set(self.current_data.get('g_list', []))
        if self.baseline_data is not None:
            all_v |= set(self.baseline_data.get('v_list', []))
            all_g |= set(self.baseline_data.get('g_list', []))
        for snap in self.snapshots.values():
            all_v |= set(snap.get('v_list', []))
            all_g |= set(snap.get('g_list', []))
        v_list = sorted(all_v)
        g_list = sorted(all_g)

        prev_v = self.v_combo.currentText()
        prev_g = self.g_combo.currentText()

        self.v_combo.blockSignals(True)
        self.g_combo.blockSignals(True)
        self.v_combo.clear()
        self.g_combo.clear()
        for v in v_list:
            self.v_combo.addItem(f'{v:.2f}')
        for g in g_list:
            self.g_combo.addItem(f'{g:.2f}')

        # restore previous selection if possible
        idx_v = self.v_combo.findText(prev_v)
        if idx_v >= 0:
            self.v_combo.setCurrentIndex(idx_v)
        idx_g = self.g_combo.findText(prev_g)
        if idx_g >= 0:
            self.g_combo.setCurrentIndex(idx_g)

        self.v_combo.blockSignals(False)
        self.g_combo.blockSignals(False)

    # ------------------------------------------------------------------ #
    #  Update all displays
    # ------------------------------------------------------------------ #
    def _update_display(self):
        # plot works with baseline/versions only (no current required).
        # table/params now also fall back to baseline when current is absent.
        self._update_plot()
        ref = (self.current_data
               if self.current_data is not None
               else self.baseline_data)
        if ref is not None:
            self._update_table(ref)
            self._update_params_tab(ref)
        else:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.params_table.setRowCount(0)

    # ------------------------------------------------------------------ #
    #  Summary table
    # ------------------------------------------------------------------ #
    def _update_table(self, ref):
        # ref drives the table (current preferred, baseline fallback).
        d = ref['diamond']
        v_list = ref['v_list']
        g_list = ref['g_list']
        # diff vs baseline only when ref is current (baseline-alone = no diff)
        ref_is_current = ref is self.current_data
        compat_ok, _ = self._baseline_compatible(
            ref, self.baseline_data) if ref_is_current else (False, '')
        show_diff = (ref_is_current
                     and self.show_baseline_check.isChecked()
                     and self.baseline_data is not None
                     and compat_ok)

        n_v = len(v_list)
        n_g = len(g_list)
        # rows = velocities, columns = g values × 3 metrics
        metrics = ['ax_max', 'ax_min', 'ay_max']
        n_cols = n_g * len(metrics)

        self.table.setRowCount(n_v)
        self.table.setColumnCount(n_cols)

        # headers
        col_headers = []
        for g in g_list:
            for m in metrics:
                col_headers.append(f'g={g:.1f}\n{m}')
        self.table.setHorizontalHeaderLabels(col_headers)

        row_headers = [f'V={v:.1f}' for v in v_list]
        self.table.setVerticalHeaderLabels(row_headers)

        base_d = self.baseline_data['diamond'] if show_diff else None

        for vi in range(n_v):
            for gi in range(n_g):
                for mi, m_key in enumerate(metrics):
                    col = gi * len(metrics) + mi
                    val = d[m_key][vi][gi]
                    text = f'{val:.3f}'

                    item = QTableWidgetItem()
                    if show_diff and base_d is not None:
                        base_val = self._get_baseline_val(base_d, m_key, vi, gi)
                        if base_val is not None:
                            delta = val - base_val
                            pct = (delta / abs(base_val) * 100) if abs(base_val) > 1e-9 else 0
                            text = f'{val:.3f}\n({delta:+.3f} {pct:+.1f}%)'
                            if delta > 0.001:
                                item.setBackground(QColor(200, 255, 200))  # green
                            elif delta < -0.001:
                                item.setBackground(QColor(255, 200, 200))  # red
                    item.setText(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(vi, col, item)

    def _get_baseline_val(self, base_d, key, vi, gi):
        try:
            return base_d[key][vi][gi]
        except (IndexError, KeyError):
            return None

    # prefix compatibility — same step + start, one can be longer
    @staticmethod
    def _baseline_compatible(current, baseline, atol=1e-6):
        if baseline is None:
            return False, 'no baseline loaded'
        cv = np.asarray(current.get('v_list', []), dtype=float)
        bv = np.asarray(baseline.get('v_list', []), dtype=float)
        cg = np.asarray(current.get('g_list', []), dtype=float)
        bg = np.asarray(baseline.get('g_list', []), dtype=float)
        n_v = min(len(cv), len(bv))
        n_g = min(len(cg), len(bg))
        if n_v < 1 or n_g < 1:
            return False, 'empty grid'
        if not np.allclose(cv[:n_v], bv[:n_v], atol=atol):
            return False, 'V grid step/start mismatch'
        if not np.allclose(cg[:n_g], bg[:n_g], atol=atol):
            return False, 'g grid step/start mismatch'
        return True, ''

    # grid-point lookup (returns None if (V, g) absent in data's grid)
    @staticmethod
    def _lookup_at(data, V, g, atol=1e-6):
        if data is None:
            return None
        v_arr = data.get('v_list', [])
        g_arr = data.get('g_list', [])
        vi = next((i for i, v in enumerate(v_arr) if abs(v - V) < atol), -1)
        gi = next((i for i, gx in enumerate(g_arr) if abs(gx - g) < atol), -1)
        if vi < 0 or gi < 0:
            return None
        d = data['diamond']
        try:
            return {
                'ax_max': d['ax_max'][vi][gi],
                'ax_min': d['ax_min'][vi][gi],
                'ay_max': d['ay_max'][vi][gi],
                'gg_exponent': d['gg_exponent'][vi][gi],
            }
        except (IndexError, KeyError, TypeError):
            return None

    # ------------------------------------------------------------------ #
    #  Diamond GG plot
    # ------------------------------------------------------------------ #
    def _update_plot(self):
        self.ax_plot.clear()
        # look up by V/g value (combo text), not by index — supports
        # grids of different lengths (prefix-compatible)
        try:
            v_val = float(self.v_combo.currentText())
            g_val = float(self.g_combo.currentText())
        except (ValueError, TypeError):
            self.canvas.draw()
            return

        # 1. baseline (gray dashed)
        if (self.baseline_data is not None
                and self.show_baseline_check.isChecked()):
            b = self._lookup_at(self.baseline_data, v_val, g_val)
            if b is not None:
                self._draw_diamond(
                    self.ax_plot,
                    b['ax_max'], b['ax_min'], b['ay_max'], b['gg_exponent'],
                    color='gray', linestyle='--',
                    label=f'Baseline ({self.baseline_data.get("vehicle_name", "?")})',
                    alpha=0.7)

        # 2. plotted versions (cycling colors)
        for i, snap_name in enumerate(sorted(self.plotted_versions)):
            sd = self.snapshots.get(snap_name)
            v = self._lookup_at(sd, v_val, g_val)
            if v is None:
                continue
            self._draw_diamond(
                self.ax_plot,
                v['ax_max'], v['ax_min'], v['ay_max'], v['gg_exponent'],
                color=_VERSION_COLORS[i % len(_VERSION_COLORS)],
                linestyle='-', label=snap_name, alpha=0.75, linewidth=1.5)

        # 3. current (blue, thick, on top)
        c = self._lookup_at(self.current_data, v_val, g_val)
        if c is not None:
            self._draw_diamond(
                self.ax_plot,
                c['ax_max'], c['ax_min'], c['ay_max'], c['gg_exponent'],
                color='#2196F3', linestyle='-',
                label='Current', alpha=1.0, linewidth=2.5)

        self.ax_plot.set_xlabel('Lateral ay [m/s²]')
        self.ax_plot.set_ylabel('Longitudinal ax [m/s²]')
        self.ax_plot.set_title(
            f'GG Diamond  V={v_val:.1f} m/s  g={g_val:.1f} m/s²')
        self.ax_plot.legend(loc='upper right', fontsize=8)
        # labels every 1.0 m/s² (major), grid lines every 0.5 m/s² (minor)
        self.ax_plot.xaxis.set_major_locator(MultipleLocator(1.0))
        self.ax_plot.yaxis.set_major_locator(MultipleLocator(1.0))
        self.ax_plot.xaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax_plot.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax_plot.grid(True, which='major', alpha=0.35)
        self.ax_plot.grid(True, which='minor', alpha=0.2)
        # fixed ±20 m/s² range on both axes — same scale across all V/g
        # combos for visual comparison. adjustable='box' keeps plot square.
        self.ax_plot.set_xlim(-20.0, 20.0)
        self.ax_plot.set_ylim(-20.0, 20.0)
        self.ax_plot.set_aspect('equal', adjustable='box')
        self.ax_plot.axhline(y=0, color='k', linewidth=0.5)
        self.ax_plot.axvline(x=0, color='k', linewidth=0.5)
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_diamond(self, ax, ax_max, ax_min, ay_max, gg_exp,
                      color='blue', linestyle='-', label=None, alpha=1.0,
                      linewidth=2):
        """Draw a GG diamond envelope using the diamond parameterization."""
        exp = max(gg_exp, 0.5)
        theta = np.linspace(0, 2 * np.pi, 200)
        # diamond parameterization: |ax/ax_lim|^exp + |ay/ay_max|^exp = 1
        ay_pts = ay_max * np.cos(theta)
        ax_pts = np.zeros_like(theta)
        for i, t in enumerate(theta):
            ay_frac = min(abs(ay_pts[i]) / ay_max, 1.0) if ay_max > 1e-9 else 0
            remaining = max(1.0 - ay_frac ** exp, 0.0)
            ax_lim = ax_max if np.sin(t) >= 0 else abs(ax_min)
            ax_pts[i] = ax_lim * remaining ** (1.0 / exp) * np.sign(np.sin(t))
        ax.plot(ay_pts, ax_pts, color=color, linestyle=linestyle,
                linewidth=linewidth, label=label, alpha=alpha)

    # ------------------------------------------------------------------ #
    #  Tuning params tab
    # ------------------------------------------------------------------ #
    def _update_params_tab(self, ref):
        # ref drives the left column (current preferred, baseline fallback).
        ref_is_current = ref is self.current_data
        ref_p = ref.get('tuning_params', {})
        base_p = (self.baseline_data.get('tuning_params', {})
                  if (ref_is_current and self.baseline_data is not None) else {})
        # Header reflects what ref column actually shows
        if ref_is_current:
            self.params_table.setHorizontalHeaderLabels(
                ['Param', 'Current', 'Baseline'])
        else:
            self.params_table.setHorizontalHeaderLabels(
                ['Param', 'Baseline', '(n/a)'])
        all_keys = sorted(set(list(ref_p.keys()) + list(base_p.keys())))

        self.params_table.setRowCount(len(all_keys))
        for i, key in enumerate(all_keys):
            self.params_table.setItem(i, 0, QTableWidgetItem(key))

            cur_val = ref_p.get(key, '')
            base_val = base_p.get(key, '')
            cur_item = QTableWidgetItem(f'{cur_val}')
            base_item = QTableWidgetItem(f'{base_val}')

            if cur_val != '' and base_val != '' and cur_val != base_val:
                cur_item.setBackground(QColor(255, 255, 180))  # highlight changed
            self.params_table.setItem(i, 1, cur_item)
            self.params_table.setItem(i, 2, base_item)

    # ------------------------------------------------------------------ #
    #  Shutdown
    # ------------------------------------------------------------------ #
    def shutdown(self):
        self._sub_results.unregister()
        self._sub_status.unregister()