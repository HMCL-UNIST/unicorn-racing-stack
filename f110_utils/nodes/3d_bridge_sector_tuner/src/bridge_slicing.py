#!/usr/bin/env python3
# numpy 1.17.4 + pandas 2.0.3 mismatch workaround.
# Mirrors the head of gen_global_racing_line.py — strips broken ~/.local
# from sys.path, spoofs numpy.__version__, and injects BitGenerator if
# missing. Without this pandas import fails on the host's pinned numpy.
import sys as _sys
_sys.path[:] = [p for p in _sys.path if '/.local/' not in p]
import numpy as _np
_np.__version__ = '1.22.4'
if not hasattr(_np.random, 'BitGenerator'):
    from numpy.random.bit_generator import BitGenerator as _BG
    _np.random.BitGenerator = _BG

# Bridge sector slicer — minimal clone of 3d_sector_tuner/sector_slicing.py.
# Same UI: idx slider + "Select S" + "Done". Each "Select S" appends an
# idx to a flat list `sector_pnts`; consecutive pairs form bridges:
# Bridge0 = (sector_pnts[0], sector_pnts[1])
# Bridge1 = (sector_pnts[2], sector_pnts[3])
# ...
# An odd dangling point is dropped on save.
#
#      The bridge *tuning* knobs (pre_m, post_m, chi_max_rad, force_center)
#      are NOT here — they live in rqt (GGTuner.cfg Bridge group) and are
#      applied globally to all bridges in this yaml at raceline-regen time.
#      This matches the established sector-tuner pattern: GUI marks
#      *positions only*, rqt tunes *parameters*.
#
#      Differences from sector_slicing.py:
#        - Loads <map>/<map>_3d_smoothed.csv (pre-opt centerline) instead of
#          global_waypoints.json (post-opt raceline). Bridges are a map
#          property, not a racing-line property.
#        - Yaml schema is {n_bridges: N, Bridge{i}: {start, end}} — no
#          per-bridge tuning fields.
#        - No closed-interval partition assertion (unlike sectors, bridges
#          are *sparse* — a track may have 0, 1, or several disjoint bridge
#          zones).
import os
import sys
import yaml
import numpy as np
import pandas as pd
import rospy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class BridgeSlicer:
    def __init__(self):
        rospy.init_node('bridge_sector_node_3d', anonymous=True)

        self.map_dir = rospy.get_param('~map_dir', '')
        self.map_name = rospy.get_param('~map', '')
        self.smoothed_csv = rospy.get_param('~smoothed_csv', '')

        if not self.map_dir or not self.map_name:
            rospy.logerr('[bridge_tuner] map_dir/map params required')
            sys.exit(1)
        if not self.smoothed_csv:
            self.smoothed_csv = os.path.join(
                self.map_dir, f'{self.map_name}_3d_smoothed.csv')
        if not os.path.exists(self.smoothed_csv):
            rospy.logerr(f'[bridge_tuner] smoothed CSV not found: '
                         f'{self.smoothed_csv}')
            sys.exit(1)

        self.yaml_path = os.path.join(self.map_dir, 'bridge_sectors.yaml')

        # runtime state (same names as sector_slicing.py)
        self.glob_slider_s = 0
        self.sector_pnts = []   # flat list of boundary idxs (pairs → bridges)

    def load_smoothed_csv(self):
        # columns: s_m, x_m, y_m, z_m, theta_rad, mu_rad, phi_rad,
        # dtheta_radpm, dmu_radpm, dphi_radpm,
        # w_tr_right_m, w_tr_left_m, omega_x/y/z_radpm
        df = pd.read_csv(self.smoothed_csv)
        self.s = df['s_m'].to_numpy()
        self.x = df['x_m'].to_numpy()
        self.y = df['y_m'].to_numpy()
        self.z = df['z_m'].to_numpy()
        self.theta = df['theta_rad'].to_numpy()
        self.w_tr_right = df['w_tr_right_m'].to_numpy()
        self.w_tr_left  = df['w_tr_left_m'].to_numpy()
        self.N = len(self.s)

        # reconstruct boundary curves for visualization (centerline +
        # lateral normal). theta is the centerline heading.
        nx = -np.sin(self.theta)
        ny =  np.cos(self.theta)
        self.bnd_lx = self.x + nx * self.w_tr_left
        self.bnd_ly = self.y + ny * self.w_tr_left
        self.bnd_lz = self.z
        self.bnd_rx = self.x + nx * self.w_tr_right
        self.bnd_ry = self.y + ny * self.w_tr_right
        self.bnd_rz = self.z

        rospy.loginfo(f'[bridge_tuner] loaded {self.N} centerline pts from '
                      f'{os.path.basename(self.smoothed_csv)}')

    def load_existing_pts(self):
        # kept for callers/tests that explicitly want to seed the GUI
        # with the existing yaml; NOT used by slice_loop. The GUI is
        # intentionally blank on every launch because users open it
        # "to mark/redraw bridges" — auto-loading old markings made
        # it look like re-launching the tuner already produced
        # bridges. "Done & Save" therefore overwrites the yaml
        # wholesale with whatever was marked in this session.
        if not os.path.exists(self.yaml_path):
            return
        try:
            with open(self.yaml_path, 'r') as f:
                d = yaml.safe_load(f) or {}
        except Exception as e:
            rospy.logwarn(f'[bridge_tuner] yaml parse failed ({e}) — '
                          f'starting empty')
            return
        n = int(d.get('n_bridges', 0))
        for i in range(n):
            b = d.get(f'Bridge{i}')
            if not isinstance(b, dict):
                continue
            try:
                s_i = int(b['start']); e_i = int(b['end'])
            except (KeyError, ValueError, TypeError):
                continue
            s_i = max(0, min(s_i, self.N - 1))
            e_i = max(0, min(e_i, self.N - 1))
            if e_i < s_i:
                s_i, e_i = e_i, s_i
            self.sector_pnts.append(s_i)
            self.sector_pnts.append(e_i)
        if self.sector_pnts:
            rospy.loginfo(f'[bridge_tuner] preloaded {len(self.sector_pnts)//2} '
                          f'bridge(s) from {self.yaml_path}')

    def slice_loop(self):
        self.load_smoothed_csv()
        # GUI always starts blank — no auto-load from existing yaml.
        # Save on "Done" overwrites the yaml with this session's marks
        # only. Users who want to keep old bridges have to remark them.
        self.sector_gui()
        print('Selected boundary IDXs:', self.sector_pnts)
        self.bridges_to_yaml()

    def sector_gui(self):
        # 3D plot + slider + Select/Done — same skeleton as
        # 3d_sector_tuner/sector_slicing.py.
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_axes([0.05, 0.25, 0.9, 0.7], projection='3d')
        axslider = fig.add_axes([0.15, 0.15, 0.7, 0.03])
        axselect = fig.add_axes([0.15, 0.08, 0.3, 0.05])
        axfinish = fig.add_axes([0.55, 0.08, 0.3, 0.05])

        self._view = {'elev': 90, 'azim': -90}

        def update_map(cur_s):
            self._view['elev'] = ax1.elev
            self._view['azim'] = ax1.azim
            ax1.cla()
            ax1.plot(self.x, self.y, self.z, 'r-', linewidth=0.7)
            ax1.plot(self.bnd_rx, self.bnd_ry, self.bnd_rz, 'g-',
                     linewidth=0.4)
            ax1.plot(self.bnd_lx, self.bnd_ly, self.bnd_lz, 'g-',
                     linewidth=0.4)
            # current slider position
            ax1.scatter(self.x[cur_s], self.y[cur_s], self.z[cur_s],
                        c='blue', s=50, zorder=10)
            # committed boundary points (already "Select"-ed)
            if self.sector_pnts:
                ax1.scatter(self.x[self.sector_pnts],
                            self.y[self.sector_pnts],
                            self.z[self.sector_pnts],
                            c='red', s=50, zorder=10)
            # draw a thick magenta segment for each completed bridge
            # (pair of consecutive sector_pnts).
            for i in range(0, len(self.sector_pnts) - 1, 2):
                a = self.sector_pnts[i]
                b = self.sector_pnts[i + 1]
                lo, hi = (a, b) if a <= b else (b, a)
                seg = slice(lo, hi + 1)
                ax1.plot(self.x[seg], self.y[seg], self.z[seg],
                         color='magenta', linewidth=2.0)
            ax1.set_xlabel('x [m]')
            ax1.set_ylabel('y [m]')
            ax1.set_zlabel('z [m]')
            ax1.set_title('Bridge Slicing (idx=%d, s=%.1fm)'
                          % (cur_s, self.s[cur_s]))
            ax1.view_init(elev=self._view['elev'], azim=self._view['azim'])
            # equal aspect (same as sector_slicing.py)
            all_x = np.concatenate([self.x, self.bnd_rx, self.bnd_lx])
            all_y = np.concatenate([self.y, self.bnd_ry, self.bnd_ly])
            all_z = np.concatenate([self.z, self.bnd_rz, self.bnd_lz])
            mid_x = (all_x.max() + all_x.min()) / 2
            mid_y = (all_y.max() + all_y.min()) / 2
            mid_z = (all_z.max() + all_z.min()) / 2
            half = max(all_x.max() - all_x.min(),
                       all_y.max() - all_y.min(),
                       all_z.max() - all_z.min()) / 2 * 1.05
            ax1.set_xlim(mid_x - half, mid_x + half)
            ax1.set_ylim(mid_y - half, mid_y + half)
            ax1.set_zlim(mid_z - half, mid_z + half)

        update_map(0)

        def update_s(val):
            idx = int(slider.val)
            if idx >= len(self.s):
                idx = len(self.s) - 1
            self.glob_slider_s = idx
            update_map(cur_s=idx)
            fig.canvas.draw_idle()

        def select_s(event):
            self.sector_pnts.append(self.glob_slider_s)
            update_map(cur_s=self.glob_slider_s)
            fig.canvas.draw_idle()

        def finish(event):
            plt.close()

        slider = Slider(axslider, 'Waypoint idx', 0, len(self.s) - 1,
                        valinit=0, valfmt='%d')
        slider.on_changed(update_s)

        btn_select = Button(axselect, 'Select S')
        btn_select.on_clicked(select_s)

        btn_finish = Button(axfinish, 'Done')
        btn_finish.on_clicked(finish)

        plt.show()

    def bridges_to_yaml(self):
        # pair consecutive sector_pnts into bridges. Odd dangling point
        # is dropped with a warning so the user knows.
        pts = list(self.sector_pnts)
        if len(pts) % 2 != 0:
            rospy.logwarn(f'[bridge_tuner] odd number of select points '
                          f'({len(pts)}) — dropping last point {pts[-1]}')
            pts = pts[:-1]
        positions = []
        for i in range(0, len(pts), 2):
            a, b = pts[i], pts[i + 1]
            lo, hi = (a, b) if a <= b else (b, a)
            positions.append((int(lo), int(hi)))

        # delegate to shared yaml IO so previously-stored tuning
        # (pre_m, post_m, force_center, chi_max_rad) survives a
        # position-only edit. New bridges (beyond the previous
        # count) get defaults from bridge_yaml_io.
        try:
            import bridge_yaml_io
            bridge_yaml_io.save_positions_preserving_tuning(
                self.map_dir, positions)
        except ImportError:
            # fallback if helper not importable for some reason —
            # write positions only.
            d = {'n_bridges': len(positions)}
            for i, (lo, hi) in enumerate(positions):
                d[f'Bridge{i}'] = {'start': lo, 'end': hi}
            with open(self.yaml_path, 'w') as f:
                yaml.dump(d, f, sort_keys=False)
        rospy.loginfo(f'[bridge_tuner] dumped {len(positions)} bridge(s) → '
                      f'{self.yaml_path}')


if __name__ == '__main__':
    BridgeSlicer().slice_loop()
