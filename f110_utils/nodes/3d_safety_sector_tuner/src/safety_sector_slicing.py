#!/usr/bin/env python3
# numpy 1.17.4 + pandas 2.0.3 mismatch workaround (matches the head of
# gen_global_racing_line.py / bridge_slicing.py). Strips ~/.local off
# sys.path and spoofs numpy.__version__ so pandas imports cleanly on
# the host's pinned numpy.
import sys as _sys
_sys.path[:] = [p for p in _sys.path if '/.local/' not in p]
import numpy as _np
_np.__version__ = '1.22.4'
if not hasattr(_np.random, 'BitGenerator'):
    from numpy.random.bit_generator import BitGenerator as _BG
    _np.random.BitGenerator = _BG

# Safety-distance sector slicer — friction_sector_slicing.py clone.
# Differences vs friction:
# - Loads <map>/<map>_3d_smoothed.csv (pre-opt centerline) instead of
# global_waypoints.json (post-opt raceline). Per-sector wall margin
# is a track property defined before the raceline NLP runs, just
# like the bridge zones.
# - yaml schema: per-sector {safety_distance, pre_m, post_m} instead
# of {friction, mu_scale_x, mu_scale_y}.
# - Same closed-interval [start, end] partition convention so the
# dyn .cfg can iterate Sector{i} in order.
#
#      Same workflow as friction:
#        1) GUI marks sector boundaries (idx slider + "Select S" + "Done")
#        2) yaml dumped to BOTH <map>/safety_sectors.yaml AND
#           <pkg>/cfg/safety_sectors.yaml
#        3) finish_sector.sh wipes generated Config and rebuilds pkg
#        4) user restarts safety_sector_server.py to pick up new slider set
import os
import sys
import time
import subprocess

import rospy
import rospkg
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class SectorSlicer:
    def __init__(self):
        rospy.init_node('safety_sector_node_3d', anonymous=True)

        self.glob_slider_s = 0
        # closed-interval convention: sector_pnts = [0, b1, b2, ..., N-1].
        # Same as friction_sector_slicing.py — the slicer seeds with 0
        # and appends N-1 on Done, so consecutive entries form
        # Sector{i} = [pnts[i] (+1 for i>0), pnts[i+1]].
        self.sector_pnts = [0]

        self.map_dir = rospy.get_param('~map_dir', '')
        self.map_name = rospy.get_param('~map', '')
        self.smoothed_csv = rospy.get_param('~smoothed_csv', '')
        if not self.map_dir or not self.map_name:
            rospy.logerr('[safety_tuner] map_dir/map params required')
            sys.exit(1)
        if not self.smoothed_csv:
            self.smoothed_csv = os.path.join(
                self.map_dir, f'{self.map_name}_3d_smoothed.csv')
        if not os.path.exists(self.smoothed_csv):
            rospy.logerr(f'[safety_tuner] smoothed CSV not found: '
                         f'{self.smoothed_csv}')
            sys.exit(1)

    def load_smoothed_csv(self):
        df = pd.read_csv(self.smoothed_csv)
        self.s = df['s_m'].to_numpy()
        self.x = df['x_m'].to_numpy()
        self.y = df['y_m'].to_numpy()
        self.z = df['z_m'].to_numpy()
        self.theta = df['theta_rad'].to_numpy()
        self.w_tr_right = df['w_tr_right_m'].to_numpy()
        self.w_tr_left  = df['w_tr_left_m'].to_numpy()
        self.N = len(self.s)

        nx = -np.sin(self.theta)
        ny =  np.cos(self.theta)
        self.bnd_lx = self.x + nx * self.w_tr_left
        self.bnd_ly = self.y + ny * self.w_tr_left
        self.bnd_lz = self.z
        self.bnd_rx = self.x + nx * self.w_tr_right
        self.bnd_ry = self.y + ny * self.w_tr_right
        self.bnd_rz = self.z

        rospy.loginfo(f'[safety_tuner] loaded {self.N} centerline pts from '
                      f'{os.path.basename(self.smoothed_csv)}')

    def slice_loop(self):
        self.load_smoothed_csv()
        self.sector_gui()
        print('Selected Sector IDXs:', self.sector_pnts)
        self.sectors_to_yaml()

    def sector_gui(self):
        s = self.s
        x, y, z = self.x, self.y, self.z

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
            ax1.plot(x, y, z, color='orange', linewidth=0.7)
            ax1.plot(self.bnd_rx, self.bnd_ry, self.bnd_rz, 'g-',
                     linewidth=0.4)
            ax1.plot(self.bnd_lx, self.bnd_ly, self.bnd_lz, 'g-',
                     linewidth=0.4)
            ax1.scatter(x[cur_s], y[cur_s], z[cur_s], c='red', s=50,
                        zorder=10)
            if len(self.sector_pnts) > 0:
                ax1.scatter(x[self.sector_pnts], y[self.sector_pnts],
                            z[self.sector_pnts], c='green', s=50, zorder=10)
            ax1.set_xlabel('x [m]')
            ax1.set_ylabel('y [m]')
            ax1.set_zlabel('z [m]')
            ax1.set_title('Safety Sector Slicing (idx=%d, s=%.1fm)'
                          % (cur_s, s[cur_s]))
            ax1.view_init(elev=self._view['elev'], azim=self._view['azim'])
            all_x = np.concatenate([x, self.bnd_rx, self.bnd_lx])
            all_y = np.concatenate([y, self.bnd_ry, self.bnd_ly])
            all_z = np.concatenate([z, self.bnd_rz, self.bnd_lz])
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
            if idx >= len(s):
                idx = len(s) - 1
            self.glob_slider_s = idx
            update_map(cur_s=idx)
            fig.canvas.draw_idle()

        def select_s(event):
            self.sector_pnts.append(self.glob_slider_s)
            update_map(cur_s=self.glob_slider_s)
            fig.canvas.draw_idle()

        def finish(event):
            plt.close()
            self.sector_pnts.append(len(s) - 1)
            self.sector_pnts = sorted(list(set(self.sector_pnts)))

        slider = Slider(axslider, 'Waypoint idx', 0, len(s) - 1,
                        valinit=0, valfmt='%d')
        slider.on_changed(update_s)

        btn_select = Button(axselect, 'Select S')
        btn_select.on_clicked(select_s)

        btn_finish = Button(axfinish, 'Done')
        btn_finish.on_clicked(finish)

        plt.show()

    def sectors_to_yaml(self):
        if len(self.sector_pnts) == 1:
            self.sector_pnts.append(self.N - 1)
        n_sectors = len(self.sector_pnts) - 1
        # preserve previous per-sector values when slot count matches —
        # hand-edited safety_distance / pre_m / post_m shouldn't reset
        # to 0.20 / 0 / 0 just because the user re-marked positions.
        prev_path = os.path.join(self.map_dir, 'safety_sectors.yaml')
        prev = {}
        if os.path.exists(prev_path):
            try:
                with open(prev_path) as f:
                    prev = yaml.safe_load(f) or {}
            except Exception:
                prev = {}

        # default safety_distance comes from the rqt /gg_tuner slider
        # (the same Raceline `safety_distance` knob users tune for the
        # uniform legacy case). On fresh marking this seeds every
        # sector's safety_distance to the rqt value so the yaml ships
        # consistent with what the user has set up top. Falls back to
        # previous yaml default and finally hardcoded 0.20.
        rqt_default = rospy.get_param('/gg_tuner/safety_distance', None)
        if rqt_default is not None:
            default_d = float(rqt_default)
            rospy.loginfo(
                f'[safety_tuner] seeding safety_distance from rqt slider: '
                f'{default_d:.3f} m')
        else:
            default_d = float(prev.get('safety_distance_default', 0.20))
            rospy.logwarn(
                f'[safety_tuner] /gg_tuner/safety_distance unavailable; '
                f'falling back to previous yaml default {default_d:.3f} m')

        dict_file = {
            'safety_distance_default': default_d,
            'n_sectors': n_sectors,
        }
        for i in range(n_sectors):
            start = self.sector_pnts[i] if i == 0 else self.sector_pnts[i] + 1
            end = self.sector_pnts[i + 1]
            # per-sector safety_distance is *always* seeded with the
            # current rqt default on a fresh marking — the user can
            # then tweak each sector individually via the live
            # /dyn_sector_tuner/safety sliders. pre_m / post_m are
            # preserved from the previous yaml slot if it existed so
            # hand-tuned ramps survive a re-mark.
            prev_sec = prev.get(f'Sector{i}', {}) if isinstance(
                prev.get(f'Sector{i}'), dict) else {}
            dict_file[f'Sector{i}'] = {
                'start': int(start),
                'end':   int(end),
                'safety_distance': float(default_d),
                'pre_m':  float(prev_sec.get('pre_m',  0.0)),
                'post_m': float(prev_sec.get('post_m', 0.0)),
            }

        # same closed-interval partition sanity checks as friction
        assert dict_file['Sector0']['start'] == 0, \
            f"Sector0.start must be 0, got {dict_file['Sector0']['start']}"
        for i in range(n_sectors - 1):
            assert dict_file[f'Sector{i+1}']['start'] == \
                    dict_file[f'Sector{i}']['end'] + 1, \
                f"Sector{i+1}.start != Sector{i}.end+1"
        assert dict_file[f'Sector{n_sectors-1}']['end'] == self.N - 1, \
            f"Last Sector.end != N-1"
        for i in range(n_sectors):
            assert dict_file[f'Sector{i}']['start'] <= \
                    dict_file[f'Sector{i}']['end']

        # 1) dump to map folder (single source of truth for the map)
        yaml_path = os.path.join(self.map_dir, 'safety_sectors.yaml')
        with open(yaml_path, 'w') as f:
            print('Dumping to {}: {}'.format(yaml_path, dict_file))
            yaml.dump(dict_file, f, sort_keys=False)

        # 2) copy to pkg cfg so dyn_safety_tuner.cfg picks it up on
        # the next rebuild (this is what makes rqt show N sliders)
        ros_path = rospkg.RosPack().get_path('safety_sector_tuner_3d')
        pkg_yaml = os.path.join(ros_path, 'cfg/safety_sectors.yaml')
        with open(pkg_yaml, 'w') as f:
            print('Dumping to {}: {}'.format(pkg_yaml, dict_file))
            yaml.dump(dict_file, f, sort_keys=False)

        time.sleep(1)
        print('Building safety_sector_tuner_3d...')
        shell_path = os.path.join(ros_path, 'scripts/finish_sector.sh')
        if os.path.exists(shell_path):
            subprocess.Popen(shell_path, shell=True)


if __name__ == "__main__":
    SectorSlicer().slice_loop()
