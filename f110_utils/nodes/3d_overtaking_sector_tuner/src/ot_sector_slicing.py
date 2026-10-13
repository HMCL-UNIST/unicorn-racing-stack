#!/usr/bin/env python3
# 3D version of ot_sector_slicing.py — loads from global_waypoints.json (no topic dependency)
import rospy, rospkg
import yaml, os, subprocess, time, json
from rospy_message_converter import message_converter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

class OvertakingSectorSlicer:
    def __init__(self):
        rospy.init_node('ot_sector_node', anonymous=True)

        self.glob_slider_s = 0
        self.sector_pnts = [0]

        self.yaml_dir = rospy.get_param('~save_dir')

    def load_from_json(self):
        """Load waypoints and bounds from global_waypoints.json"""
        json_path = os.path.join(self.yaml_dir, 'global_waypoints.json')
        print(f'Loading from {json_path}...')
        with open(json_path, 'r') as f:
            d = json.load(f)
        self.glb_wpnts = message_converter.convert_dictionary_to_ros_message(
            'f110_msgs/WpntArray', d['global_traj_wpnts_iqp'])
        self.glb_sp_wpnts = message_converter.convert_dictionary_to_ros_message(
            'f110_msgs/WpntArray', d['global_traj_wpnts_sp'])
        self.track_bounds = message_converter.convert_dictionary_to_ros_message(
            'visualization_msgs/MarkerArray', d['trackbounds_markers'])

    def slice_loop(self):
        self.load_from_json()

        self.sector_gui()
        print('Selected Overtaking Sector IDXs:', self.sector_pnts)

        self.sectors_to_yaml()

    def sector_gui(self):
        x = np.array([w.x_m for w in self.glb_wpnts.wpnts])
        y = np.array([w.y_m for w in self.glb_wpnts.wpnts])
        z = np.array([w.z_m for w in self.glb_wpnts.wpnts])
        s = np.array([w.s_m for w in self.glb_wpnts.wpnts])

        x_sp = np.array([w.x_m for w in self.glb_sp_wpnts.wpnts])
        y_sp = np.array([w.y_m for w in self.glb_sp_wpnts.wpnts])
        z_sp = np.array([w.z_m for w in self.glb_sp_wpnts.wpnts])

        # split right/left bounds by color (right: b=0.5, left: g=1.0)
        r_markers = [m for m in self.track_bounds.markers if m.color.b > 0.4]
        l_markers = [m for m in self.track_bounds.markers if m.color.g > 0.9]
        bnd_rx = np.array([m.pose.position.x for m in r_markers])
        bnd_ry = np.array([m.pose.position.y for m in r_markers])
        bnd_rz = np.array([m.pose.position.z for m in r_markers])
        bnd_lx = np.array([m.pose.position.x for m in l_markers])
        bnd_ly = np.array([m.pose.position.y for m in l_markers])
        bnd_lz = np.array([m.pose.position.z for m in l_markers])

        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_axes([0.05, 0.25, 0.9, 0.7], projection='3d')
        axslider = fig.add_axes([0.15, 0.15, 0.7, 0.03])
        axselect = fig.add_axes([0.15, 0.08, 0.3, 0.05])
        axfinish = fig.add_axes([0.55, 0.08, 0.3, 0.05])

        # save/restore view angle across updates
        self._view = {'elev': 90, 'azim': -90}

        def update_map(cur_s):
            self._view['elev'] = ax1.elev
            self._view['azim'] = ax1.azim
            ax1.cla()
            ax1.plot(x, y, z, 'b-', linewidth=0.7, label='IQP')
            ax1.plot(x_sp, y_sp, z_sp, 'r-', linewidth=0.7, label='SP')
            ax1.plot(bnd_rx, bnd_ry, bnd_rz, 'g-', linewidth=0.4)
            ax1.plot(bnd_lx, bnd_ly, bnd_lz, 'g-', linewidth=0.4)
            ax1.scatter(x[cur_s], y[cur_s], z[cur_s], c='blue', s=50, zorder=10)
            if len(self.sector_pnts) > 0:
                ax1.scatter(x[self.sector_pnts], y[self.sector_pnts], z[self.sector_pnts],
                            c='red', s=50, zorder=10)
            ax1.set_xlabel('x [m]')
            ax1.set_ylabel('y [m]')
            ax1.set_zlabel('z [m]')
            ax1.set_title('OT Sector Slicing (idx=%d, s=%.1fm)' % (cur_s, s[cur_s]))
            ax1.view_init(elev=self._view['elev'], azim=self._view['azim'])
            # equal aspect for x/y/z
            all_x = np.concatenate([x, bnd_rx, bnd_lx])
            all_y = np.concatenate([y, bnd_ry, bnd_ly])
            all_z = np.concatenate([z, bnd_rz, bnd_lz])
            mid_x, mid_y, mid_z = (all_x.max()+all_x.min())/2, (all_y.max()+all_y.min())/2, (all_z.max()+all_z.min())/2
            half = max(all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()) / 2 * 1.05
            ax1.set_xlim(mid_x - half, mid_x + half)
            ax1.set_ylim(mid_y - half, mid_y + half)
            ax1.set_zlim(mid_z - half, mid_z + half)
            ax1.legend(fontsize=7)

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
            # closed-interval [start, end] convention -> last boundary is last valid idx
            self.sector_pnts.append(len(s) - 1)
            self.sector_pnts = sorted(list(set(self.sector_pnts)))

        slider = Slider(axslider, 'Waypoint idx', 0, len(s)-1, valinit=0, valfmt='%d')
        slider.on_changed(update_s)

        btn_select = Button(axselect, 'Select OT S')
        btn_select.on_clicked(select_s)

        btn_finish = Button(axfinish, 'Done')
        btn_finish.on_clicked(finish)

        plt.show()

    def sectors_to_yaml(self):
        if len(self.sector_pnts) == 1:
            # closed-interval fallback -> last boundary is last valid idx
            self.sector_pnts.append(len(self.glb_wpnts.wpnts) - 1)

        n_sectors = len(self.sector_pnts) - 1
        dict_file = {
            'n_sectors': n_sectors,
            'yeet_factor': 1.25,
            'spline_len': 30,
            'ot_sector_begin': 0.5
        }
        for i in range(0, n_sectors):
            dict_file['Overtaking_sector' + str(i)] = {
                'start': self.sector_pnts[i] if i == 0 else self.sector_pnts[i] + 1,
                'end': self.sector_pnts[i+1]}
            dict_file['Overtaking_sector' + str(i)].update({'ot_flag': False})

        # sanity check inclusive [start,end] partition (no gap, no overlap, full coverage)
        N = len(self.glb_wpnts.wpnts)
        assert dict_file['Overtaking_sector0']['start'] == 0, f"Overtaking_sector0.start must be 0, got {dict_file['Overtaking_sector0']['start']}"
        for i in range(n_sectors - 1):
            assert dict_file[f'Overtaking_sector{i+1}']['start'] == dict_file[f'Overtaking_sector{i}']['end'] + 1, \
                f"Overtaking_sector{i+1}.start ({dict_file[f'Overtaking_sector{i+1}']['start']}) != Overtaking_sector{i}.end+1 ({dict_file[f'Overtaking_sector{i}']['end']+1})"
        assert dict_file[f'Overtaking_sector{n_sectors-1}']['end'] == N - 1, \
            f"Last Overtaking_sector.end ({dict_file[f'Overtaking_sector{n_sectors-1}']['end']}) != len(wpnts)-1 ({N-1})"
        for i in range(n_sectors):
            assert dict_file[f'Overtaking_sector{i}']['start'] <= dict_file[f'Overtaking_sector{i}']['end'], \
                f"Overtaking_sector{i} has start>end: ({dict_file[f'Overtaking_sector{i}']['start']},{dict_file[f'Overtaking_sector{i}']['end']})"

        yaml_path = os.path.join(self.yaml_dir, 'ot_sectors.yaml')
        with open(yaml_path, 'w') as file:
            print('Dumping to {}: {}'.format(yaml_path, dict_file))
            yaml.dump(dict_file, file, sort_keys=False)

        ros_path = rospkg.RosPack().get_path('overtaking_sector_tuner_3d')
        yaml_path = os.path.join(ros_path, 'cfg/ot_sectors.yaml')
        with open(yaml_path, 'w') as file:
            print('Dumping to {}: {}'.format(yaml_path, dict_file))
            yaml.dump(dict_file, file, sort_keys=False)

        # rebuild 3D package
        time.sleep(1)
        print('Building overtaking_sector_tuner_3d...')
        shell_dir = os.path.join(ros_path, 'scripts/finish_sector_3d.sh')
        if os.path.exists(shell_dir):
            subprocess.Popen(shell_dir, shell=True)

if __name__ == "__main__":
    ot_sector_slicer = OvertakingSectorSlicer()
    ot_sector_slicer.slice_loop()
