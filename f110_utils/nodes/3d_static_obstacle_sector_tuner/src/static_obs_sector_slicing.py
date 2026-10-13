#!/usr/bin/env python3
# 3D version of static_obs_sector_slicing.py — loads from global_waypoints.json (no topic dependency)
import rospy, rospkg
import yaml, os, subprocess, time, json
from rospy_message_converter import message_converter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

class StaticObstacleSectorSlicer:
    def __init__(self):
        rospy.init_node('static_obs_sector_slicer_node', anonymous=True)

        self.sector_pnts_indices = [0]

        self.yaml_dir = rospy.get_param('~save_dir')

    def load_from_json(self):
        """Load waypoints and bounds from global_waypoints.json"""
        json_path = os.path.join(self.yaml_dir, 'global_waypoints.json')
        rospy.loginfo(f'Loading from {json_path}...')
        with open(json_path, 'r') as f:
            d = json.load(f)
        self.glb_wpnts = message_converter.convert_dictionary_to_ros_message(
            'f110_msgs/WpntArray', d['global_traj_wpnts_iqp'])
        self.track_bounds = message_converter.convert_dictionary_to_ros_message(
            'visualization_msgs/MarkerArray', d['trackbounds_markers'])

    def slice_loop(self):
        self.load_from_json()

        self.sector_gui()
        rospy.loginfo(f'Selected Static Obstacle Sector Indices: {self.sector_pnts_indices}')

        self.sectors_to_yaml()

    def sector_gui(self):
        x = np.array([w.x_m for w in self.glb_wpnts.wpnts])
        y = np.array([w.y_m for w in self.glb_wpnts.wpnts])
        z = np.array([w.z_m for w in self.glb_wpnts.wpnts])
        s = np.array([w.s_m for w in self.glb_wpnts.wpnts])

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

        def update_map(cur_idx):
            self._view['elev'] = ax1.elev
            self._view['azim'] = ax1.azim
            ax1.cla()
            ax1.plot(x, y, z, 'm-', linewidth=0.7)
            ax1.plot(bnd_rx, bnd_ry, bnd_rz, 'g-', linewidth=0.4)
            ax1.plot(bnd_lx, bnd_ly, bnd_lz, 'g-', linewidth=0.4)
            ax1.scatter(x[cur_idx], y[cur_idx], z[cur_idx], c='red', s=50, zorder=10)
            if len(self.sector_pnts_indices) > 0:
                ax1.scatter(x[self.sector_pnts_indices], y[self.sector_pnts_indices],
                            z[self.sector_pnts_indices], c='green', s=50, zorder=10)
            ax1.set_xlabel('x [m]')
            ax1.set_ylabel('y [m]')
            ax1.set_zlabel('z [m]')
            ax1.set_title('Static Obs Sector Slicing (idx=%d, s=%.1fm)' % (cur_idx, s[cur_idx]))
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

        update_map(0)

        def update_s(val):
            idx = int(slider.val)
            if idx >= len(s):
                idx = len(s) - 1
            update_map(cur_idx=idx)
            self.glob_slider_idx = idx
            fig.canvas.draw_idle()

        def select_s(event):
            self.sector_pnts_indices.append(self.glob_slider_idx)
            update_map(cur_idx=self.glob_slider_idx)
            fig.canvas.draw_idle()

        def finish(event):
            plt.close()
            self.sector_pnts_indices.append(len(s) - 1)
            self.sector_pnts_indices = sorted(list(set(self.sector_pnts_indices)))

        self.glob_slider_idx = 0

        slider = Slider(axslider, 'Waypoint idx', 0, len(s)-1, valinit=0, valfmt='%d')
        slider.on_changed(update_s)

        btn_select = Button(axselect, 'Select Static Obs S')
        btn_select.on_clicked(select_s)

        btn_finish = Button(axfinish, 'Done')
        btn_finish.on_clicked(finish)

        plt.show()

    def sectors_to_yaml(self):
        if len(self.sector_pnts_indices) <= 1:
            rospy.logwarn("No sectors selected. Creating a single sector for the whole track.")
            self.sector_pnts_indices = [0, len(self.glb_wpnts.wpnts) - 1]

        n_sectors = len(self.sector_pnts_indices) - 1
        dict_file = {'n_sectors': n_sectors}

        for i in range(n_sectors):
            # closed-interval [start, end] convention -> next.start = prev.end + 1 (no overlap)
            start_idx = self.sector_pnts_indices[i] if i == 0 else self.sector_pnts_indices[i] + 1
            end_idx = self.sector_pnts_indices[i + 1]

            s_start = self.glb_wpnts.wpnts[start_idx].s_m
            s_end = self.glb_wpnts.wpnts[end_idx].s_m

            sector_key = f"Static_Obs_sector{i}"
            dict_file[sector_key] = {
                'start': int(start_idx),
                'end': int(end_idx),
                's_start': float(s_start),
                's_end': float(s_end),
                'name': f"sector_{i + 1}",
                'static_obs_section': False
            }

        # sanity check inclusive [start,end] partition (no gap, no overlap, full coverage)
        N = len(self.glb_wpnts.wpnts)
        assert dict_file['Static_Obs_sector0']['start'] == 0, f"Static_Obs_sector0.start must be 0, got {dict_file['Static_Obs_sector0']['start']}"
        for i in range(n_sectors - 1):
            assert dict_file[f'Static_Obs_sector{i+1}']['start'] == dict_file[f'Static_Obs_sector{i}']['end'] + 1, \
                f"Static_Obs_sector{i+1}.start ({dict_file[f'Static_Obs_sector{i+1}']['start']}) != Static_Obs_sector{i}.end+1 ({dict_file[f'Static_Obs_sector{i}']['end']+1})"
        assert dict_file[f'Static_Obs_sector{n_sectors-1}']['end'] == N - 1, \
            f"Last Static_Obs_sector.end ({dict_file[f'Static_Obs_sector{n_sectors-1}']['end']}) != len(wpnts)-1 ({N-1})"
        for i in range(n_sectors):
            assert dict_file[f'Static_Obs_sector{i}']['start'] <= dict_file[f'Static_Obs_sector{i}']['end'], \
                f"Static_Obs_sector{i} has start>end: ({dict_file[f'Static_Obs_sector{i}']['start']},{dict_file[f'Static_Obs_sector{i}']['end']})"

        yaml_path = os.path.join(self.yaml_dir, 'static_obs_sectors.yaml')
        with open(yaml_path, 'w') as file:
            rospy.loginfo(f'Dumping to {yaml_path}: {dict_file}')
            yaml.dump(dict_file, file, default_flow_style=False, sort_keys=False)

        ros_path = rospkg.RosPack().get_path('static_obstacle_sector_tuner_3d')
        cfg_yaml_path = os.path.join(ros_path, 'cfg/static_obs_sectors.yaml')
        with open(cfg_yaml_path, 'w') as file:
            rospy.loginfo(f'Dumping to {cfg_yaml_path}: {dict_file}')
            yaml.dump(dict_file, file, default_flow_style=False, sort_keys=False)

        # rebuild 3D package
        time.sleep(1)
        rospy.loginfo('Building static_obstacle_sector_tuner_3d...')
        shell_dir = os.path.join(ros_path, 'scripts/finish_sector_3d.sh')
        if os.path.exists(shell_dir):
            subprocess.Popen(shell_dir, shell=True)

if __name__ == "__main__":
    slicer = StaticObstacleSectorSlicer()
    slicer.slice_loop()
