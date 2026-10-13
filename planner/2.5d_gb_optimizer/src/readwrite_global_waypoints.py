'''
Shared functions to read and write map information (global waypoints)

Previously, the global waypoints obtained during the mapping phase were saved in a rosbag.

Now, this is done using binary files.
'''

import rospkg
import os
import json
from rospy_message_converter import message_converter

from visualization_msgs.msg import MarkerArray
from f110_msgs.msg import WpntArray
from std_msgs.msg import String, Float32
from typing import Tuple, List, Dict

def write_global_waypoints(map_name:str,
    map_info_str:str,
    est_lap_time:Float32,
    centerline_markers:MarkerArray,
    centerline_waypoints:WpntArray,
    global_traj_markers_iqp:MarkerArray,
    global_traj_wpnts_iqp:WpntArray,
    global_traj_markers_sp:MarkerArray,
    global_traj_wpnts_sp:WpntArray,
    trackbounds_markers:MarkerArray,
    map_editor_bool = False,
    # optional velocity cylinder markers (3D mode adds these)
    global_traj_vel_markers_sp:MarkerArray = None
                           )->None:
    '''
    Writes map information to a JSON file with map name specified by `map_name`.
    '''

    # Get path of stack_master package to get the map waypoint path
    r = rospkg.RosPack()
    if not map_editor_bool:
        path = os.path.join(r.get_path('stack_master'), 'maps', map_name, 'global_waypoints.json')
    else:
        path = os.path.join(r.get_path('stack_master'), 'maps', map_name, 'global_waypoints.json')
    print(f"[INFO] WRITE_GLOBAL_WAYPOINTS: Writing global waypoints to {path}")

    # Dictionary will be converted into a JSON for serialization
    d: Dict[str, Dict] = {}
    d['map_info_str'] = {'data': map_info_str}
    d['est_lap_time'] = {'data': est_lap_time}
    d['centerline_markers'] = message_converter.convert_ros_message_to_dictionary(centerline_markers)
    d['centerline_waypoints'] = message_converter.convert_ros_message_to_dictionary(centerline_waypoints)
    d['global_traj_markers_iqp'] = message_converter.convert_ros_message_to_dictionary(global_traj_markers_iqp)
    d['global_traj_wpnts_iqp'] = message_converter.convert_ros_message_to_dictionary(global_traj_wpnts_iqp)
    d['global_traj_markers_sp'] = message_converter.convert_ros_message_to_dictionary(global_traj_markers_sp)
    d['global_traj_wpnts_sp'] = message_converter.convert_ros_message_to_dictionary(global_traj_wpnts_sp)
    d['trackbounds_markers'] = message_converter.convert_ros_message_to_dictionary(trackbounds_markers)

    # write velocity markers if provided (3D mode)
    if global_traj_vel_markers_sp is not None:
        d['global_traj_vel_markers_sp'] = message_converter.convert_ros_message_to_dictionary(global_traj_vel_markers_sp)

    # serialize
    with open(path, 'w') as f:
        json.dump(d, f)

def read_global_waypoints(map_name:str)->Tuple[
    String, Float32, MarkerArray, WpntArray, MarkerArray, WpntArray, MarkerArray, WpntArray, MarkerArray
]:
    '''
    Reads map information from a JSON file with map name specified by `map_name`.
    '''

    # Get path of stack_master package to get the map waypoint path
    r = rospkg.RosPack()
    path = os.path.join(r.get_path('stack_master'), 'maps', map_name, 'global_waypoints.json')

    print(f"[INFO] READ_GLOBAL_WAYPOINTS: Reading global waypoints from {path}")
    # Deserialize JSON and Reconstruct the maps elements
    with open(path, 'r') as f:
        d: Dict[str, List] = json.load(f)

    # backfill psi_centerline_rad on old JSONs (compat with older exports).
    # Wpnt.msg now has float64 psi_centerline_rad (centerline tangent at the
    # wpnt's matched s_opt). Solvers convert d_eff = d_left/cos(psi_rad − psi_centerline_rad).
    # Source priority: (1) wpnt['psi_centerline_rad'] if already present (new exports),
    # (2) centerline_ref.psi_center_rad[k] (intermediate exports), (3) wpnt['psi_rad']
    # with warning (very old exports — racing-line tangent, biased by sin(chi_opt)).
    psi_center_arr = d.get('centerline_ref', {}).get('psi_center_rad')
    _backfill_warned = False
    for key in ('global_traj_wpnts_iqp', 'global_traj_wpnts_sp'):
        if key not in d:
            continue
        for k, w in enumerate(d[key].get('wpnts', [])):
            if 'psi_centerline_rad' in w:
                continue
            if psi_center_arr is not None and k < len(psi_center_arr):
                w['psi_centerline_rad'] = float(psi_center_arr[k])
            else:
                w['psi_centerline_rad'] = float(w.get('psi_rad', 0.0))
                if not _backfill_warned:
                    print(f"[WARN] READ_GLOBAL_WAYPOINTS: '{map_name}' has no centerline_ref.psi_center_rad; "
                          f"falling back to psi_rad (racing-line tangent — boundary corridor will be off "
                          f"by sin(chi_opt) at corners). Re-export to fix.")
                    _backfill_warned = True
    # also backfill centerline_waypoints (its psi_centerline_rad ≡ its own psi_rad)
    if 'centerline_waypoints' in d:
        for w in d['centerline_waypoints'].get('wpnts', []):
            if 'psi_centerline_rad' not in w:
                w['psi_centerline_rad'] = float(w.get('psi_rad', 0.0))

    map_info_str = message_converter.convert_dictionary_to_ros_message('std_msgs/String', d['map_info_str'])
    est_lap_time = message_converter.convert_dictionary_to_ros_message('std_msgs/Float32', d['est_lap_time'])
    centerline_markers = message_converter.convert_dictionary_to_ros_message(
                                            'visualization_msgs/MarkerArray',
                                            d['centerline_markers'])
    centerline_waypoints = message_converter.convert_dictionary_to_ros_message(
                                            'f110_msgs/WpntArray',
                                            d['centerline_waypoints'])
    global_traj_markers_iqp = message_converter.convert_dictionary_to_ros_message(
                                            'visualization_msgs/MarkerArray',
                                            d['global_traj_markers_iqp'])
    global_traj_wpnts_iqp = message_converter.convert_dictionary_to_ros_message(
                                            'f110_msgs/WpntArray',
                                            d['global_traj_wpnts_iqp'])
    global_traj_markers_sp = message_converter.convert_dictionary_to_ros_message(
                                            'visualization_msgs/MarkerArray',
                                            d['global_traj_markers_sp'])
    global_traj_wpnts_sp = message_converter.convert_dictionary_to_ros_message(
                                            'f110_msgs/WpntArray',
                                            d['global_traj_wpnts_sp'])
    trackbounds_markers = message_converter.convert_dictionary_to_ros_message(
                                            'visualization_msgs/MarkerArray',
                                            d['trackbounds_markers'])
    # read velocity markers if present (3D export adds these)
    global_traj_vel_markers_sp = None
    if 'global_traj_vel_markers_sp' in d:
        global_traj_vel_markers_sp = message_converter.convert_dictionary_to_ros_message(
                                                'visualization_msgs/MarkerArray',
                                                d['global_traj_vel_markers_sp'])

    return map_info_str, est_lap_time,\
            centerline_markers, centerline_waypoints, \
            global_traj_markers_iqp, global_traj_wpnts_iqp, \
            global_traj_markers_sp, global_traj_wpnts_sp, trackbounds_markers, \
            global_traj_vel_markers_sp
