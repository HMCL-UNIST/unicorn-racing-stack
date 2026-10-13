# Stack Master Scripts

## Track and racing line
- `gg_tuner_node.py`: rqt front end for the GGV generation, the racing line, the sector tools and the global velocity engines.
- `3d_optimized_vel_planner.py`: re-optimizes the speed profile of the fixed racing line (NLP on the 3D GGV); `velopt` engine of the gg_tuner.
- `fbga_velocity_planner.py`: global speed profile with FBGA; `fbga` engine of the gg_tuner.
- `global_velocity_planner_3d.py`: speed profile with the [2.5D velocity planner](../../f110_utils/libs/2.5d_vel_planner/README.md), tunable through `cfg/VelPlanner3D.cfg`.
- `boundary_editor_qt.py`: editor for track boundary and 3D track CSVs; uses `regen_center.py` to regenerate the centerline.
- `regen_global_waypoints.py`, `rebuild_waypoints.sh`: regenerate `global_waypoints.json` of a map from its CSVs.
- `switch_map.sh` (alias `switch_map`): copies the sector yamls of a map into the sector tuner packages and rebuilds them.

## Running on the car
- `simple_mux.py`: selects between manual (joystick / RC) and autonomous commands; started by `low_level.launch`.
- `relay_node.py`: relays the topics listed in `config/relay_topics.yaml`.
- `extract_pose_from_odom.py`: publishes `/car_state/pose` from the odometry.
- `fake_odom_publisher.py`: odometry along the racing line for tests without the car or simulator (`test_3d_frenet.launch`).
- `apply_isolation.sh`: CPU isolation of the control and perception nodes from the localization.

## Calibration and recording
- `record_steering_bags.py`, `fit_steering_poly.py`: servo calibration (`calibration:=true` in `3d_base_system.launch`).
- `test_control_publisher.py`: manual launch-control test rig (`test_launch_control.launch`).
- `bag.sh` (alias `record`), `record_with_metrics.sh`: rosbag recording.
