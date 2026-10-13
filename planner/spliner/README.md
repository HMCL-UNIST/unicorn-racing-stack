# Spliner
Spline-based planners for static obstacles and for the race start.

## Static avoidance (`3d_static_avoidance_node.py`)
Plans a spline around static obstacles ahead of the car. The passing side is chosen from the free space next to the obstacle and the global racing line; the path is smoothed (Savitzky-Golay) and blended back into the racing line, and checked against the 3D track boundaries with [`Track3DValidator`](../../f110_utils/libs/track_3d_validator/README.md) (`safety_margin`).
- Output: `/planner/avoidance/otwpnts`, plus markers under `/planner/avoidance/`.
- Parameters: tuned live through `dynamic_spline_server.py` (`cfg/dyn_spliner_tuner.cfg`): clearance to the obstacle (`evasion_dist`), which obstacles are considered (`obs_traj_tresh`), distance to the track bounds (`spline_bound_mindist`), and the spline shape before/after the apex (`pre_apex_dist*`, `post_apex_dist*`).

## Start planner (`start_quintic_node.py`)
Generates the path from the start position onto the racing line: a Frenet quintic, or a spline through points clicked in RViz (`path_mode`, `cfg/start_quintic.cfg`). When enabled in rqt, the path is sent to the state machine on `/planner/start_wpnts` for the `START` state.

## Usage
Both nodes are started by `3d_headtohead.launch` and `3d_headtohead_fbga.launch` in `stack_master`.
