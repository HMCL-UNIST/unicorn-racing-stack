# Track 3D Validator
Python library (`track_3d_validator`) that checks whether a planned path stays inside the track on 3D tracks. It replaces the occupancy-grid check (`GridFilter`) of the 2D stack, which fails where the track overlaps itself in XY (bridges, crossings at different heights).

## How it works
`Track3DValidator(converter, gb_wpnts, safety_margin)` uses the global waypoints and the track boundaries loaded in the `FrenetConverter`:
1. **Bound check** — every point's lateral offset `d` is compared with `d_left` / `d_right` of the raceline at its `s`, minus `safety_margin`. This is a fast vectorized pre-filter.
2. **Wall crossing** — only if some point fails the bound check, the path segments are ray-cast against the boundary segments near the same `s` to tell whether the path actually crosses a wall. Selecting the boundary by `s` instead of by XY keeps the layers of a bridge apart. A path that fails the bound check is invalid in either case; this stage only reports the reason.

`validate_trajectory(samples_xy, s_arr, d_arr=None)` returns `(valid, first_invalid_idx, fail_stage)`, where `fail_stage` is `0` (valid), `1` (outside the d-bounds) or `2` (wall crossing). `is_point_valid(x, y, s, d)` checks a single point. `frenet_utils` provides helpers for circular `s` arithmetic (distance, unwrapping, range checks).

## Used by
- `planner/recovery_spliner` (`3d_recovery_spliner_node.py`)
- `planner/spliner` (`3d_static_avoidance_node.py`)
- `planner/sqp_planner` (`3d_sqp_avoidance_node.py`)
- `prediction/gp_traj_predictor` (`3d_opp_prediction.py`)
