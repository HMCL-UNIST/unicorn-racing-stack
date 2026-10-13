# FBGA Runner
Python wrapper (`fbga_runner.FBGARunner`) around the [FBGA](../FBGA/README.md) forward-backward velocity solver, used to compute speed profiles on the 3D GGV inside ROS nodes.

## How it works
- Uses the FBGA binary, the GGV (`gg.bin`, packed from the 3D planner's GGV) and the vehicle parameters.
- The apparent gravity depends on the speed, so the profile is solved as a fixed-point iteration.
- Runs in-process through the pybind11 module `fbga_native` when it is built (`libs/FBGA/bin/`), otherwise calls the binary. `FBGA_DISABLE_NATIVE=1` forces the binary.
- Options: GGV scaling (`gg_scale`), smoothing of pitch and curvature, slope correction (`bridge_effect`) and an optional speed cap on slopes.

## API
- `solve_open(s, kappa, mu, v_seed=None, v_start=None)`: speed profile on an open path segment.
- `run_once(...)`: a single FBGA solve.
- `refresh_from_disk()`: reloads the GGV after it was regenerated.

## Used by
`state_machine/src/fbga/fbga_state_machine_node.py` (`3d_headtohead_fbga.launch`) and `planner/fast_sqp_planner` (`velocity_mode: 3d`).
