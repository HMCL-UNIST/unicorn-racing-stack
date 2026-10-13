# Friction Sector Tuner
Per-sector friction for track parts with different grip (e.g. a bridge surface), used by the velocity planners and the controller.

- `friction_sector_slicing.py`: GUI that splits the racing line of `<map>/global_waypoints.json` into sectors, writes `<map>/friction_scaling.yaml` and rebuilds the package (`scripts/finish_sector.sh`).
- `friction_sector_server.py`: dynamic_reconfigure server with a friction value per sector (optionally split into longitudinal and lateral scale). The values are mirrored to the `/friction_map_params/...` rosparams, published on `/friction_map` and shown on `/friction_sector_markers`; `save_params` writes the yaml.

The rosparams are read by `3d_optimized_vel_planner.py` (`vo_use_friction_scaling`), `vel_planner_25d` and the controller.

## Usage
From the `gg_tuner` (H_Friction): `friction_sector_tuner` opens the slicer, `friction_sector_setting` starts the server. Directly: `roslaunch friction_sector_tuner friction_tuner.launch map:=<map>` (slicer) and `friction_setting.launch map:=<map>` (server).
