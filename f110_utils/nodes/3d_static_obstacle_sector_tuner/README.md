# Static Obstacle Sector Tuner 3D
Per-sector flag for static-obstacle handling (package `static_obstacle_sector_tuner_3d`), read by the state machine and the static avoidance planner.

- `static_obs_sector_slicing.py`: GUI that splits the racing line of `<map>/global_waypoints.json` into sectors, writes `<map>/static_obs_sectors.yaml` and rebuilds the package (`scripts/finish_sector_3d.sh`).
- `static_obs_sector_server.py` (`/dyn_sector_tuner/static_obs`): dynamic_reconfigure server with one flag per sector; `save_params` writes the yaml. Sectors are shown on `/static_obs_sector_markers`.

## Usage
Started like the [speed sector tuner](../3d_sector_tuner/README.md): the server by `3d_base_system.launch`, the slicer from the `gg_tuner` or `3d_sector_slicing.launch`.
