# Sector Tuner 3D
Per-sector speed scaling of the global racing line (package `sector_tuner_3d`).

- `sector_slicing.py`: GUI that splits the racing line of `<map>/global_waypoints.json` into sectors. It writes `<map>/speed_scaling.yaml` and rebuilds the package (`scripts/finish_sector_3d.sh`) so that dynamic_reconfigure shows one slider per sector.
- `sector_server.py` (`/dyn_sector_tuner/speed`): dynamic_reconfigure server with a global limit and a scaling per sector (plus FTG flags); `save_params` writes the values back to the yaml. Sectors are shown on `/sector_markers`.
- `vel_scaler_node.py`: scales the speed of `/global_waypoints` per sector, blending smoothly across sector borders, and publishes `/global_waypoints_scaled` (and `/shortest_path`).

## Usage
The server and the velocity scaler are started by `3d_base_system.launch` (through `3d_sector_respawn.launch`). The slicer is started from the `gg_tuner` (E_Sectors: `run_sector_slicing`, then `apply_new_sectors` to load the new sectors without restarting), or with `roslaunch stack_master 3d_sector_slicing.launch map:=<map>`.
