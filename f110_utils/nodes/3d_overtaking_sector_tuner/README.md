# Overtaking Sector Tuner 3D
Per-sector overtaking permission (package `overtaking_sector_tuner_3d`). The state machine only overtakes in sectors where it is enabled.

- `ot_sector_slicing.py`: GUI that splits the racing line of `<map>/global_waypoints.json` into sectors, writes `<map>/ot_sectors.yaml` and rebuilds the package (`scripts/finish_sector_3d.sh`).
- `ot_sector_server.py` (`/dyn_sector_tuner/overtake`): dynamic_reconfigure server with an overtaking flag per sector (and an option to disable static-obstacle avoidance there); `save_params` writes the yaml. Sectors are shown on `/ot_sector_markers`.
- `ot_interpolator.py`: publishes the overtaking line `/global_waypoints/overtaking`, where the scaled racing line is blended toward the shortest path in the overtaking sectors and its speed is scaled by `yeet_factor`.

## Usage
Started like the [speed sector tuner](../3d_sector_tuner/README.md): server and interpolator by `3d_base_system.launch`, the slicer from the `gg_tuner` or `3d_sector_slicing.launch`.
