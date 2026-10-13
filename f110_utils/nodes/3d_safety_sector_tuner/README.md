# Safety Sector Tuner 3D
Per-sector wall margin (`safety_distance`) for the racing line optimization (package `safety_sector_tuner_3d`), e.g. a larger margin on narrow or risky parts of the track.

- `safety_sector_slicing.py`: GUI on the smoothed track `<map>/<map>_3d_smoothed.csv` (sectors are a track property, defined before the racing line exists). It writes `<map>/safety_sectors.yaml` and rebuilds the package (`scripts/finish_sector.sh`).
- `safety_sector_server.py` (`/dyn_sector_tuner/safety`): dynamic_reconfigure server with `safety_distance` and an extension before/after (`pre_m`, `post_m`) per sector. The values are mirrored to the `/safety_sector_params/...` rosparams; `save_params` writes the yaml.

`gen_global_racing_line.py` uses the per-sector margins when `safety_sector_use` is on, from the rosparams if the server runs and from the yaml otherwise; when it is off, the uniform `safety_distance` applies.

## Usage
From the `gg_tuner` (C_Safety): `safety_sector_tuner` opens the slicer, `safety_sector_setting` starts the server, `safety_sector_use` applies the sectors in the next racing line run. Directly: `roslaunch safety_sector_tuner_3d safety_tuner.launch map:=<map>` and `safety_setting.launch map:=<map>`.
