# Bridge Sector Tuner 3D
Marks bridge zones on the track (package `bridge_sector_tuner_3d`). On a bridge the racing line optimization can keep the line centered and limit its heading, so the car crosses the bridge straight.

- `bridge_slicing.py`: GUI on the smoothed track `<map>/<map>_3d_smoothed.csv`; each pair of selected points is one bridge. It writes the positions to `<map>/bridge_sectors.yaml`.
- `bridge_sector_server.py`: shows the bridges on `/bridge_sector_markers` (started by `3d_base_system.launch`).
- `bridge_yaml_io.py`: read/write helpers for the yaml, shared with the `gg_tuner`.

The constraint settings (extension before/after, force center, heading limit) are not per bridge: they are set in the `gg_tuner` and applied to all bridges by `run_pipeline.sh` (`--bridge-*`).

## Usage
From the `gg_tuner` (B_Bridge): `run_bridge_tuner` opens the GUI, `bridge_constraints_enable` and the `bridge_*` sliders control the constraints. Directly: `roslaunch bridge_sector_tuner_3d bridge_tuner.launch map:=<map>`.
