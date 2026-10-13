# rqt GG Viewer
rqt plugin that visualizes the GGV diagram of the vehicle as diamonds (maximum longitudinal and lateral acceleration) per speed `V` and apparent gravity `g`. It is the viewer for the GGV stage of the `gg_tuner` node (`stack_master/scripts/gg_tuner_node.py`) and is used to compare a new GGV against earlier versions while tuning.

## Features
- **Diamond + Table** tab: the diamond for the selected `V` / `g` and a summary table over all grid points. When a baseline is shown, the table includes the difference to it.
- **Tuning Params** tab: the vehicle/tuning parameters the GGV was generated with, next to those of the baseline.
- **Baseline and versions**: the snapshots in `planner/3d_gb_optimizer/global_line/data/gg_diagrams/` can be loaded as a baseline or overlaid as additional versions. Grids of different size are compared on their common `V` / `g` points.
- **Save Current as Version** / **Restore Baseline → Latest**: call the `gg_tuner` services to store the current GGV as a new version, or to copy a stored version back to the active (`_latest`) one.

## Input/Output Topic Signature
The plugin subscribes to:
- `/gg_results`: GGV results published (latched) by `gg_tuner`; on startup the latest GGV is also read from disk.
- `/gg_compute_status`: progress messages shown in the status bar.

It calls the services `/gg_tuner/save_version` and `/gg_tuner/restore_version` (`std_srvs/Trigger`).

## Usage
```bash
rosrun stack_master gg_tuner_node.py
rqt --standalone rqt_gg_viewer
```
The plugin is also available in rqt under *Plugins → Visualization → GG Viewer*.
