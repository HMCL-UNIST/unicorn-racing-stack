# 3D Global Planner
Turns a mapped track into a time-optimal racing line on a 3D surface (slopes, banking, bridges). It builds a smoothed 3D track from a driven lap and the segmented walls, computes the vehicle's GGV diagram, and solves a minimum-lap-time problem on that track. The result is `global_waypoints.json`, loaded by `stack_master`. The ROS package is `global_line_3d` (folder `global_line/`).

| Folder | Content |
|--------|---------|
| `global_line/track_processing` | boundary extraction, 3D track generation and smoothing |
| `global_line/global_racing_line` | racing line optimization, waypoint export, FBGA velocity profile |
| `global_line/local_racing_line` | event-triggered local racing line node |
| `global_line/src` | `Track3D` track model, point-mass model, GGV manager |
| `gg_diagram_generation`, `fast_ggv_gen` | GGV diagram generation (full / fast) |

## Track and racing line pipeline
[`global_line/scripts/run_pipeline.sh`](./global_line/scripts/run_pipeline.sh) runs the chain below. In practice steps 1–4 are done on a local PC with [unicorn_racing_tools](https://github.com/HangyoCho/unicorn_racing_tools), which runs the same steps with interactive segmentation and boundary editors; its `<track>_3d_smoothed.csv` is copied into the map folder on the car, and the pipeline runs from step 5 (`--start-from 5`).

| Step | Script | Output |
|------|--------|--------|
| 1. Boundaries | `track_processing/csv_generators/gen_track_boundary_csv.py` | `<track>_bounds_3d.csv` |
| 2. Close loop | (inline) | same file |
| 3. 3D track | `track_processing/gen_3d_track_data_v2.py` | `<track>_3d.csv` |
| 4. Smoothing | `src/track3D.py` | `<track>_3d_smoothed.csv` |
| 5. Racing line | `global_racing_line/gen_global_racing_line.py` | `<track>_3d_<vehicle>_timeoptimal.csv` |
| 6. Export | `global_racing_line/export_global_waypoints.py` | `global_waypoints.json` |
| 7. Plot | `visualization/plot_raceline.py` | figures |

Run from step 1, the pipeline reads the GLIL pose (`/glim_ros/imu_pose`) of one driven lap from a rosbag and searches the segmented `wall.pcd` on both sides of it; the 3D search keeps overlapping layers (bridges, crossings) apart.

Inputs and outputs live in a map folder (`--map-dir stack_master/maps/<map>`: `<map>.bag`, `wall.pcd`), or with `--track <name>` in `global_line/bag/`, `pcd/` and `data/`. Further options select the vehicle (`--vehicle`), the GGV shape (`--gg-mode`), the wall margin (`--safety-distance`), the discretization, bridge constraints (`--bridge-*`), and where to restart (`--start-from`); see the argument parsing in `run_pipeline.sh`.

## GGV diagrams
Maximum accelerations over speed and apparent gravity, stored in `global_line/data/gg_diagrams/<vehicle>/`.
- `gg_diagram_generation/gen_gg_diagrams.py`: full NLP (`3d_global_line.launch run_gg:=true`).
- `fast_ggv_gen/fast_gen_gg_diagrams.py`: builds the NLP once and reuses it, much faster. `--tuning` applies `tuning_<vehicle>.yml` on top of the vehicle parameters.

The IPOPT problems of this package use the HSL linear solver `ma27` when it is installed, MUMPS otherwise. HSL is much faster but not included: it requires a (free) license and each user has to download and build it, see [`fast_ggv_gen/solver`](./fast_ggv_gen/solver/README.md).

## Velocity profile and local racing line
- `global_racing_line/run_fwbw.py`: recomputes the speed profile of a racing line with FBGA on the 3D GGV; wrapped by [`f110_utils/libs/FBGA/run_fwbw.sh`](../../f110_utils/libs/FBGA/run_fwbw.sh).
- `local_racing_line/local_raceline_mux_node.py`: replans a short local racing line when the car deviates too far from the global one. Needs acados (`thirdparty/setup_acados.sh`); enabled with `use_local_racing_line:=true` in `3d_base_system.launch`.

## Usage
The `gg_tuner` node (`rosrun stack_master gg_tuner_node.py`, rqt_reconfigure) is the front end: GGV generation, sector tools, racing line (from step 5) and the optional velocity engine (`fbga` / `velopt`). Directly:
```bash
roslaunch stack_master 3d_global_line.launch map:=<map> vehicle:=rc_car_10th start_from:=5
# or, standalone
cd planner/3d_gb_optimizer/global_line && ./scripts/run_pipeline.sh --track <name>
```
