# Stack Master
The main interface to the stack: launch files, car configurations, maps and tools.

| Folder | Content |
|--------|---------|
| `launch/` | system and race launch files |
| `config/<car>/` | per-car configuration: devices (VESC, IMU, LiDAR, static TFs), controller, lookup tables |
| `maps/<map>/` | per-track files: `wall.pcd`, track CSVs, `global_waypoints.json`, sector yamls |
| `scripts/` | tools and nodes, see [scripts/README.md](./scripts/README.md) |

The track preparation (mapping → racing line) is described in the [main README](../README.md#track-preparation-pipeline).

## Base system
Hardware drivers, GLIL localization, frenet conversion, global waypoints and sector tuners:
```bash
roslaunch stack_master 3d_base_system.launch map:=<map> racecar_version:=<car>
```
- `sim:=true`: no hardware or localization; odometry comes from the Gazebo simulation ([roboracer_gazebo](https://github.com/HangyoCho/roboracer_gazebo)).
- `use_local_racing_line:=true`: event-triggered local racing line (needs acados).
- `detection:=true`: obstacle detection and tracking (external `ra_imm` package).
- `calibration:=true`: servo calibration sweep, updates `vesc.yaml` (see `launch/servo_calibration.launch`).

`low_level.launch` (included) starts the VESC, the LiDAR, the IMU and the RC/joystick receiver; it can also be run alone.

## Head-to-head
Perception, prediction, planners, state machine and controller:
```bash
roslaunch stack_master 3d_headtohead_fbga.launch
```
- `ctrl_algo:=PP|MAP`: Pure Pursuit (default) or MAP; MAP needs a lookup table, selected with `LU_table:=<name>`.
- `fbga_states`: states in which the state machine fills the speed of the local waypoints with FBGA. `3d_headtohead.launch` is the variant without FBGA.

## Tools
- `gg_tuner_node.py`: front end for GGV generation, racing line, sector tools and velocity optimization (see the main README).
- `switch_map.sh` (alias `switch_map <map>`): copies the sector yamls of a map into the sector tuner packages and rebuilds them.
- `boundary_editor_qt.py`: editor for track boundary CSVs.
