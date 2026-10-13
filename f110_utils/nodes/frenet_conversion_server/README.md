# Frenet Conversion Server
ROS services that convert between Cartesian coordinates and the Frenet frame of `/global_waypoints`. Started by `middle_level.launch` (car) and `3d_base_system.launch` (sim).

Services (definitions in `f110_utils/libs/frenet_conversion/srv`):
- `convert_glob2frenet_service`, `convert_glob2frenetarr_service`: `x, y, z` → `s, d, idx`.
- `convert_frenet2glob_service`, `convert_frenet2globarr_service`: `s, d` → `x, y, z`.

Each service also exists with a `_perception` suffix (e.g. `convert_glob2frenetarr_perception_service`) for the perception nodes. The `z` fields are used on 3D tracks to select the right track layer where it overlaps itself (bridges).

```bash
rosservice call /convert_glob2frenetarr_service "{x: [1.1], y: [2.1], z: [0.0]}"
```
In Python, the [frenet conversion library](../../libs/frenet_conversion/README.md) (`FrenetConverter`) gives the same conversion without a service call.
