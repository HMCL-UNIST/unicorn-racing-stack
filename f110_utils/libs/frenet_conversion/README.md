# Frenet Conversion Library
Conversion between Cartesian coordinates and the Frenet frame (`s` along, `d` across the global waypoints).

- **C++**: `src/frenet_conversion.cc`, used for example by the [Frenet Odom Republisher](../../nodes/frenet_odom_republisher/frenet_odom_republisher/README.md).
- **Python**: `frenet_converter.frenet_converter.FrenetConverter`, with 3D variants (`get_frenet_3d`, `get_cartesian_3d`) that use `z` to tell apart overlapping track layers (bridges). Used by most planner and prediction nodes.
- **Services**: the definitions of the [frenet conversion server](../../nodes/frenet_conversion_server/README.md) (`srv/`).
