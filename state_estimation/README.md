# State Estimation
The car is localized with [GLIL](./GLIL_unicorn_racing), a tightly-coupled LiDAR-inertial localization on a 3D prior map based on [GLIM](https://github.com/koide3/glim). It uses the LiDAR point cloud and IMU and works on tracks with slopes and bridges, where a 2D map is ambiguous.

- **Mapping**: GLIM builds the 3D point-cloud map of the track (step 1 of the [track preparation pipeline](../README.md#track-preparation-pipeline)).
- **Localization**: GLIL localizes on this map and publishes the TF `map -> odom -> base_link` and the odometry `glim_ros/base_odom`.
- **Integration**: `stack_master/launch/SE/glil_only.launch` (selected with `SE:=glil_only`, the default of `3d_base_system.launch`) starts GLIL and relays its odometry to `/car_state/odom`.

Configuration (sensor extrinsics, topics, prior map) and build details are in the [GLIL README](./GLIL_unicorn_racing).
