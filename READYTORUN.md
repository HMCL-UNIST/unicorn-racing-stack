# Ready to Run
Check the following on your car before driving. Car-specific configuration lives in `stack_master/config/<car>/`.

## TF
1. The localization TF `map -> odom -> base_link` is published by GLIL; the frame IDs are set in the GLIL configuration (`glim/config/config_ros.json`). `base_link` is the center of the rear axle.
2. The LiDAR–IMU extrinsics are a GLIL parameter (`T_lidar_imu` in `glim/config/config_sensors.json`), not a TF. Measure them on your car and mount the LiDAR rigidly, so that they stay valid at high speed and on slopes.
3. The remaining static transforms (e.g. the VESC IMU) are defined in `stack_master/config/<car>/devices/static_transforms.launch.xml`. If an IMU reports its z-axis pointing toward the ground, publish an intermediate frame that rotates it upward (without translation):
   ```xml
   <node pkg="tf2_ros" type="static_transform_publisher" name="base_link_to_vesc_imu"
         args="0.10 0.0 0.127 0.0 0.0 -0.7071 0.7071 base_link vesc_imu" />
   <node pkg="tf2_ros" type="static_transform_publisher" name="vesc_imu_to_vesc_imu_rot"
         args="0.0 0.0 0.0 0.0 0.0 0.7071 0.7071 vesc_imu vesc_imu_rot" />
   ```
4. Wheel odometry must not publish TF (`vesc_to_odom/publish_tf: false` in `devices/vesc.yaml`).

## Localization (GLIL)
GLIL localizes the car on the 3D prior map from the mapping step, using the point cloud and IMU of the 3D LiDAR. Its pose (`glim_ros/base_odom`) is relayed to `/car_state/odom` by `stack_master/launch/SE/glil_only.launch`. The prior map is set in the GLIL localization configuration; see the [GLIL README](./state_estimation/GLIL_unicorn_racing).
