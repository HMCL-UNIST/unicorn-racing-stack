# Config
Per-car parameters live in `config/<car>/` and are selected with `racecar_version:=<car>` (or the `CAR_NAME` environment variable). To add a car, copy an existing folder (e.g. `SRX1`) and tune:
- `devices/vesc.yaml`: `speed_to_erpm_gain/offset` (VESC) and `steering_angle_to_servo_gain/offset` (hardware); see the [MuSHR tuning guide](https://mushr.io/tutorials/tuning/). The servo mapping can also be fitted with `3d_base_system.launch calibration:=true`.
- `devices/static_transforms.launch.xml`: static transforms of the sensors (see [READYTORUN.md](../../READYTORUN.md)).
- `devices/imu.yaml`, `devices/lidar.yaml`, `devices/*_mux.yaml`, `devices/joy_teleop.yaml`: drivers and command multiplexers.
- `controller.yaml`: controller parameters.
