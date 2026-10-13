# ELRS Joy
Driver for an ExpressLRS (ELRS) receiver on a USB-TTL adapter. It parses the CRSF RC-channel frames and publishes them as `sensor_msgs/Joy` in the Xbox layout the stack expects, so an RC transmitter can replace the joystick for manual driving and for switching autonomous mode (`simple_mux`).

Two nodes, one per adapter (the udev rule `.devcontainer/.install_utils/udev_rules_elrs.sh` creates both device links):
- `elrs_joy_ft232_node.py` (FT232RL, `/dev/ELRS_FT232`): exact CRSF baud rate with CRC-8 check. Default in `stack_master/launch/low_level.launch`.
- `elrs_joy_node.py` (CP2102, `/dev/ELRS`): frames are checked by the channel value range only.

## Configuration
- **Mapping**: which CRSF channels feed which Joy axes and buttons, with inversion, deadzone and per-axis calibration (`axes_*`, `button_*`).
- **Safety**: since a corrupted frame could toggle a mode switch, button outputs are debounced, ignored for a short time after a reconnect (`settling_sec`), and the autonomous-start button (RB) only accepts a clean rising edge. On signal loss (`failsafe_timeout`) the axes are zeroed.
- Debug topics: `~debug_channels` (raw channels), `~debug_stats` (FT232 node, frame statistics).

## Usage
```bash
roslaunch elrs_joy elrs_joy_ft232.launch   # FT232RL adapter
roslaunch elrs_joy elrs_joy.launch         # CP2102 adapter
```
