# rqt Launch Control
rqt plugin that previews the launch (standing start) motor-current profile `i_cmd(t)` of `simple_mux` live. It reads the launch parameters from the `simple_mux` dynamic_reconfigure server (`stack_master/cfg/SimpleMux.cfg`), recomputes the same profile the node commands, and redraws the curve whenever a slider changes, so the profile can be shaped before testing it on the car.

## Launch profile
`launch_profile_mode` selects the profile:
- `CONSTANT`: a flat current `launch_current_A` for `launch_t_total`.
- `CURVE`: three phases, shaded in the plot:
  1. **Backlash** — linear ramp from 0 to `launch_I_back` until `launch_t_back`, to take up the drivetrain slack gently.
  2. **Exponential rise** — rise from `launch_I_back` towards `launch_I_peak` with time constant `launch_tau`, finished by `launch_t_rise_end`.
  3. **Plateau** — `launch_I_peak` until `launch_t_total`.

The values are edited in rqt_reconfigure (`/vesc/simple_mux`); `save_params` / `load_params` write and read the per-car preset `stack_master/config/<CAR_NAME>/devices/launch_control.yaml`, which `low_level.launch` loads at startup.

## Usage
```bash
rqt --standalone rqt_launch_control
```
The plugin is also available in rqt under *Plugins → Visualization → Launch Control*. The dynamic_reconfigure target can be changed in the plugin (default `/vesc/simple_mux`); without a running server the default values are shown. For bench tests, `stack_master/launch/test_launch_control.launch` starts the VESC stack with a manual command source.
