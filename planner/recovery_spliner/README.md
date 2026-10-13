# Recovery Spliner
Plans a path back onto the racing line, e.g. after an overtake or when the car is far off the line. The state machine uses it in its RECOVERY state.

## How it works
1. **Target point**: among the raceline points ahead (up to the next inflection point), the one whose heading best matches the direction from the car, so the car merges tangentially.
2. **Spline**: a Bernstein-polynomial spline from the car pose to the target (`spline_scale` sets the end tangents), extended by a short piece along the raceline.
3. **Validation** against the track boundaries before publishing.

The node `3d_recovery_spliner_node.py` validates with `Track3DValidator` (`safety_margin`) and fills the 3D waypoint fields.

## Topics and parameters
- Output: `/planner/recovery/wpnts`, plus markers under `/planner/recovery/`.
- Tuned live through `dynamic_recovery_spline_server.py` (`cfg/dyn_recovery_spliner_tuner.cfg`).

## Usage
Started by the 3D head-to-head launch files in `stack_master` (`3d_headtohead*.launch`).
