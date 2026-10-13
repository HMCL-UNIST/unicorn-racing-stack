# Fast SQP Planner
Real-time avoidance planner for head-to-head racing. Every cycle it returns a path around the opponent with a speed profile, which the state machine uses both to overtake and to follow the opponent (trailing). When the obstacle is gone, the path converges back to the raceline.

It optimizes the same lateral offset `d(s)` around the GP-predicted opponent as the [SQP planner](../sqp_planner/README.md), but in a form that can be re-solved every cycle:
- **Soft constraints**: obstacle clearance and curvature are penalties in the cost (CasADi + IPOPT) instead of hard SLSQP constraints, so a path is always returned.
- **Warm start** from the previous solution, so consecutive paths stay consistent.
- **Rolling horizon** (`rolling_mode`): a fixed time ahead of the car instead of one full maneuver.

## Pipeline
1. **Path** (`sqp_casadi.py`): the opponent's GP prediction defines the obstacle, its clearance grows with the prediction covariance.
2. **Speed profile** (`velocity_mode`): `2_5d` (`vel_planner_25d`), `nlp` (`nlp_velocity.py`) or `3d` (FBGA, needs `fbga_enable:=true`). With `optimize_velocity` the speed is optimized together with the path instead.
3. **Abort check** (`abort_checker.py`): logs when overtaking is unsafe or slower than trailing; the state machine decides between overtaking and trailing.

Outside the overtaking sectors the planner solves without publishing (dry run, shown in RViz).

## Topics and parameters
- Output: `/planner/fast_sqp_planner/otwpnts` (remapped by the launch files), plus markers and diagnostics under `/planner/fast_sqp_planner/`.
- Parameters: `config/*.yaml`; horizon, obstacle clearance, cost weights and speed settings can be tuned live via dynamic_reconfigure (`cfg/FastSQPPlanner.cfg`).
- IPOPT uses the HSL solver `ma27` if installed, MUMPS otherwise ([INSTALLATION.md](../../INSTALLATION.md#optional-components)).

## Usage
- `launch/fast_sqp_planner.launch`: publishes directly to `/planner/avoidance/otwpnts` in place of the SQP planner.
- `launch/fast_sqp_planner_state.launch`: `attach_to_statemachine:=true` publishes to `/planner/avoidance/otwpnts`, otherwise to `/planner/avoidance/otwpnts_observation` for side-by-side comparison.
- `state_machine/src/fbga/fbga_sm_fast_sqp_node.py`: FBGA state machine whose dynamic-obstacle transitions (GB_TRACK / OVERTAKE / TRAILING) follow this planner's path.
