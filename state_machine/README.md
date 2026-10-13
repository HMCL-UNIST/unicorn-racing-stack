# State Machine
Decides the driving behavior from the obstacles, the sectors and the planner outputs, and publishes the local waypoints that the controller follows.

## Nodes
- `3d_state_machine_node.py` (`3d_headtohead.launch`): the 3D state machine. Transitions are in `state_transitions.py`, the waypoints of each state in `states.py`.
- `fbga/fbga_state_machine_node.py` (`3d_headtohead_fbga.launch`): same state machine; in the states listed in `fbga_states` it fills the speed of the local waypoints with an FBGA profile on the same GGV as the global racing line ([fbga_runner](../f110_utils/libs/fbga_runner/README.md)), computed in real time. Speed and GGV scaling per mode are tuned live (`cfg/fbga_tuner.cfg`).
- `fbga/fbga_sm_fast_sqp_node.py`: FBGA state machine whose dynamic-obstacle transitions follow the [fast_sqp_planner](../planner/fast_sqp_planner/README.md) path (`fast_sqp_planner/`).
- `dynamic_statemachine_server.py`, `dynamic_planners_server.py`: dynamic_reconfigure servers of the state machine and planner parameters.

## States
- `GB_TRACK`: follow the global racing line.
- `OVERTAKE`: follow the avoidance path of the planners, only in overtaking sectors.
- `TRAILING`: follow the opponent at a distance.
- `RECOVERY`: return to the racing line with the recovery spliner path.
- `START`: start trajectory from the grid.
- `SMART_STATIC`: static obstacle avoidance path.
- `FTGONLY`, `ATTACK`, `LOSTLINE`: special modes.

## Input/Output Topic Signature
The node subscribes to:
- `/car_state/odom`, `/car_state/odom_frenet`: ego state.
- `/global_waypoints`, `/global_waypoints_scaled`, `/global_waypoints/overtaking`: global waypoints.
- `/tracking/obstacles`, `/opponent_prediction/obstacles_pred`: obstacles and their prediction.
- `/planner/avoidance/otwpnts`, `/planner/avoidance/static_otwpnts`, `/planner/recovery/wpnts`, `/planner/start_wpnts`: planner paths.
- `/dyn_sector_tuner/speed|overtake/parameter_updates`: sectors.

The node publishes:
- `local_waypoints` (+ `/markers`): local waypoints for the controller.
- `state_machine`, `behavior_strategy`: current state and behavior for the other nodes.
- `/ot_section_check`: whether the car is in an overtaking sector.
- `/state_marker`, `/state_machine/*`: visualization.

The state indicator (`state_indicator/`) shows the current state on the car (blink(1) LED).
