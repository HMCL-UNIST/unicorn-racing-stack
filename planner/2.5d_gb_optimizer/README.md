# 2.5D Global Optimizer
2.5D variant of the [`gb_optimizer`](../gb_optimizer/README.md) (package `gb_optimizer_25d`). `3d_global_planner_node.py` computes the minimum-curvature (IQP) and shortest-path lines of the TUM [global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization) for a track in `stack_master/maps/<map>/`, with the speed profile of the [2.5D velocity planner](../../f110_utils/libs/2.5d_vel_planner/README.md) for slopes, and publishes and saves the global waypoints like the 2D planner.

For time-optimal racing lines on 3D tracks use the [3D global planner](../3d_gb_optimizer/README.md).

## Usage
```bash
roslaunch stack_master 3d_mapping.launch map:=<map>
```
