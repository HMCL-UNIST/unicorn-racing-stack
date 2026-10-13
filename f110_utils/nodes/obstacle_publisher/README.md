# Obstacle Publisher
Test tools that publish dummy obstacles, for running the planners and the state machine without a real opponent.

- `3d_obstacle_publisher.py` (`launch/3d_obstacle_publisher.launch`): a dummy opponent driving along a global trajectory on 3D tracks (`trajectory`, `start_s`, `speed_scaler`). It can oscillate laterally (`d_oscillate`, `reactive`) and mimics the tracker output (detection gate, coasting, position noise, Kalman filter), publishing `/tracking/obstacles` (and the noise-free `/tracking/obstacles_truth`). With `enable_gazebo_static_obs` it also adds the static obstacles of the Gazebo simulation.
- `obstacle_publisher.py`, `obstacle_publisher_grid.py`: the corresponding 2D publishers (`launch/obstacle_publisher.launch`, `obstacle_publisher_grid.launch`).
- `collision_detector.py`: reports collisions between the ego car and the obstacles in Frenet coordinates (`/opponent_collision`, `/opponent_dist`); started by the launch files.

```bash
roslaunch obstacle_publisher 3d_obstacle_publisher.launch trajectory:=min_curv speed_scaler:=0.5
```
