# 2.5D Velocity Planner
Python library (`vel_planner_25d`) for forward-backward velocity profiles on tracks with slopes. It extends `calc_vel_profile` of the TUM [trajectory_planning_helpers](https://github.com/TUMFTM/trajectory_planning_helpers) and keeps its interface; without the 3D inputs the result is the planar profile.

## 3D extensions
- **Slope**: gravity along the slope reduces acceleration uphill and braking downhill (`slope_correction` scales the effect).
- **Apparent gravity**: grip scales with the apparent gravity (`grip_scale_exp`); at crests the speed is capped where the car would lose contact.
- **Slope speed limit** (optional): `slope_brake_vmax` on slope sections, starting `slope_brake_margin` before them.
- **Smoothing** of the measured pitch (`smooth_mu`).

## API
```python
from vel_planner_25d.vel_planner import calc_vel_profile

vx = calc_vel_profile(ax_max_machines, kappa, el_lengths, closed, drag_coeff, m_veh,
                      b_ax_max_machines, ggv=ggv, v_max=v_max,
                      slope=slope,                    # pitch per point [rad]
                      track_3d_params=track_3d_params)
```
`track_3d_params` holds the 3D track quantities along the path and the optional settings above. `mu` remains the friction per point, as in the original.

## Used by
`stack_master/scripts/global_velocity_planner_3d.py`, `planner/fast_sqp_planner` (`velocity_mode: 2_5d`), `planner/2.5d_gb_optimizer`.
