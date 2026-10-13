int32 id

# frenet coordinates
float64 s_m
float64 d_m

# map coordinates
float64 x_m
float64 y_m
# add z_m for 3D track-surface elevation. Set by publishers from spline_z(s_m).
float64 z_m

# track information
float64 proj_vs_mps
float64 vd_mps

#variance
float64 d_var
float64 vs_var

# true once a GP posterior has replaced the initial raceline-vx fallback.
# False => proj_vs_mps is the placeholder (raceline vx_mps * 0.9).
bool is_observed

