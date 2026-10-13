int32 id

# frenet coordinates
float64 s_m
float64 d_m

# map coordinates
float64 x_m
float64 y_m
# add z_m for 3D terrain height
float64 z_m

# track bound distance
float64 d_right
float64 d_left

# track information
float64 psi_rad
float64 kappa_radpm
float64 vx_mps
float64 ax_mps2

# add mu_rad for 3D slope angle (pitch)
float64 mu_rad

# centerline tangent at this wpnt's matched s_opt.
# d_left/d_right are measured along centerline normal axis, so solvers
# need this for corridor conversion: d_eff = d_left / cos(psi_rad − psi_centerline_rad).
# Also used for accurate boundary marker drawing (centerline-normal direction).
float64 psi_centerline_rad

