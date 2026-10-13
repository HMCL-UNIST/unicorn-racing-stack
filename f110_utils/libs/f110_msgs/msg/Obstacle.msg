int32 id

float64 s_start
float64 s_end
float64 d_right           # defined as right bound of the obstacle
float64 d_left            # defined as left bound of the obstacle

float64 x_m
float64 y_m
# add z_m for 3D obstacle position
float64 z_m

bool is_actually_a_gap    # used by the frenet planner to choos points through which we want to go
float64 s_center
float64 d_center
float64 size
float64 vs
float64 vd
float64 s_var
float64 d_var
float64 vs_var
float64 vd_var
bool is_static
bool is_visible

# Static obstacle sector information
int32 sector_id                    # Static obs sector ID (-1 if not in any sector)
bool in_static_obs_sector          # True if in sector with static_obs_section=true

# Fixed path Frenet coordinates (empty/0.0 if Fixed path not available)
float64 s_start_fixed
float64 s_end_fixed
float64 d_right_fixed
float64 d_left_fixed
float64 s_center_fixed
float64 d_center_fixed
float64 vs_fixed
float64 vd_fixed
float64 s_var_fixed
float64 d_var_fixed
float64 vs_var_fixed
float64 vd_var_fixed