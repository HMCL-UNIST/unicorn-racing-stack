import numpy as np
import math
import trajectory_planning_helpers.conv_filt
try:
    import rospy
    HAS_ROSPY = True
except ImportError:
    HAS_ROSPY = False

# optional SG filter for noise-robust mu / dmu_ds in 3D track params
try:
    from scipy.signal import savgol_filter
    HAS_SCIPY_SG = True
except ImportError:
    HAS_SCIPY_SG = False

_g_friction_cache = None  # module-level friction cache


def calc_vel_profile(ax_max_machines: np.ndarray,
                     kappa: np.ndarray,
                     el_lengths: np.ndarray,
                     closed: bool,
                     drag_coeff: float,
                     m_veh: float,
                     b_ax_max_machines: np.ndarray,
                     ggv: np.ndarray = None,
                     loc_gg: np.ndarray = None,
                     v_max: float = None,
                     dyn_model_exp: float = 1.0,
                     mu: np.ndarray = None,
                     v_start: float = None,
                     v_end: float = None,
                     filt_window: int = None,
                     slope: np.ndarray = None,
                     track_3d_params: dict = None,
                     grip_scale_exp: float = 1.0,
                     s_global: np.ndarray = None,
                     smooth_mu: bool = True,                # LPF mu/dmu_ds inside lib
                     mu_smooth_window: int = 21,            # SG window size (odd)
                     mu_smooth_polyorder: int = 3,          # SG polyorder
                     ) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    modified by:
    Tim Stahl

    .. description::
    Calculates a velocity profile using the tire and motor limits as good as possible.

    .. inputs::
    :param ax_max_machines: longitudinal acceleration limits by the electrical motors: [vx, ax_max_machines]. Velocity
                            in m/s, accelerations in m/s2. They should be handed in without considering drag resistance,
                            i.e. simply by calculating F_x_drivetrain / m_veh
    :type ax_max_machines:  np.ndarray
    :param kappa:           curvature profile of given trajectory in rad/m (always unclosed).
    :type kappa:            np.ndarray
    :param el_lengths:      element lengths (distances between coordinates) of given trajectory.
    :type el_lengths:       np.ndarray
    :param closed:          flag to set if the velocity profile must be calculated for a closed or unclosed trajectory.
    :type closed:           bool
    :param drag_coeff:      drag coefficient including all constants: drag_coeff = 0.5 * c_w * A_front * rho_air
    :type drag_coeff:       float
    :param m_veh:           vehicle mass in kg.
    :type m_veh:            float
    :param ggv:             ggv-diagram to be applied: [vx, ax_max, ay_max]. Velocity in m/s, accelerations in m/s2.
                            ATTENTION: Insert either ggv + mu (optional) or loc_gg!
    :type ggv:              np.ndarray
    :param loc_gg:          local gg diagrams along the path points: [[ax_max_0, ay_max_0], [ax_max_1, ay_max_1], ...],
                            accelerations in m/s2. ATTENTION: Insert either ggv + mu (optional) or loc_gg!
    :type loc_gg:           np.ndarray
    :param v_max:           Maximum longitudinal speed in m/s (optional if ggv is supplied, taking the minimum of the
                            fastest velocities covered by the ggv and ax_max_machines arrays then).
    :type v_max:            float
    :param dyn_model_exp:   exponent used in the vehicle dynamics model (usual range [1.0,2.0]).
    :type dyn_model_exp:    float
    :param mu:              friction coefficients (always unclosed).
    :type mu:               np.ndarray
    :param v_start:         start velocity in m/s (used in unclosed case only).
    :type v_start:          float
    :param v_end:           end velocity in m/s (used in unclosed case only).
    :type v_end:            float
    :param filt_window:     filter window size for moving average filter (must be odd).
    :type filt_window:      int

    .. outputs::
    :return vx_profile:     calculated velocity profile (always unclosed).
    :rtype vx_profile:      np.ndarray

    .. notes::
    All inputs must be inserted unclosed, i.e. kappa[-1] != kappa[0], even if closed is set True! (el_lengths is kind of
    closed if closed is True of course!)

    case closed is True:
    len(kappa) = len(el_lengths) = len(mu) = len(vx_profile)

    case closed is False:
    len(kappa) = len(el_lengths) + 1 = len(mu) = len(vx_profile)
    """

    # ------------------------------------------------------------------------------------------------------------------
    # INPUT CHECKS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check if either ggv (and optionally mu) or loc_gg are handed in
    # print(len(b_ax_max_machines))
    
    
    if (ggv is not None or mu is not None) and loc_gg is not None:
        raise RuntimeError("Either ggv and optionally mu OR loc_gg must be supplied, not both (or all) of them!")

    if ggv is None and loc_gg is None:
        raise RuntimeError("Either ggv or loc_gg must be supplied!")

    # check shape of loc_gg
    if loc_gg is not None:
        if loc_gg.ndim != 2:
            raise RuntimeError("loc_gg must have two dimensions!")

        if loc_gg.shape[0] != kappa.size:
            raise RuntimeError("Length of loc_gg and kappa must be equal!")

        if loc_gg.shape[1] != 2:
            raise RuntimeError("loc_gg must consist of two columns: [ax_max, ay_max]!")

    # check shape of ggv
    if ggv is not None and ggv.shape[1] != 3:
        raise RuntimeError("ggv diagram must consist of the three columns [vx, ax_max, ay_max]!")

    # check size of mu
    if mu is not None and kappa.size != mu.size:
        raise RuntimeError("kappa and mu must have the same length!")

    # check size of kappa and element lengths
    if closed and kappa.size != el_lengths.size:
        raise RuntimeError("kappa and el_lengths must have the same length if closed!")

    elif not closed and kappa.size != el_lengths.size + 1:
        raise RuntimeError("kappa must have the length of el_lengths + 1 if unclosed!")

    # check start and end velocities
    if not closed and v_start is None:
        raise RuntimeError("v_start must be provided for the unclosed case!")

    if v_start is not None and v_start < 0.0:
        v_start = 0.0
        # print('WARNING: Input v_start was < 0.0. Using v_start = 0.0 instead!')

    if v_end is not None and v_end < 0.0:
        v_end = 0.0
        print('WARNING: Input v_end was < 0.0. Using v_end = 0.0 instead!')

    # check dyn_model_exp
    if not 1.0 <= dyn_model_exp <= 2.0:
        print('WARNING: Exponent for the vehicle dynamics model should be in the range [1.0, 2.0]!')

    # check shape of ax_max_machines
    if ax_max_machines.shape[1] != 2:
        raise RuntimeError("ax_max_machines must consist of the two columns [vx, ax_max_machines]!")

    # check v_max
    if v_max is None:
        if ggv is None:
            raise RuntimeError("v_max must be supplied if ggv is None!")
        else:
            v_max = min(ggv[-1, 0], ax_max_machines[-1, 0])

    else:
        # check if ggv covers velocity until v_max
        if ggv is not None and ggv[-1, 0] < v_max:
            raise RuntimeError("ggv has to cover the entire velocity range of the car (i.e. >= v_max)!")

        # check if ax_max_machines covers velocity until v_max
        if ax_max_machines[-1, 0] < v_max:
            raise RuntimeError("ax_max_machines has to cover the entire velocity range of the car (i.e. >= v_max)!")

    # ------------------------------------------------------------------------------------------------------------------
    # BRINGING GGV OR LOC_GG INTO SHAPE FOR EQUAL HANDLING AFTERWARDS --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    """For an equal/easier handling of every case afterwards we bring all cases into a form where the local ggv is made
    available for every waypoint, i.e. [ggv_0, ggv_1, ggv_2, ...] -> we have a three dimensional array p_ggv (path_ggv)
    where the first dimension is the waypoint, the second is the velocity and the third is the two acceleration columns
    -> DIM = NO_WAYPOINTS_CLOSED x NO_VELOCITY ENTRIES x 3"""

    # CASE 1: ggv supplied -> copy it for every waypoint
    if ggv is not None:
        p_ggv = np.repeat(np.expand_dims(ggv, axis=0), kappa.size, axis=0)
        op_mode = 'ggv'

    # CASE 2: local gg diagram supplied -> add velocity dimension (artificial velocity of 10.0 m/s)
    else:
        p_ggv = np.expand_dims(np.column_stack((np.ones(loc_gg.shape[0]) * 10.0, loc_gg)), axis=1)
        op_mode = 'loc_gg'

    # ------------------------------------------------------------------------------------------------------------------
    # SPEED PROFILE CALCULATION (FB) -----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # transform curvature kappa into corresponding radii (abs because curvature has a sign in our convention)
    radii = np.abs(np.divide(1.0, kappa, out=np.full(kappa.size, np.inf), where=kappa != 0.0))

    # default slope to zeros if not provided (pure 2D behavior)
    if slope is None:
        slope = np.zeros(kappa.size)

    # Cache friction sector params once (avoid repeated rosparam calls)
    _friction_cache = None
    if HAS_ROSPY:
        try:
            n_sec = rospy.get_param('/friction_map_params/n_sectors', 0)
            if n_sec > 0:
                _friction_cache = {
                    'global_friction_limit': rospy.get_param('/friction_map_params/global_friction_limit', 1.0),
                    'n_sectors': n_sec,
                    'sectors': []
                }
                for si in range(n_sec):
                    _friction_cache['sectors'].append({
                        'start': rospy.get_param(f'/friction_map_params/Sector{si}/start', 0),
                        'end': rospy.get_param(f'/friction_map_params/Sector{si}/end', 0),
                        's_start': rospy.get_param(f'/friction_map_params/Sector{si}/s_start', -1.0),
                        's_end': rospy.get_param(f'/friction_map_params/Sector{si}/s_end', -1.0),
                        'friction': rospy.get_param(f'/friction_map_params/Sector{si}/friction', 1.0),
                    })
        except Exception:
            pass

    # Store friction cache as module-level for access from calc_ax_poss
    global _g_friction_cache
    _g_friction_cache = _friction_cache
    if track_3d_params is not None:
        track_3d_params['_friction_cache'] = _friction_cache

    # Apply sector friction to mu before passing to solvers (both closed and unclosed)
    # Uses s-based lookup when available, falls back to index-based
    if _friction_cache is not None:
        if mu is None:
            mu = np.ones(kappa.size)
        else:
            mu = mu.copy()
        fric_limit = _friction_cache.get('global_friction_limit', 1.0)
        if s_global is not None:
            s_arr = s_global
        else:
            s_arr = np.concatenate([[0], np.cumsum(el_lengths)])
        for j in range(len(mu)):
            s_j = s_arr[j] if j < len(s_arr) else s_arr[-1]
            for sec in _friction_cache['sectors']:
                if sec.get('s_start', -1) >= 0:
                    if sec['s_start'] <= s_j <= sec['s_end']:
                        mu[j] *= min(sec['friction'], fric_limit)
                        break
                else:
                    if sec['start'] <= j <= sec['end']:
                        mu[j] *= min(sec['friction'], fric_limit)
                        break

    # 3D-track-params hygiene (always run when track_3d_params is set)
    # Two stages:
    # (1) closed-loop wrap fix for dmu_ds / omega_* boundary points.
    # np.gradient at the caller side uses 1st-order diff at the seam,
    # which underestimates derivatives there for closed tracks.
    # We always overwrite the 2 boundary samples with wrap-central diff.
    # (2) Optional SG filter on mu (default ON via smooth_mu) to suppress
    # PCD-style noise. Without this, dmu/ds blows up by (2pi/lambda)^2
    # and crashes v at curvature inflections via -omega_y*v^2 in g_tilde.
    # SG window=21 (~2.1m at ds=0.1), polyorder=3 preserves real slope
    # features (>=2m wavelength) while killing cm-scale noise.
    if track_3d_params is not None and 'mu' in track_3d_params:
        ds_lpf = float(el_lengths[0]) if len(el_lengths) > 0 else 0.1
        mu_arr = np.asarray(track_3d_params['mu'], dtype=float)
        kappa_arr = np.asarray(kappa, dtype=float)
        N_mu = len(mu_arr)
        phi_arr = np.asarray(track_3d_params.get('phi', np.zeros_like(mu_arr)),
                              dtype=float)

        # decide whether SG can run
        run_sg = (smooth_mu and HAS_SCIPY_SG
                  and N_mu >= mu_smooth_window)

        if run_sg:
            win = (mu_smooth_window if mu_smooth_window % 2 == 1
                   else mu_smooth_window + 1)
            poly = mu_smooth_polyorder
            if closed:
                pad = win // 2
                mu_pad = np.concatenate([mu_arr[-pad:], mu_arr, mu_arr[:pad]])
                mu_new = savgol_filter(mu_pad, win, poly)[pad:-pad]
                dmu_new = savgol_filter(mu_pad, win, poly,
                                          deriv=1, delta=ds_lpf)[pad:-pad]
            else:
                mu_new = savgol_filter(mu_arr, win, poly)
                dmu_new = savgol_filter(mu_arr, win, poly,
                                          deriv=1, delta=ds_lpf)
        else:
            # SG off: keep mu as-is, but still need dmu_ds for the closed wrap fix
            mu_new = mu_arr
            dmu_new = np.asarray(track_3d_params.get(
                'dmu_ds', np.gradient(mu_arr, ds_lpf)), dtype=float).copy()
            # closed-loop boundary fix: wrap central diff at i=0 and i=N-1
            if closed and N_mu >= 3:
                dmu_new[0] = (mu_arr[1] - mu_arr[-1]) / (2.0 * ds_lpf)
                dmu_new[-1] = (mu_arr[0] - mu_arr[-2]) / (2.0 * ds_lpf)

        # rebuild angular rates with refreshed mu/dmu_ds (phi=0 -> dphi_ds=0)
        omega_y_new = (np.cos(phi_arr) * dmu_new
                        + np.cos(mu_new) * np.sin(phi_arr) * kappa_arr)
        omega_x_new = -np.sin(mu_new) * kappa_arr
        omega_z_new = (-np.sin(phi_arr) * dmu_new
                        + np.cos(mu_new) * np.cos(phi_arr) * kappa_arr)

        # propagate (do not mutate caller's dict)
        track_3d_params = dict(track_3d_params)
        track_3d_params['mu'] = mu_new
        track_3d_params['dmu_ds'] = dmu_new
        track_3d_params['omega_x'] = omega_x_new
        track_3d_params['omega_y'] = omega_y_new
        track_3d_params['omega_z'] = omega_z_new
        if slope is not None:
            slope = mu_new  # slope mirrors mu in the 3D path

    # call solver
    if not closed:
        vx_profile = __solver_fb_unclosed(p_ggv=p_ggv,
                                          ax_max_machines=ax_max_machines,
                                          v_max=v_max,
                                          radii=radii,
                                          el_lengths=el_lengths,
                                          mu=mu,
                                          v_start=v_start,
                                          v_end=v_end,
                                          dyn_model_exp=dyn_model_exp,
                                          drag_coeff=drag_coeff,
                                          m_veh=m_veh,
                                          op_mode=op_mode,
                                          b_ax_max_machines=b_ax_max_machines,
                                          slope=slope,
                                          track_3d_params=track_3d_params)  # slope + g_tilde

    else:
        vx_profile = __solver_fb_closed(p_ggv=p_ggv,
                                        ax_max_machines=ax_max_machines,
                                        v_max=v_max,
                                        radii=radii,
                                        el_lengths=el_lengths,
                                        mu=mu,
                                        dyn_model_exp=dyn_model_exp,
                                        drag_coeff=drag_coeff,
                                        m_veh=m_veh,
                                        op_mode=op_mode,
                                        b_ax_max_machines=b_ax_max_machines,
                                        slope=slope,
                                        track_3d_params=track_3d_params,
                                        grip_scale_exp=grip_scale_exp)  # g_tilde + nonlinear grip

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if filt_window is not None:
        vx_profile = trajectory_planning_helpers.conv_filt.conv_filt(signal=vx_profile,
                                                                     filt_window=filt_window,
                                                                     closed=closed)

    return vx_profile


def __solver_fb_unclosed(p_ggv: np.ndarray,
                         ax_max_machines: np.ndarray,
                         v_max: float,
                         radii: np.ndarray,
                         el_lengths: np.ndarray,
                         v_start: float,
                         drag_coeff: float,
                         m_veh: float,
                         op_mode: str,
                         b_ax_max_machines: np.ndarray,
                         mu: np.ndarray = None,
                         v_end: float = None,
                         dyn_model_exp: float = 1.0,
                         slope: np.ndarray = None,
                         track_3d_params: dict = None) -> np.ndarray:  # g_tilde

    # ------------------------------------------------------------------------------------------------------------------
    # FORWARD BACKWARD SOLVER ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # handle mu
    if mu is None:
        mu = np.ones(radii.size)
        mu_mean = 1.0
    else:
        mu_mean = np.mean(mu)

    # run through all the points and check for possible lateral acceleration
    if op_mode == 'ggv':
        # in ggv mode all ggvs are equal -> we can use the first one
        ay_max_global = mu_mean * np.amin(p_ggv[0, :, 2])   # get first lateral acceleration estimate
        vx_profile = np.sqrt(ay_max_global * radii)         # get first velocity profile estimate

        ay_max_curr = mu * np.interp(vx_profile, p_ggv[0, :, 0], p_ggv[0, :, 2])
        vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))

    else:
        # in loc_gg mode all ggvs consist of a single line due to the missing velocity dependency, mu is None in this
        # case
        vx_profile = np.sqrt(p_ggv[:, 0, 2] * radii)        # get first velocity profile estimate

    # cut vx_profile to car's top speed
    vx_profile[vx_profile > v_max] = v_max

    # consider v_start
    if vx_profile[0] > v_start:
        vx_profile[0] = v_start

    # calculate acceleration profile
    vx_profile = __solver_fb_acc_profile(p_ggv=p_ggv,
                                         ax_max_machines=ax_max_machines,
                                         v_max=v_max,
                                         radii=radii,
                                         el_lengths=el_lengths,
                                         mu=mu,
                                         vx_profile=vx_profile,
                                         backwards=False,
                                         dyn_model_exp=dyn_model_exp,
                                         drag_coeff=drag_coeff,
                                         m_veh=m_veh,
                                         b_ax_max_machines=b_ax_max_machines,
                                         slope=slope,
                                         track_3d_params=track_3d_params)  # g_tilde

    # consider v_end
    if v_end is not None and vx_profile[-1] > v_end:
        vx_profile[-1] = v_end

    # calculate deceleration profile
    vx_profile = __solver_fb_acc_profile(p_ggv=p_ggv,
                                         ax_max_machines=ax_max_machines,
                                         v_max=v_max,
                                         radii=radii,
                                         el_lengths=el_lengths,
                                         mu=mu,
                                         vx_profile=vx_profile,
                                         backwards=True,
                                         dyn_model_exp=dyn_model_exp,
                                         drag_coeff=drag_coeff,
                                         m_veh=m_veh,
                                         b_ax_max_machines=b_ax_max_machines,
                                         slope=slope,
                                         track_3d_params=track_3d_params)  # g_tilde

    return vx_profile


def __solver_fb_closed(p_ggv: np.ndarray,
                       ax_max_machines: np.ndarray,
                       v_max: float,
                       radii: np.ndarray,
                       el_lengths: np.ndarray,
                       drag_coeff: float,
                       m_veh: float,
                       op_mode: str,
                       b_ax_max_machines: np.ndarray,
                       mu: np.ndarray = None,
                       dyn_model_exp: float = 1.0,
                       slope: np.ndarray = None,
                       track_3d_params: dict = None,
                       grip_scale_exp: float = 1.0) -> np.ndarray:  # g_tilde + nonlinear grip

    # ------------------------------------------------------------------------------------------------------------------
    # FORWARD BACKWARD SOLVER ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = radii.size

    # handle mu (friction already applied in calc_vel_profile)
    if mu is None:
        mu = np.ones(no_points)
        mu_mean = 1.0
    else:
        mu_mean = np.mean(mu)

    # run through all the points and check for possible lateral acceleration
    if op_mode == 'ggv':
        # in ggv mode all ggvs are equal -> we can use the first one
        ay_max_global = mu_mean * np.amin(p_ggv[0, :, 2])   # get first lateral acceleration estimate
        vx_profile = np.sqrt(ay_max_global * radii)         # get first velocity estimate (radii must be positive!)

        # iterate until the initial velocity profile converges (break after max. 100 iterations)
        converged = False

        for i in range(100):
            vx_profile_prev_iteration = vx_profile

            ay_max_curr = mu * np.interp(vx_profile, p_ggv[0, :, 0], p_ggv[0, :, 2])
            vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))

            # break the loop if the maximum change of the velocity profile was below 0.5%
            if np.max(np.abs(vx_profile / vx_profile_prev_iteration - 1.0)) < 0.005:
                converged = True
                break

        if not converged:
            print("The initial vx profile did not converge after 100 iterations, please check radii and ggv!")

    else:
        # in loc_gg mode all ggvs consist of a single line due to the missing velocity dependency, mu is None in this
        # case
        vx_profile = np.sqrt(p_ggv[:, 0, 2] * radii)        # get first velocity estimate (radii must be positive!)

    # cut vx_profile to car's top speed
    vx_profile[vx_profile > v_max] = v_max

    # g_tilde + mu based Vmax correction (same approach as 3d_gb_optimizer / FBGA)
    # Skip when grip_scale_exp=0 and slope_correction=0 (pure 2D equivalent)
    #
    # 3d_gb_optimizer constraint: (|ax_tilde|/ax_max)^p + (|ay_tilde|/ay_max)^p <= 1
    # Even at constant speed (ax=0), on a slope ax_tilde = -g*sin(mu) != 0
    # -> less grip is left for ay:
    #   ay_max_eff = ay_max * (1 - (|g*sin(mu)|/ax_max)^p)^(1/p)
    # -> Vmax = sqrt(ay_max_eff / |kappa|)
    #
    # + speed cap at crests (dmu/ds>0) where g_tilde=0
    slope_corr_val = track_3d_params.get('slope_correction', 0.0) if track_3d_params is not None else 0.0
    if track_3d_params is not None and (grip_scale_exp > 0 or slope_corr_val > 0):
        mu_arr = track_3d_params['mu']
        dmu_ds_arr = track_3d_params['dmu_ds']
        omega_y_arr = track_3d_params['omega_y']
        omega_x_arr = track_3d_params['omega_x']
        h_cog = track_3d_params['h']

        for i in range(len(vx_profile)):
            vx_i = vx_profile[i]
            mu_i = mu_arr[i]

            # g_tilde at current speed estimate
            V_omega = omega_y_arr[i] * vx_i * vx_i
            centrifugal = (omega_x_arr[i]**2 - omega_y_arr[i]**2) * h_cog * vx_i**2
            g_tilde_i = max(-V_omega + centrifugal + 9.81 * math.cos(mu_i), 0.0)
            # nonlinear grip scaling (same as calc_ax_poss)
            grip_scale_i = math.pow(g_tilde_i / 9.81, grip_scale_exp) if g_tilde_i > 0 else 0.0

            # Friction is already applied via mu array (set in __solver_fb_closed)
            # No additional friction scaling in grip_scale_i here

            # ay_max with grip_scale, ax_max WITHOUT grip_scale (for diamond ratio)
            # Use per-waypoint mu[i] (includes friction) instead of mu_mean
            mu_i_fric = mu[i] if i < len(mu) else mu_mean
            ay_max_i = mu_i_fric * np.interp(vx_i, p_ggv[0, :, 0], p_ggv[0, :, 2]) * grip_scale_i
            ax_max_raw = mu_i_fric * np.interp(vx_i, p_ggv[0, :, 0], p_ggv[0, :, 1])  # original, no grip_scale

            # (1) diamond constraint: mu gravity vs original tire ax capacity
            slope_corr = track_3d_params.get('slope_correction', 1.0)
            ax_grav = abs(9.81 * math.sin(mu_i)) * slope_corr
            p = dyn_model_exp
            if ax_max_raw > 1e-6 and ax_grav > 1e-6:
                ratio = min(ax_grav / ax_max_raw, 1.0)
                ay_max_eff = ay_max_i * math.pow(max(1.0 - math.pow(ratio, p), 0.0), 1.0 / p)
            else:
                ay_max_eff = ay_max_i

            # (2) Vmax from lateral grip
            if radii[i] < 1e4 and ay_max_eff > 0:
                v_lat = math.sqrt(ay_max_eff * radii[i])
                vx_profile[i] = min(vx_profile[i], v_lat)

            # (3) crest (dmu_ds > 0): g_tilde=0 speed limit (only when slope_correction > 0)
            if slope_corr > 0:
                dmu_i = dmu_ds_arr[i]
                if dmu_i > 1e-4:
                    v_gtilde_max = math.sqrt(9.81 * math.cos(mu_i) / dmu_i)
                    vx_profile[i] = min(vx_profile[i], v_gtilde_max)

    # Friction Vmax correction removed — friction is already in mu array for both 2D and 3D

    # Pre-slope braking: Vmax from margin before slope entry through slope end
    if track_3d_params is not None:
        brake_margin = track_3d_params.get('slope_brake_margin', 0.0)
        brake_vmax = track_3d_params.get('slope_brake_vmax', 5.0)
        if brake_margin > 0:
            mu_arr = track_3d_params['mu']
            ds = el_lengths[0] if len(el_lengths) > 0 else 0.1
            slope_threshold = math.radians(2.0)
            in_slope = np.abs(mu_arr) > slope_threshold
            slope_diff = np.diff(in_slope.astype(int))
            slope_entries = np.where(slope_diff == 1)[0] + 1
            slope_exits = np.where(slope_diff == -1)[0] + 1
            margin_pts = int(round(brake_margin / ds))
            for entry_idx in slope_entries:
                exits_after = slope_exits[slope_exits > entry_idx]
                exit_idx = exits_after[0] if len(exits_after) > 0 else len(vx_profile)
                start_idx = max(0, entry_idx - margin_pts)
                end_idx = min(len(vx_profile), exit_idx)
                for j in range(start_idx, end_idx):
                    vx_profile[j] = min(vx_profile[j], brake_vmax)

    """Use 3 laps and extract the middle lap to avoid boundary effects."""

    # Triple arrays (3 laps)
    vx_profile_triple = np.concatenate((vx_profile, vx_profile, vx_profile), axis=0)
    radii_triple = np.concatenate((radii, radii, radii), axis=0)
    el_lengths_triple = np.concatenate((el_lengths, el_lengths, el_lengths), axis=0)
    mu_triple = np.concatenate((mu, mu, mu), axis=0)
    p_ggv_triple = np.concatenate((p_ggv, p_ggv, p_ggv), axis=0)
    # triple slope for 3-lap boundary effect removal
    slope_triple = np.concatenate((slope, slope, slope), axis=0)
    # triple track_3d_params arrays
    if track_3d_params is not None:
        track_3d_params_triple = {}
        for key, val in track_3d_params.items():
            if isinstance(val, np.ndarray):
                track_3d_params_triple[key] = np.concatenate((val, val, val), axis=0)
            else:
                track_3d_params_triple[key] = val  # scalar (e.g. h)
    else:
        track_3d_params_triple = None

    # Forward pass on 3 laps
    vx_profile_triple = __solver_fb_acc_profile(p_ggv=p_ggv_triple,
                                                ax_max_machines=ax_max_machines,
                                                v_max=v_max,
                                                radii=radii_triple,
                                                el_lengths=el_lengths_triple,
                                                mu=mu_triple,
                                                vx_profile=vx_profile_triple,
                                                backwards=False,
                                                dyn_model_exp=dyn_model_exp,
                                                drag_coeff=drag_coeff,
                                                m_veh=m_veh,
                                                b_ax_max_machines=b_ax_max_machines,
                                                slope=slope_triple,
                                                track_3d_params=track_3d_params_triple,
                                                grip_scale_exp=grip_scale_exp)  # g_tilde + nonlinear grip

    # Backward pass on 3 laps
    vx_profile_triple = __solver_fb_acc_profile(p_ggv=p_ggv_triple,
                                                ax_max_machines=ax_max_machines,
                                                v_max=v_max,
                                                radii=radii_triple,
                                                el_lengths=el_lengths_triple,
                                                mu=mu_triple,
                                                vx_profile=vx_profile_triple,
                                                backwards=True,
                                                dyn_model_exp=dyn_model_exp,
                                                drag_coeff=drag_coeff,
                                                m_veh=m_veh,
                                                b_ax_max_machines=b_ax_max_machines,
                                                slope=slope_triple,
                                                track_3d_params=track_3d_params_triple,
                                                grip_scale_exp=grip_scale_exp)  # g_tilde + nonlinear grip

    # Extract middle lap (lap 2) - this has proper boundary conditions from both sides
    vx_profile = vx_profile_triple[no_points:2*no_points]

    # Check continuity at closing point
    final_mismatch = abs(vx_profile[-1] - vx_profile[0])
    if final_mismatch < 0.1:
        print(f"Closed loop converged (mismatch: {final_mismatch:.3f} m/s)")
    else:
        print(f"WARNING: Closed loop mismatch: {final_mismatch:.3f} m/s")
    return vx_profile
        
    

def __solver_fb_acc_profile(p_ggv: np.ndarray,
                            ax_max_machines: np.ndarray,
                            v_max: float,
                            radii: np.ndarray,
                            el_lengths: np.ndarray,
                            mu: np.ndarray,
                            vx_profile: np.ndarray,
                            drag_coeff: float,
                            m_veh: float,
                            b_ax_max_machines: np.ndarray,
                            dyn_model_exp: float = 1.0,
                            backwards: bool = False,
                            slope: np.ndarray = None,
                            track_3d_params: dict = None,
                            grip_scale_exp: float = 1.0) -> np.ndarray:  # g_tilde + nonlinear grip

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = vx_profile.size

    # default slope to zeros if not provided
    if slope is None:
        slope = np.zeros(no_points)

    # flipud track_3d_params arrays for backward pass
    # Note: the backward pass does not mean the car drives in reverse.
    # The algorithm only sweeps the track in reverse order; the physical slope direction is unchanged.
    # So the arrays are only flipped (flipud), without a sign change.
    if track_3d_params is not None:
        if backwards:
            t3d_mod = {}
            for key, val in track_3d_params.items():
                t3d_mod[key] = np.flipud(val) if isinstance(val, np.ndarray) else val
        else:
            t3d_mod = track_3d_params
    else:
        t3d_mod = None

    # check for reversed direction
    if backwards:
        radii_mod = np.flipud(radii)
        el_lengths_mod = np.flipud(el_lengths)
        mu_mod = np.flipud(mu)
        # slope is only reordered here; the gravity sign reversal for the backward
        # pass is applied to ax_gravity in calc_ax_poss (mode == 'decel_backw').
        vx_profile = np.flipud(vx_profile)
        slope_mod = np.flipud(slope)  # flip slope order for backward pass (no sign change)
        mode = 'decel_backw'
    else:
        radii_mod = radii
        el_lengths_mod = el_lengths
        mu_mod = mu
        slope_mod = slope
        mode = 'accel_forw'

    # ------------------------------------------------------------------------------------------------------------------
    # SEARCH START POINTS FOR ACCELERATION PHASES ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    vx_diffs = np.diff(vx_profile)
    acc_inds = np.where(vx_diffs > 0.0)[0]                  # indices of points with positive acceleration
    if acc_inds.size != 0:
        # check index diffs -> we only need the first point of every acceleration phase
        acc_inds_diffs = np.diff(acc_inds)
        acc_inds_diffs = np.insert(acc_inds_diffs, 0, 2)    # first point is always a starting point
        acc_inds_rel = acc_inds[acc_inds_diffs > 1]         # starting point indices for acceleration phases
    else:
        acc_inds_rel = []                                   # if vmax is low and can be driven all the time

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE VELOCITY PROFILE ---------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # cast np.array as a list
    acc_inds_rel = list(acc_inds_rel)

    # while we have indices remaining in the list
    while acc_inds_rel:
        # set index to first list element
        i = acc_inds_rel.pop(0)

        # start from current index and run until either the end of the lap or a termination criterion are reached
        while i < no_points - 1:

            ax_possible_cur = calc_ax_poss(vx_start=vx_profile[i],
                                           radius=radii_mod[i],
                                           ggv=p_ggv[i],
                                           ax_max_machines=ax_max_machines,
                                           mu=mu_mod[i],
                                           mode=mode,
                                           dyn_model_exp=dyn_model_exp,
                                           drag_coeff=drag_coeff,
                                           m_veh=m_veh,
                                           b_ax_max_machines=b_ax_max_machines,
                                           slope=slope_mod[i],
                                           track_3d_params=t3d_mod,
                                           point_idx=i,
                                           grip_scale_exp=grip_scale_exp)  # g_tilde + nonlinear grip

            vx_possible_next_sq = math.pow(vx_profile[i], 2) + 2 * ax_possible_cur * el_lengths_mod[i]
            vx_possible_next = math.sqrt(max(vx_possible_next_sq, 0.0))

            if backwards:
                """
                We have to loop the calculation if we are in the backwards iteration (currently just once). This is 
                because we calculate the possible ax at a point i which does not necessarily fit for point i + 1 
                (which is i - 1 in the real direction). At point i + 1 (or i - 1 in real direction) we have a different 
                start velocity (vx_possible_next), radius and mu value while the absolute value of ax remains the same 
                in both directions.
                """

                # looping just once at the moment
                for j in range(1):
                    ax_possible_next = calc_ax_poss(vx_start=vx_possible_next,
                                                    radius=radii_mod[i + 1],
                                                    ggv=p_ggv[i + 1],
                                                    ax_max_machines=ax_max_machines,
                                                    mu=mu_mod[i + 1],
                                                    mode=mode,
                                                    dyn_model_exp=dyn_model_exp,
                                                    drag_coeff=drag_coeff,
                                                    m_veh=m_veh,
                                                    b_ax_max_machines=b_ax_max_machines,
                                                    slope=slope_mod[i + 1],
                                                    track_3d_params=t3d_mod,
                                                    point_idx=i + 1,
                                                    grip_scale_exp=grip_scale_exp)  # g_tilde + nonlinear grip

                    vx_tmp_sq = math.pow(vx_profile[i], 2) + 2 * ax_possible_next * el_lengths_mod[i]
                    vx_tmp = math.sqrt(max(vx_tmp_sq, 0.0))

                    if vx_tmp < vx_possible_next:
                        vx_possible_next = vx_tmp
                    else:
                        break

            # save possible next velocity if it is smaller than the current value
            if vx_possible_next < vx_profile[i + 1]:
                vx_profile[i + 1] = vx_possible_next

            i += 1

            # break current acceleration phase if next speed would be higher than the maximum vehicle velocity or if we
            # are at the next acceleration phase start index
            if vx_possible_next > v_max or (acc_inds_rel and i >= acc_inds_rel[0]):
                break

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # flip output vel_profile if necessary
    if backwards:
        vx_profile = np.flipud(vx_profile)

    return vx_profile


def calc_ax_poss(vx_start: float,
                 radius: float,
                 ggv: np.ndarray,
                 mu: float,
                 dyn_model_exp: float,
                 drag_coeff: float,
                 m_veh: float,
                 b_ax_max_machines: np.ndarray,
                 ax_max_machines: np.ndarray = None,
                 mode: str = 'accel_forw',
                 slope: float = 0.0,
                 track_3d_params: dict = None,
                 point_idx: int = 0,
                 grip_scale_exp: float = 1.0,
                 s_m: float = -1.0) -> float:  # s position for friction sector lookup (-1=use point_idx)
    """
    This function returns the possible longitudinal acceleration in the current step/point.

    .. inputs::
    :param vx_start:        [m/s] velocity at current point
    :type vx_start:         float
    :param radius:          [m] radius on which the car is currently driving
    :type radius:           float
    :param ggv:             ggv-diagram to be applied: [vx, ax_max, ay_max]. Velocity in m/s, accelerations in m/s2.
    :type ggv:              np.ndarray
    :param mu:              [-] current friction value
    :type mu:               float
    :param dyn_model_exp:   [-] exponent used in the vehicle dynamics model (usual range [1.0,2.0]).
    :type dyn_model_exp:    float
    :param drag_coeff:      [m2*kg/m3] drag coefficient incl. all constants: drag_coeff = 0.5 * c_w * A_front * rho_air
    :type drag_coeff:       float
    :param m_veh:           [kg] vehicle mass
    :type m_veh:            float
    :param ax_max_machines: longitudinal acceleration limits by the electrical motors: [vx, ax_max_machines]. Velocity
                            in m/s, accelerations in m/s2. They should be handed in without considering drag resistance.
                            Can be set None if using one of the decel modes.
    :type ax_max_machines:  np.ndarray
    :param mode:            [-] operation mode, can be 'accel_forw', 'decel_forw', 'decel_backw'
                            -> determines if machine limitations are considered and if ax should be considered negative
                            or positive during deceleration (for possible backwards iteration)
    :type mode:             str

    .. outputs::
    :return ax_final:       [m/s2] final acceleration from current point to next one
    :rtype ax_final:        float
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check inputs
    if mode not in ['accel_forw', 'decel_forw', 'decel_backw']:
        raise RuntimeError("Unknown operation mode for calc_ax_poss!")

    if mode == 'accel_forw' and ax_max_machines is None:
        raise RuntimeError("ax_max_machines is required if operation mode is accel_forw!")

    if ggv.ndim != 2 or ggv.shape[1] != 3:
        raise RuntimeError("ggv must have two dimensions and three columns [vx, ax_max, ay_max]!")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER TIRE POTENTIAL ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # g_tilde — effective gravity considering 3D terrain effects
    # Based on TUMRT calc_apparent_accelerations (track3D.py L860)
    # g_tilde = fmax(w_dot - V_omega + (omega_x² - omega_y²)*h + g*cos(mu)*cos(phi), 0)
    g = 9.81

    if track_3d_params is not None:
        i = point_idx
        mu_t = track_3d_params['mu'][i]
        phi_t = track_3d_params['phi'][i]
        omega_x_t = track_3d_params['omega_x'][i]
        omega_y_t = track_3d_params['omega_y'][i]
        h_t = track_3d_params['h']

        # V_omega: velocity x road rotation interaction (dominant term for convex/concave terrain)
        # V_omega = (-Omega_x*sin(chi) + Omega_y*cos(chi)) * s_dot * V
        # chi ≈ 0 (raceline close to centerline direction), s_dot ≈ V
        V_omega = omega_y_t * vx_start * vx_start

        # centrifugal: (omega_x² - omega_y²) * h
        centrifugal = (omega_x_t**2 - omega_y_t**2) * h_t * vx_start**2

        # w_dot ≈ 0 (requires n and Omega_x, both small for no-bank tracks)

        # g_tilde = fmax(w_dot - V_omega + centrifugal + g*cos(mu)*cos(phi), 0)
        g_tilde = max(-V_omega + centrifugal + g * math.cos(mu_t) * math.cos(phi_t), 0.0)
        # nonlinear grip scaling: grip_scale = (g_tilde / g) ^ grip_scale_exp
        if grip_scale_exp == 0:
            grip_scale = 1.0  # g_tilde disabled
        elif g_tilde > 0:
            grip_scale = math.pow(g_tilde / g, grip_scale_exp)
        else:
            grip_scale = 0.0
    else:
        # fallback: simple cos(slope) scaling with nonlinear exponent
        grip_scale = math.pow(math.cos(slope), grip_scale_exp)

    # Friction is already applied via mu array (set in __solver_fb_closed)
    # No additional friction scaling in grip_scale here to avoid double-apply

    # ax_tilde constraint inside the Kamm circle (same as 3d_gb_optimizer)
    #
    # Physics: ax_tilde = ax - g*sin(mu), constraint: |ax_tilde| <= ax_max_grip
    #       -> the g*sin(mu) share of the tire grip consumed by gravity is accounted for inside the Kamm circle
    #
    # The backward pass is a reversed (flipud) simulation:
    #   driving in reverse swaps uphill and downhill -> ax_gravity sign flips
    #
    # forward:  ax_gravity = +g*sin(mu)  (uphill mu<0 -> hinders acceleration)
    # backward: ax_gravity = -g*sin(mu)  (uphill becomes downhill -> helps acceleration)

    # slope_correction: scale factor for longitudinal gravity effect (tunable)
    slope_corr = track_3d_params.get('slope_correction', 1.0) if track_3d_params is not None else 1.0
    ax_gravity = g * math.sin(slope) * slope_corr
    if mode == 'decel_backw':
        ax_gravity = -ax_gravity

    # --- tire grip limits (tilde frame, scaled by g_tilde) ---
    ax_max_tires_tilde = mu * np.interp(vx_start, ggv[:, 0], ggv[:, 1]) * grip_scale
    ay_max_tires = mu * np.interp(vx_start, ggv[:, 0], ggv[:, 2]) * grip_scale
    ay_used = math.pow(vx_start, 2) / radius

    if mode in ['accel_forw', 'decel_backw'] and ax_max_tires_tilde < 0.0:
        print("WARNING: Inverting sign of ax_max_tires because it should be positive but was negative!")
        ax_max_tires_tilde *= -1.0
    elif mode == 'decel_forw' and ax_max_tires_tilde > 0.0:
        print("WARNING: Inverting sign of ax_max_tires because it should be negative but was positve!")
        ax_max_tires_tilde *= -1.0

    # --- Kamm circle in tilde frame ---
    # if ay_max_tires is 0 there is no grip -> ax is 0 as well
    if abs(ay_max_tires) < 1e-6:
        ax_avail_tilde = 0.0
    else:
        radicand = 1.0 - math.pow(ay_used / ay_max_tires, dyn_model_exp)
        if radicand > 0.0:
            ax_avail_tilde = ax_max_tires_tilde * math.pow(radicand, 1.0 / dyn_model_exp)
        else:
            ax_avail_tilde = 0.0

    # --- tilde → real: ax_real = ax_tilde + g*sin(mu) ---
    ax_avail_tires = ax_avail_tilde + ax_gravity

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER MACHINE LIMITATIONS -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Machine limits are also constrained by g_tilde (grip_scale) and tire-road friction (mu)
    # mu already contains friction from sector tuner, grip_scale contains g_tilde only
    if mode == 'accel_forw':
        ax_max_machines_tmp = np.interp(vx_start, ax_max_machines[:, 0], ax_max_machines[:, 1]) * grip_scale * mu
        ax_avail_vehicle = min(ax_avail_tires, ax_max_machines_tmp)
    else:
        bx_max_machines_tmp = np.interp(vx_start, b_ax_max_machines[:, 0], b_ax_max_machines[:, 1]) * grip_scale * mu
        ax_avail_vehicle = min(ax_avail_tires, bx_max_machines_tmp)

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER DRAG ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    ax_drag = -math.pow(vx_start, 2) * drag_coeff / m_veh

    # gravity is already included in ax_avail_tires; only drag is handled per mode
    # forward: drag as is (decelerating)
    # backward: drag reversed (reverse simulation)
    if mode in ['accel_forw', 'decel_forw']:
        ax_final = ax_avail_vehicle + ax_drag
    else:
        ax_final = ax_avail_vehicle - ax_drag

    return ax_final


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass