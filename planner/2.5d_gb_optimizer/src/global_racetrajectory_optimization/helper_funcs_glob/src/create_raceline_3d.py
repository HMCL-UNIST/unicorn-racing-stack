# 3D versions of tph functions — copied from trajectory_planning_helpers with minimal z additions
import numpy as np
import math
import trajectory_planning_helpers as tph


# ======================================================================================================================
# calc_spline_lengths_3d — original tph.calc_spline_lengths + coeffs_z
# ======================================================================================================================

def calc_spline_lengths_3d(coeffs_x: np.ndarray,
                           coeffs_y: np.ndarray,
                           coeffs_z: np.ndarray,  # added
                           quickndirty: bool = False,
                           no_interp_points: int = 15) -> np.ndarray:

    # check inputs
    if coeffs_x.shape[0] != coeffs_y.shape[0]:
        raise RuntimeError("Coefficient matrices must have the same length!")
    # check z
    if coeffs_x.shape[0] != coeffs_z.shape[0]:
        raise RuntimeError("coeffs_z must have the same length as coeffs_x/y!")

    # catch case with only one spline
    if coeffs_x.size == 4 and coeffs_x.shape[0] == 4:
        coeffs_x = np.expand_dims(coeffs_x, 0)
        coeffs_y = np.expand_dims(coeffs_y, 0)
        coeffs_z = np.expand_dims(coeffs_z, 0)

    # get number of splines and create output array
    no_splines = coeffs_x.shape[0]
    spline_lengths = np.zeros(no_splines)

    if quickndirty:
        for i in range(no_splines):
            spline_lengths[i] = math.sqrt(math.pow(np.sum(coeffs_x[i]) - coeffs_x[i, 0], 2)
                                          + math.pow(np.sum(coeffs_y[i]) - coeffs_y[i, 0], 2)
                                          + math.pow(np.sum(coeffs_z[i]) - coeffs_z[i, 0], 2))  # +dz²

    else:
        t_steps = np.linspace(0.0, 1.0, no_interp_points)
        spl_coords = np.zeros((no_interp_points, 3))  # 3 columns

        for i in range(no_splines):
            spl_coords[:, 0] = coeffs_x[i, 0] \
                               + coeffs_x[i, 1] * t_steps \
                               + coeffs_x[i, 2] * np.power(t_steps, 2) \
                               + coeffs_x[i, 3] * np.power(t_steps, 3)
            spl_coords[:, 1] = coeffs_y[i, 0] \
                               + coeffs_y[i, 1] * t_steps \
                               + coeffs_y[i, 2] * np.power(t_steps, 2) \
                               + coeffs_y[i, 3] * np.power(t_steps, 3)
            # z coordinate
            spl_coords[:, 2] = coeffs_z[i, 0] \
                               + coeffs_z[i, 1] * t_steps \
                               + coeffs_z[i, 2] * np.power(t_steps, 2) \
                               + coeffs_z[i, 3] * np.power(t_steps, 3)

            spline_lengths[i] = np.sum(np.sqrt(np.sum(np.power(np.diff(spl_coords, axis=0), 2), axis=1)))

    return spline_lengths


# ======================================================================================================================
# interp_splines_3d — original tph.interp_splines + coeffs_z
# ======================================================================================================================

def interp_splines_3d(coeffs_x: np.ndarray,
                      coeffs_y: np.ndarray,
                      coeffs_z: np.ndarray,  # added
                      spline_lengths: np.ndarray = None,
                      incl_last_point: bool = False,
                      stepsize_approx: float = None,
                      stepnum_fixed: list = None) -> tuple:

    # check sizes
    if coeffs_x.shape[0] != coeffs_y.shape[0]:
        raise RuntimeError("Coefficient matrices must have the same length!")
    # check z
    if coeffs_x.shape[0] != coeffs_z.shape[0]:
        raise RuntimeError("coeffs_z must have the same length as coeffs_x/y!")

    if spline_lengths is not None and coeffs_x.shape[0] != spline_lengths.size:
        raise RuntimeError("coeffs_x/y and spline_lengths must have the same length!")

    if not (coeffs_x.ndim == 2 and coeffs_y.ndim == 2):
        raise RuntimeError("Coefficient matrices do not have two dimensions!")

    if (stepsize_approx is None and stepnum_fixed is None) \
            or (stepsize_approx is not None and stepnum_fixed is not None):
        raise RuntimeError("Provide one of 'stepsize_approx' and 'stepnum_fixed' and set the other to 'None'!")

    if stepnum_fixed is not None and len(stepnum_fixed) != coeffs_x.shape[0]:
        raise RuntimeError("The provided list 'stepnum_fixed' must hold an entry for every spline!")

    if stepsize_approx is not None:
        if spline_lengths is None:
            spline_lengths = calc_spline_lengths_3d(coeffs_x=coeffs_x, coeffs_y=coeffs_y,
                                                    coeffs_z=coeffs_z, quickndirty=False)  # 3d

        dists_cum = np.cumsum(spline_lengths)
        no_interp_points = math.ceil(dists_cum[-1] / stepsize_approx) + 1
        dists_interp = np.linspace(0.0, dists_cum[-1], no_interp_points)

    else:
        no_interp_points = sum(stepnum_fixed) - (len(stepnum_fixed) - 1)
        dists_interp = None

    path_interp = np.zeros((no_interp_points, 3))  # 3 columns (x, y, z)
    spline_inds = np.zeros(no_interp_points, dtype=int)
    t_values = np.zeros(no_interp_points)

    if stepsize_approx is not None:

        for i in range(no_interp_points - 1):
            j = np.argmax(dists_interp[i] < dists_cum)
            spline_inds[i] = j

            if j > 0:
                t_values[i] = (dists_interp[i] - dists_cum[j - 1]) / spline_lengths[j]
            else:
                if spline_lengths.ndim == 0:
                    t_values[i] = dists_interp[i] / spline_lengths
                else:
                    t_values[i] = dists_interp[i] / spline_lengths[0]

            path_interp[i, 0] = coeffs_x[j, 0] \
                                + coeffs_x[j, 1] * t_values[i] \
                                + coeffs_x[j, 2] * math.pow(t_values[i], 2) \
                                + coeffs_x[j, 3] * math.pow(t_values[i], 3)

            path_interp[i, 1] = coeffs_y[j, 0] \
                                + coeffs_y[j, 1] * t_values[i] \
                                + coeffs_y[j, 2] * math.pow(t_values[i], 2) \
                                + coeffs_y[j, 3] * math.pow(t_values[i], 3)

            # z coordinate
            path_interp[i, 2] = coeffs_z[j, 0] \
                                + coeffs_z[j, 1] * t_values[i] \
                                + coeffs_z[j, 2] * math.pow(t_values[i], 2) \
                                + coeffs_z[j, 3] * math.pow(t_values[i], 3)

    else:
        j = 0

        for i in range(len(stepnum_fixed)):
            if i < len(stepnum_fixed) - 1:
                t_values[j:(j + stepnum_fixed[i] - 1)] = np.linspace(0, 1, stepnum_fixed[i])[:-1]
                spline_inds[j:(j + stepnum_fixed[i] - 1)] = i
                j += stepnum_fixed[i] - 1
            else:
                t_values[j:(j + stepnum_fixed[i])] = np.linspace(0, 1, stepnum_fixed[i])
                spline_inds[j:(j + stepnum_fixed[i])] = i
                j += stepnum_fixed[i]

        t_set = np.column_stack((np.ones(no_interp_points), t_values, np.power(t_values, 2), np.power(t_values, 3)))

        n_samples = np.array(stepnum_fixed)
        n_samples[:-1] -= 1

        path_interp[:, 0] = np.sum(np.multiply(np.repeat(coeffs_x, n_samples, axis=0), t_set), axis=1)
        path_interp[:, 1] = np.sum(np.multiply(np.repeat(coeffs_y, n_samples, axis=0), t_set), axis=1)
        # z
        path_interp[:, 2] = np.sum(np.multiply(np.repeat(coeffs_z, n_samples, axis=0), t_set), axis=1)

    if incl_last_point:
        path_interp[-1, 0] = np.sum(coeffs_x[-1])
        path_interp[-1, 1] = np.sum(coeffs_y[-1])
        path_interp[-1, 2] = np.sum(coeffs_z[-1])
        spline_inds[-1] = coeffs_x.shape[0] - 1
        t_values[-1] = 1.0
    else:
        path_interp = path_interp[:-1]
        spline_inds = spline_inds[:-1]
        t_values = t_values[:-1]
        if dists_interp is not None:
            dists_interp = dists_interp[:-1]

    return path_interp, spline_inds, t_values, dists_interp


# ======================================================================================================================
# create_raceline_3d — original tph.create_raceline + z coordinate
# ======================================================================================================================

def create_raceline_3d(refline: np.ndarray,
                       normvectors: np.ndarray,
                       alpha: np.ndarray,
                       z: np.ndarray,  # z at each refline point (M points, unclosed)
                       stepsize_interp: float) -> tuple:
    """
    3D version of tph.create_raceline
    Same as original but:
    - z is included in spline fitting and arc length calculation
    - returns [x, y, z] raceline and z-related outputs
    - no bank → raceline z = centerline z (alpha shift doesn't change z)

    Returns same as original + (z_interp, dz_ds) at the end.
    """

    # calculate raceline on the basis of the optimized alpha values
    raceline = refline + np.expand_dims(alpha, 1) * normvectors

    # calculate new splines on the basis of the raceline (xy — needed for heading/kappa)
    raceline_cl = np.vstack((raceline, raceline[0]))

    coeffs_x_raceline, coeffs_y_raceline, A_raceline, normvectors_raceline = tph.calc_splines.\
        calc_splines(path=raceline_cl,
                     use_dist_scaling=False)

    # z spline — fit cubic splines on z using same knot structure
    # no bank → raceline z = centerline z at same index
    z_cl = np.append(z, z[0])
    # use tph.calc_splines on a "fake path" [s, z] to get z spline coefficients
    # simpler approach: fit z as function of cumulative 2D arc length, then evaluate at same t values
    # but to keep it consistent with xy spline structure, we fit z directly as parametric spline
    no_splines = coeffs_x_raceline.shape[0]
    coeffs_z_raceline = np.zeros((no_splines, 4))

    # for each spline segment, fit z(t) = a0 + a1*t + a2*t² + a3*t³
    # boundary conditions: z(0) = z[i], z(1) = z[i+1], continuity of z' at knots
    # use same linear system approach as tph.calc_splines but for 1D
    # simplest: use scipy CubicSpline on index-parametrized z, then extract polynomial coefficients per segment
    from scipy.interpolate import CubicSpline as CubicSplineScipy
    t_knots = np.arange(len(z_cl), dtype=float)
    z_cs = CubicSplineScipy(t_knots, z_cl, bc_type='periodic')

    for i in range(no_splines):
        # CubicSpline gives polynomial coefficients in (t - t_knots[i])
        # P(dt) = c[3]*dt³ + c[2]*dt² + c[1]*dt + c[0]
        # but we need P(t) where t ∈ [0, 1] (normalized)
        # CubicSpline uses dt = t_global - t_knots[i], and our segment has dt ∈ [0, 1]
        # so coefficients directly map: a0=c[3], a1=c[2], a2=c[1], a3=c[0] (scipy order is [c3,c2,c1,c0] high→low)
        # scipy CubicSpline: c[i] has shape (4,) with c[0]=highest power
        # P(dt) = c[0]*dt³ + c[1]*dt² + c[2]*dt + c[3]
        # our convention: a0 + a1*t + a2*t² + a3*t³
        c = z_cs.c[:, i]  # [c3_coeff, c2_coeff, c1_coeff, c0_coeff] (high to low)
        coeffs_z_raceline[i, 0] = c[3]   # a0
        coeffs_z_raceline[i, 1] = c[2]   # a1
        coeffs_z_raceline[i, 2] = c[1]   # a2
        coeffs_z_raceline[i, 3] = c[0]   # a3

    # calculate 3D spline lengths
    spline_lengths_raceline = calc_spline_lengths_3d(coeffs_x=coeffs_x_raceline,
                                                     coeffs_y=coeffs_y_raceline,
                                                     coeffs_z=coeffs_z_raceline)

    # interpolate using 3D arc length
    raceline_interp_3d, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = \
        interp_splines_3d(spline_lengths=spline_lengths_raceline,
                          coeffs_x=coeffs_x_raceline,
                          coeffs_y=coeffs_y_raceline,
                          coeffs_z=coeffs_z_raceline,
                          incl_last_point=False,
                          stepsize_approx=stepsize_interp)

    # separate outputs
    raceline_interp = raceline_interp_3d[:, :2]  # [x, y] for compatibility with tph functions
    z_interp = raceline_interp_3d[:, 2]

    # calculate element lengths (3D)
    s_tot_raceline = float(np.sum(spline_lengths_raceline))
    el_lengths_raceline_interp = np.diff(s_raceline_interp)
    el_lengths_raceline_interp_cl = np.append(el_lengths_raceline_interp, s_tot_raceline - s_raceline_interp[-1])

    # compute dz/ds analytically from z spline coefficients
    # dz/dt = a1 + 2*a2*t + 3*a3*t²
    # ds/dt ≈ spline_length_of_segment (since t ∈ [0,1] maps to one segment)
    # dz/ds = (dz/dt) / (ds/dt)
    K = len(t_values_raceline_interp)
    dz_ds = np.zeros(K)
    for i in range(K):
        j = spline_inds_raceline_interp[i]
        t = t_values_raceline_interp[i]
        dz_dt = coeffs_z_raceline[j, 1] + 2 * coeffs_z_raceline[j, 2] * t + 3 * coeffs_z_raceline[j, 3] * t * t

        # ds/dt from xy+z: approximate as spline_length / 1.0 (since t goes 0→1)
        # more accurate: ds/dt = sqrt((dx/dt)² + (dy/dt)² + (dz/dt)²)
        dx_dt = coeffs_x_raceline[j, 1] + 2 * coeffs_x_raceline[j, 2] * t + 3 * coeffs_x_raceline[j, 3] * t * t
        dy_dt = coeffs_y_raceline[j, 1] + 2 * coeffs_y_raceline[j, 2] * t + 3 * coeffs_y_raceline[j, 3] * t * t
        ds_dt = math.sqrt(dx_dt * dx_dt + dy_dt * dy_dt + dz_dt * dz_dt)

        if ds_dt > 1e-10:
            dz_ds[i] = dz_dt / ds_dt
        else:
            dz_ds[i] = 0.0

    return raceline_interp, A_raceline, coeffs_x_raceline, coeffs_y_raceline, spline_inds_raceline_interp, \
           t_values_raceline_interp, s_raceline_interp, spline_lengths_raceline, el_lengths_raceline_interp_cl, \
           z_interp, dz_ds
