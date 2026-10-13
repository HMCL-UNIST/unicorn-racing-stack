#!/usr/bin/env python3
"""
IQP Handler Wrapper with iters_max support

Original: trajectory_planning_helpers.iqp_handler
This wrapper adds iters_max parameter to prevent infinite loops.
"""

import numpy as np
import trajectory_planning_helpers as tph


def iqp_handler(reftrack: np.ndarray,
                normvectors: np.ndarray,
                A: np.ndarray,
                kappa_bound: float,
                w_veh: float,
                print_debug: bool,
                plot_debug: bool,
                stepsize_interp: float,
                iters_min: int = 3,
                iters_max: int = None,
                curv_error_allowed: float = 0.01) -> tuple:
    """
    IQP handler with maximum iteration limit.

    This is a wrapper around the original iqp_handler that adds iters_max
    to prevent infinite optimization loops when convergence is not possible.

    All parameters are identical to tph.iqp_handler.iqp_handler except:
    :param iters_max: maximum number of iterations (default: None = no limit)

    Returns:
        alpha_mincurv_tmp: solution vector (lateral shift in m)
        reftrack_tmp: reference track data [x, y, w_tr_right, w_tr_left]
        normvectors_tmp: normalized normal vectors [x, y]
    """

    # Set initial data
    reftrack_tmp = reftrack
    normvectors_tmp = normvectors
    A_tmp = A

    # Loop
    iter_cur = 0

    while True:
        iter_cur += 1

        # Calculate intermediate solution and catch sum of squared curvature errors
        alpha_mincurv_tmp, curv_error_max_tmp = tph.opt_min_curv.opt_min_curv(
            reftrack=reftrack_tmp,
            normvectors=normvectors_tmp,
            A=A_tmp,
            kappa_bound=kappa_bound,
            w_veh=w_veh,
            print_debug=print_debug,
            plot_debug=plot_debug
        )

        # Print progress
        if print_debug:
            print("Minimum curvature IQP: iteration %i, curv_error_max: %.4frad/m" % (iter_cur, curv_error_max_tmp))

        # Restrict solution space to improve validity of the linearization during the first steps
        if iter_cur < iters_min:
            alpha_mincurv_tmp *= iter_cur * 1.0 / iters_min

        # Check termination criterion: minimum iterations and curvature error
        if iter_cur >= iters_min and curv_error_max_tmp <= curv_error_allowed:
            if print_debug:
                print("Finished IQP!")
            break

        # Check termination criterion: maximum iterations reached (only if iters_max is set)
        if iters_max is not None and iter_cur >= iters_max:
            if curv_error_max_tmp > curv_error_allowed:
                raise RuntimeError(
                    "IQP failed to converge: reached max iterations (%d) with curv_error %.4frad/m > allowed %.4frad/m"
                    % (iters_max, curv_error_max_tmp, curv_error_allowed)
                )
            else:
                if print_debug:
                    print("IQP reached max iterations (%d), but curv_error %.4frad/m is acceptable" % (iters_max, curv_error_max_tmp))
                break

        # ------------------------------------------------------------------------------
        # INTERPOLATION FOR EQUAL STEPSIZES
        # ------------------------------------------------------------------------------

        refline_tmp, _, _, _, spline_inds_tmp, t_values_tmp = tph.create_raceline.create_raceline(
            refline=reftrack_tmp[:, :2],
            normvectors=normvectors_tmp,
            alpha=alpha_mincurv_tmp,
            stepsize_interp=stepsize_interp
        )[:6]

        # Calculate new track boundaries on the basis of the intermediate alpha values
        reftrack_tmp[:, 2] -= alpha_mincurv_tmp
        reftrack_tmp[:, 3] += alpha_mincurv_tmp

        ws_track_tmp = tph.interp_track_widths.interp_track_widths(
            w_track=reftrack_tmp[:, 2:],
            spline_inds=spline_inds_tmp,
            t_values=t_values_tmp,
            incl_last_point=False
        )

        # Create new reftrack
        reftrack_tmp = np.column_stack((refline_tmp, ws_track_tmp))

        # ------------------------------------------------------------------------------
        # CALCULATE NEW SPLINES ON THE BASIS OF THE INTERPOLATED REFERENCE TRACK
        # ------------------------------------------------------------------------------

        # Calculate new splines
        refline_tmp_cl = np.vstack((reftrack_tmp[:, :2], reftrack_tmp[0, :2]))

        coeffs_x_tmp, coeffs_y_tmp, A_tmp, normvectors_tmp = tph.calc_splines.calc_splines(
            path=refline_tmp_cl,
            use_dist_scaling=False
        )

    return alpha_mincurv_tmp, reftrack_tmp, normvectors_tmp
