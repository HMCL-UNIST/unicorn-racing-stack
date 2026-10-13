# lightweight NLP velocity optimizer for fast_sqp_planner
#  Two-stage approach:
#    1) Pre-compute per-knot ax/ay limits via g_tilde iteration (numpy, fast)
#    2) Solve small parametrized NLP with simple box constraints (CasADi+IPOPT)
#  Solver is built once per N and cached.
from __future__ import annotations

import time
import numpy as np
import casadi as ca

try:
    import rospy
    _log_info = rospy.loginfo
    _log_warn = rospy.logwarn
except ImportError:
    _log_info = print
    _log_warn = print

# detect HSL linear solver availability
try:
    _test_opts = {'ipopt.linear_solver': 'ma27', 'ipopt.print_level': 0, 'print_time': 0}
    _x = ca.MX.sym('x')
    _probe = ca.nlpsol('_test', 'ipopt', {'x': _x, 'f': (_x - 1.0)**2}, _test_opts)
    # creating the solver succeeds without libhsl; only a solve tells
    _probe(x0=0.0)
    _LINEAR_SOLVER = 'ma27' if _probe.stats().get('success', False) else 'mumps'
except Exception:
    _LINEAR_SOLVER = 'mumps'


class VelocityNLP:
    """Lightweight parametrized NLP. Build once, solve many.

    Pre-computed per-knot limits (ax_min, ax_max, ay_max) are passed as
    parameters — no CasADi interpolant inside the NLP graph.
    """

    def __init__(self, N: int, V_min: float = 0.5,
                 w_T: float = 1.0, w_jx: float = 1e-2,
                 max_iter: int = 30, print_level: int = 0):
        self.N = N
        self.V_min = V_min
        t0 = time.time()
        self._build(N, V_min, w_T, w_jx, max_iter, print_level)
        _log_info(f'[nlp_vel] solver built: N={N}, {(time.time()-t0)*1000:.0f}ms')

    def _build(self, N, V_min, w_T, w_jx, max_iter, print_level):
        # ---- parameters (change each tick) ----
        p_kappa   = ca.MX.sym('kappa', N)       # curvature
        p_ds      = ca.MX.sym('ds', N - 1)      # segment lengths
        p_ax_max  = ca.MX.sym('ax_max', N)       # pre-computed ax upper
        p_ax_min  = ca.MX.sym('ax_min', N)       # pre-computed ax lower (negative)
        p_ay_max  = ca.MX.sym('ay_max', N)       # pre-computed ay limit
        p_vstart  = ca.MX.sym('vstart')
        p_vmax    = ca.MX.sym('vmax')
        P = ca.vertcat(p_kappa, p_ds, p_ax_max, p_ax_min, p_ay_max,
                        p_vstart, p_vmax)
        self._n_params = P.shape[0]

        # ---- decision variables ----
        w, lbw, ubw = [], [], []
        g_con, lbg, ubg = [], [], []
        J = 0.0

        Vk = ca.MX.sym('V_0')
        axk = ca.MX.sym('ax_0')
        w += [Vk, axk]
        lbw += [V_min, -30.0]
        ubw += [1e6, 30.0]

        # fix v_start
        g_con += [Vk - p_vstart]
        lbg += [0.0]; ubg += [0.0]

        for k in range(N):
            ay_k = Vk ** 2 * p_kappa[k]

            # ay feasibility: |ay| <= ay_max
            g_con += [p_ay_max[k] - ca.fabs(ay_k)]
            lbg += [0.0]; ubg += [np.inf]

            # ax bounds: ax_min <= ax <= ax_max
            g_con += [axk - p_ax_min[k]]   # ax >= ax_min
            lbg += [0.0]; ubg += [np.inf]
            g_con += [p_ax_max[k] - axk]   # ax <= ax_max
            lbg += [0.0]; ubg += [np.inf]

            # V <= vmax
            g_con += [p_vmax - Vk]
            lbg += [0.0]; ubg += [np.inf]

            if k == N - 1:
                break

            # control: jerk
            jxk = ca.MX.sym(f'jx_{k}')
            w += [jxk]
            lbw += [-200.0]
            ubw += [200.0]

            # Euler integration
            s_dot = ca.fmax(Vk, 0.1)
            Vk1 = Vk + (axk / s_dot) * p_ds[k]
            axk1 = axk + (jxk / s_dot) * p_ds[k]
            J += (w_T / s_dot + w_jx * (jxk / s_dot) ** 2) * p_ds[k]

            # next state
            Vk = ca.MX.sym(f'V_{k+1}')
            axk = ca.MX.sym(f'ax_{k+1}')
            w += [Vk, axk]
            lbw += [V_min, -30.0]
            ubw += [1e6, 30.0]

            # dynamics continuity
            g_con += [Vk1 - Vk, axk1 - axk]
            lbg += [0.0, 0.0]
            ubg += [0.0, 0.0]

        w_vec = ca.vertcat(*w)
        g_vec = ca.vertcat(*g_con)

        sol_opt = {
            'ipopt.max_iter': max_iter,
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 5e-2,
            'ipopt.acceptable_iter': 3,
            'ipopt.constr_viol_tol': 1e-3,
            'ipopt.linear_solver': _LINEAR_SOLVER,
            'ipopt.print_level': print_level,
            'print_time': 0,
            'ipopt.warm_start_init_point': 'yes',
        }

        nlp = {'f': J, 'x': w_vec, 'g': g_vec, 'p': P}
        self._solver = ca.nlpsol('vel_nlp', 'ipopt', nlp, sol_opt)
        self._lbw = np.array(lbw, dtype=float)
        self._ubw = np.array(ubw, dtype=float)
        self._lbg = np.array(lbg, dtype=float)
        self._ubg = np.array(ubg, dtype=float)
        self._n_w = w_vec.shape[0]
        self._last_sol = None
        self._last_lam_g = None
        self._last_lam_x = None

    def solve(self, kappa, el_lengths, ax_max, ax_min, ay_max,
              v_start, v_max, v_init=None):
        N = self.N
        V_min = self.V_min

        p_val = np.concatenate([
            kappa, el_lengths, ax_max, ax_min, ay_max,
            [max(v_start, V_min)], [v_max],
        ])

        # warm-start
        if self._last_sol is not None:
            x0 = self._last_sol
        elif v_init is not None:
            x0 = np.zeros(self._n_w)
            stride = 3
            for k in range(N - 1):
                x0[k * stride] = max(float(v_init[k]), V_min)
            x0[(N - 1) * stride] = max(float(v_init[-1]), V_min)
        else:
            x0 = np.full(self._n_w, max(v_start, V_min))

        kwargs = dict(x0=x0, p=p_val,
                      lbx=self._lbw, ubx=self._ubw,
                      lbg=self._lbg, ubg=self._ubg)
        if self._last_lam_g is not None:
            kwargs['lam_g0'] = self._last_lam_g
            kwargs['lam_x0'] = self._last_lam_x

        t0 = time.time()
        sol = self._solver(**kwargs)
        solve_ms = (time.time() - t0) * 1000.0

        success = self._solver.stats()['success']
        sol_x = np.array(sol['x']).flatten()
        self._last_sol = sol_x
        self._last_lam_g = np.array(sol['lam_g']).flatten()
        self._last_lam_x = np.array(sol['lam_x']).flatten()

        # extract V, ax
        V_opt = np.zeros(N)
        ax_opt = np.zeros(N)
        stride = 3
        for k in range(N - 1):
            off = k * stride
            V_opt[k] = sol_x[off]
            ax_opt[k] = sol_x[off + 1]
        off = (N - 1) * stride
        V_opt[-1] = sol_x[off]
        ax_opt[-1] = sol_x[off + 1]

        _log_info(f'[nlp_vel] solve={solve_ms:.1f}ms ok={success} '
                  f'V=[{V_opt.min():.2f},{V_opt.max():.2f}]')

        return V_opt, ax_opt, success


# module-level cache
_cached_nlp: VelocityNLP | None = None


def solve_velocity_nlp(
    kappa: np.ndarray,
    el_lengths: np.ndarray,
    mu: np.ndarray,
    gg,
    v_start: float,
    v_max: float,
    v_init: np.ndarray | None = None,
    V_min: float = 0.5,
    w_T: float = 1.0,
    w_jx: float = 1e-2,
    max_iter: int = 30,
    print_level: int = 0,
    ggv_lookup_fn=None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Two-stage velocity NLP: pre-compute limits, then solve.

    ggv_lookup_fn: callable(v_arr, mu_arr, g_values=None) -> (ax_max, ay_max)
        If provided, used for g_tilde iteration. Otherwise uses gg directly.
    """
    global _cached_nlp
    N = len(kappa)

    # ---- stage 1: pre-compute per-knot limits via g_tilde iteration ----
    ds_mean = float(el_lengths.mean())
    mu_ext = np.concatenate([[mu[0]], mu, [mu[-1]]])
    dmu_ds = (mu_ext[2:] - mu_ext[:-2]) / (2.0 * ds_mean)

    if ggv_lookup_fn is not None:
        # use the node's _ggv_lookup with g_tilde iteration
        v_iter = v_init.copy() if v_init is not None else np.full(N, max(v_start, 1.0))
        ggv_data_g_list = None
        for _ in range(5):
            g_tilde = 9.81 * np.cos(mu) - v_iter ** 2 * dmu_ds
            ax_max_arr, ay_max_arr = ggv_lookup_fn(v_iter, mu, g_values=g_tilde)
            # simple fwbw estimate for next iteration
            radii = np.where(np.abs(kappa) > 1e-5, 1.0 / np.abs(kappa), 1e4)
            v_lat = np.sqrt(np.clip(ay_max_arr * radii, 0, v_max ** 2))
            v_new = np.minimum(v_lat, v_max)
            if np.max(np.abs(v_new - v_iter)) < 0.05:
                break
            v_iter = v_new
        ax_min_arr = -ax_max_arr  # symmetric braking approx
    else:
        # direct GGManager numpy lookup (fallback)
        from scipy.interpolate import RegularGridInterpolator
        v_arr = v_init if v_init is not None else np.full(N, max(v_start, 1.0))
        g_eff = 9.81 * np.cos(mu)
        v_c = np.clip(v_arr, 0, float(gg.V_max))
        g_c = np.clip(g_eff, float(gg.g_list[1]), float(gg.g_max))
        # use raw npy data from gg
        pts = np.column_stack([v_c, g_c])
        ax_max_arr = RegularGridInterpolator(
            (gg.V_list, gg.g_list), gg.ax_max_list,
            bounds_error=False, fill_value=None)(pts)
        ay_max_arr = RegularGridInterpolator(
            (gg.V_list, gg.g_list), gg.ay_max_list,
            bounds_error=False, fill_value=None)(pts)
        ax_min_arr = RegularGridInterpolator(
            (gg.V_list, gg.g_list), gg.ax_min_list,
            bounds_error=False, fill_value=None)(pts)

    # clamp to sane values
    ax_max_arr = np.clip(ax_max_arr, 0.1, 30.0)
    ax_min_arr = np.clip(ax_min_arr, -30.0, -0.1)
    ay_max_arr = np.clip(ay_max_arr, 0.1, 30.0)

    # ---- stage 2: solve cached NLP ----
    if _cached_nlp is None or _cached_nlp.N != N:
        _cached_nlp = VelocityNLP(
            N=N, V_min=V_min, w_T=w_T, w_jx=w_jx,
            max_iter=max_iter, print_level=print_level)

    return _cached_nlp.solve(
        kappa, el_lengths, ax_max_arr, ax_min_arr, ay_max_arr,
        v_start, v_max, v_init)
