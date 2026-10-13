from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import casadi as ca
import numpy as np


#probe HSL ma27; fall back to MUMPS if libhsl.so not loadable.
def _select_linear_solver(verbose: bool = True) -> str:
    try:
        _x = ca.MX.sym('x')
        _probe = ca.nlpsol('hsl_probe', 'ipopt',
                           {'x': _x, 'f': (_x - 1.0) ** 2},
                           {'ipopt.linear_solver': 'ma27',
                            'ipopt.print_level': 0, 'print_time': 0})
        _probe(x0=0.0)
        if _probe.stats().get('success', False):
            if verbose:
                print('[sqp_casadi] linear_solver: ma27 (HSL)')
            return 'ma27'
    except Exception:
        pass
    if verbose:
        print('[sqp_casadi] linear_solver: mumps (HSL not available, fallback)')
    return 'mumps'


LINEAR_SOLVER = _select_linear_solver()


@dataclass
class SQPProblem:
    n_knots: int
    delta_s: float
    d_init: np.ndarray           # warm-start guess, shape (n_knots,)
    current_d: float             # ego lateral offset at start
    bounds_lower: np.ndarray     # right boundary (d_min), shape (n_knots,)
    bounds_upper: np.ndarray     # left boundary  (d_max), shape (n_knots,)
    obs_center_d: np.ndarray     # (n_knots,)
    obs_min_dist: np.ndarray     # (n_knots,)
    desired_side: str            # "left" | "right" | "any"
    kappa_limit: float           # 1/R_min given current speed
    lambda_reg: float            # regularization weight
    lambda_smooth: float = 100.0
    lambda_start_heading: float = 1000.0
    lambda_apex_bias: float = 10.0
    lambda_side: float = 50.0
    lambda_jerk: float = 0.0
    lambda_term: float = 0.0
    #all-soft — obstacle, curvature, GG penalties
    lambda_obs: float = 10000.0
    lambda_kappa: float = 5000.0
    lambda_gg: float = 5000.0
    lambda_near_reg: float = 0.0
    near_knots_K: int = 0
    #obstacle-vicinity lateral rate suppression
    lambda_obs_smooth: float = 0.0
    obs_ramp_knots: int = 5

    #joint velocity optimization (optional)
    optimize_velocity: bool = False

    v_init: Optional[np.ndarray] = None
    v_current: float = 1.0
    v_max_arr: Optional[np.ndarray] = None
    kappa_ref: Optional[np.ndarray] = None
    ax_max_arr: Optional[np.ndarray] = None
    ay_max_arr: Optional[np.ndarray] = None
    lambda_progress: float = 1.0


class CasadiSQPSolver:

    def __init__(self):
        self._solver = None
        self._n_knots = None
        self._near_k = 0
        self._obs_ramp = 0
        self._opt_vel = False

    def _build(self, n_knots: int, near_k: int = 0,
               obs_ramp: int = 5, opt_vel: bool = False) -> None:
        d = ca.SX.sym('d', n_knots)

        p_d_init = ca.SX.sym('d_init', n_knots)
        p_cur_d = ca.SX.sym('cur_d')
        p_obs_c = ca.SX.sym('obs_c', n_knots)
        p_obs_m = ca.SX.sym('obs_m', n_knots)
        p_kappa_lim = ca.SX.sym('kappa_lim')
        p_lam_reg = ca.SX.sym('lam_reg')
        p_lam_smooth = ca.SX.sym('lam_smooth')
        p_lam_start = ca.SX.sym('lam_start')
        p_lam_apex = ca.SX.sym('lam_apex')
        p_lam_side = ca.SX.sym('lam_side')
        p_lam_jerk = ca.SX.sym('lam_jerk')
        p_lam_term = ca.SX.sym('lam_term')
        p_lam_obs = ca.SX.sym('lam_obs')
        p_lam_kappa = ca.SX.sym('lam_kappa')
        p_lam_gg = ca.SX.sym('lam_gg')
        p_lam_near_reg = ca.SX.sym('lam_near_reg')
        p_lam_obs_smooth = ca.SX.sym('lam_obs_smooth')
        p_delta_s = ca.SX.sym('ds')
        p_side = ca.SX.sym('side')   # +1 left, -1 right, 0 any

        # ---- cost ----
        dd = d[2:] - 2 * d[1:-1] + d[:-2]
        ddd = d[3:] - 3 * d[2:-1] + 3 * d[1:-2] - d[:-3]
        J = (p_lam_smooth * ca.sumsqr(dd)
             + p_lam_jerk * ca.sumsqr(ddd)
             + p_lam_start * (d[1] - d[0]) ** 2
             + p_lam_apex * ca.sumsqr(d)
             + p_lam_reg * ca.sumsqr(d - p_d_init)
             + p_lam_side * ca.sumsqr(ca.fmax(-p_side * d, 0.0)))

        #soft terminal — always available
        J = J + p_lam_term * d[-1] ** 2

        #soft obstacle penalty — ramp scale: 0→1 over obs_ramp knots
        for k in range(n_knots):
            violation = p_obs_m[k] ** 2 - (d[k] - p_obs_c[k]) ** 2
            obs_scale = min(k / max(obs_ramp, 1), 1.0) if obs_ramp > 0 else 1.0
            J = J + p_lam_obs * obs_scale * ca.fmax(violation, 0.0)

        #soft curvature penalty — max(κ² - κ_lim², 0)
        for i in range(1, n_knots - 1):
            kappa = (d[i + 1] - 2 * d[i] + d[i - 1]) / (p_delta_s ** 2)
            J = J + p_lam_kappa * ca.fmax(kappa ** 2 - p_kappa_lim ** 2, 0.0)

        #obstacle-vicinity lateral rate penalty — suppress d' near obstacle
        for k in range(n_knots - 1):
            rate = (d[k + 1] - d[k]) / p_delta_s
            J = J + p_lam_obs_smooth * p_obs_m[k] * rate ** 2

        #near-term extra regularization
        if near_k > 0:
            k_clip = min(int(near_k), n_knots)
            J = J + p_lam_near_reg * ca.sumsqr(d[:k_clip] - p_d_init[:k_clip])

        #velocity decision variables (optional)
        if opt_vel:
            v = ca.SX.sym('v', n_knots)
            p_v_init = ca.SX.sym('v_init', n_knots)
            p_cur_v = ca.SX.sym('cur_v')
            p_v_max = ca.SX.sym('v_max', n_knots)
            p_kappa_ref = ca.SX.sym('kappa_ref', n_knots)
            p_ax_max = ca.SX.sym('ax_max', n_knots)
            p_ay_max = ca.SX.sym('ay_max', n_knots)
            p_lam_prog = ca.SX.sym('lam_prog')
            J = J - p_lam_prog * ca.sum1(v) + 0.1 * p_lam_reg * ca.sumsqr(v - p_v_init)

            #soft GG penalty
            for i in range(1, n_knots):
                ax = (v[i] ** 2 - v[i - 1] ** 2) / (2.0 * p_delta_s)
                if 1 <= i <= n_knots - 2:
                    kappa_d = (d[i + 1] - 2 * d[i] + d[i - 1]) / (p_delta_s ** 2)
                else:
                    kappa_d = 0.0
                kappa_tot = p_kappa_ref[i] + kappa_d
                ay = v[i] ** 2 * kappa_tot
                gg_violation = (ax / p_ax_max[i]) ** 2 + (ay / p_ay_max[i]) ** 2 - 1.0
                J = J + p_lam_gg * ca.fmax(gg_violation, 0.0)

            x_all = ca.vertcat(d, v)
        else:
            x_all = d

        # ---- constraints (only hard equalities) ----
        g_list = [d[0] - p_cur_d]
        if opt_vel:
            v = x_all[n_knots:]
            g_list.append(v[0] - p_cur_v)

        g = ca.vertcat(*g_list)

        p_list = [p_d_init, p_cur_d, p_obs_c, p_obs_m,
                  p_kappa_lim, p_lam_reg, p_lam_smooth,
                  p_lam_start, p_lam_apex, p_lam_side,
                  p_lam_jerk, p_lam_term, p_lam_obs,
                  p_lam_kappa, p_lam_gg,
                  p_lam_near_reg, p_lam_obs_smooth, p_delta_s, p_side]
        if opt_vel:
            p_list.extend([p_v_init, p_cur_v, p_v_max, p_kappa_ref,
                           p_ax_max, p_ay_max, p_lam_prog])
        p = ca.vertcat(*p_list)

        nlp = {'x': x_all, 'p': p, 'f': J, 'g': g}
        opts = {
            'print_time': 0,
            'ipopt.print_level': 0,
            'ipopt.max_iter': 50,
            'ipopt.linear_solver': LINEAR_SOLVER,
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.acceptable_iter': 5,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.warm_start_slack_bound_push': 1e-6,
            'ipopt.mu_init': 1e-4,
            'ipopt.hessian_approximation': 'limited-memory',
            #JIT — compile symbolic NLP to native C for faster eval
            'jit': True,
            'compiler': 'shell',
            'jit_options': {'flags': ['-O3', '-march=native', '-ffast-math'],
                            'verbose': False},
        }
        self._solver = ca.nlpsol('rhp_sqp', 'ipopt', nlp, opts)
        self._n_knots = n_knots
        self._near_k = int(near_k)
        self._obs_ramp = int(obs_ramp)
        self._opt_vel = bool(opt_vel)

    def solve(self, prob: SQPProblem) -> tuple[np.ndarray, dict]:
        n = prob.n_knots
        ov = bool(prob.optimize_velocity)

        obs_ramp = int(prob.obs_ramp_knots)
        need_rebuild = (self._solver is None
                        or self._n_knots != n
                        or self._near_k != int(prob.near_knots_K)
                        or self._obs_ramp != obs_ramp
                        or self._opt_vel != ov)
        if need_rebuild:
            self._build(n, near_k=int(prob.near_knots_K), obs_ramp=obs_ramp, opt_vel=ov)

        side_flag = {'left': 1.0, 'right': -1.0}.get(prob.desired_side, 0.0)

        p_val = np.concatenate([
            prob.d_init,
            np.array([prob.current_d]),
            prob.obs_center_d,
            prob.obs_min_dist,
            np.array([prob.kappa_limit]),
            np.array([prob.lambda_reg]),
            np.array([prob.lambda_smooth]),
            np.array([prob.lambda_start_heading]),
            np.array([prob.lambda_apex_bias]),
            np.array([prob.lambda_side]),
            np.array([prob.lambda_jerk]),
            np.array([prob.lambda_term]),
            np.array([prob.lambda_obs]),
            np.array([prob.lambda_kappa]),
            np.array([prob.lambda_gg]),
            np.array([prob.lambda_near_reg]),
            np.array([prob.lambda_obs_smooth]),
            np.array([prob.delta_s]),
            np.array([side_flag]),
        ])
        if ov:
            p_val = np.concatenate([
                p_val,
                prob.v_init if prob.v_init is not None else np.full(n, prob.v_current),
                np.array([prob.v_current]),
                prob.v_max_arr if prob.v_max_arr is not None else np.full(n, 10.0),
                prob.kappa_ref if prob.kappa_ref is not None else np.zeros(n),
                prob.ax_max_arr if prob.ax_max_arr is not None else np.full(n, 5.0),
                prob.ay_max_arr if prob.ay_max_arr is not None else np.full(n, 4.5),
                np.array([prob.lambda_progress]),
            ])

        n_eq = 1
        if ov:
            n_eq += 1  # v[0] = current_v
        n_g = n_eq
        lbg = np.full(n_g, -1e-2)
        ubg = np.full(n_g, 1e-2)

        # decision variable bounds & init
        if ov:
            v_lo = np.full(n, 0.5)
            v_hi = prob.v_max_arr if prob.v_max_arr is not None else np.full(n, 10.0)
            lbx = np.concatenate([prob.bounds_lower, v_lo])
            ubx = np.concatenate([prob.bounds_upper, v_hi])
            v_init = prob.v_init if prob.v_init is not None else np.full(n, prob.v_current)
            x0_default = np.concatenate([prob.d_init, v_init])
        else:
            lbx = prob.bounds_lower
            ubx = prob.bounds_upper
            x0_default = prob.d_init

        #always use shifted d_init as IPOPT start point.
        #  _last_x is position-mismatched (not shifted) and causes IPOPT
        #  to start from wrong basin → different local minima each tick.
        x0 = x0_default

        args = dict(x0=x0, p=p_val, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        sol = self._solver(**args)
        stats = self._solver.stats()
        ipopt_ok = bool(stats.get('success', False))
        x_opt = np.array(sol['x']).flatten()

        d_opt = x_opt[:n]
        v_opt = x_opt[n:] if ov else None

        info = {
            'success': True,
            'ipopt_success': ipopt_ok,
            'status': str(stats.get('return_status', '')),
            'iter_count': int(stats.get('iter_count', 0)),
            'cost': float(sol['f']),
            'v_opt': v_opt,
        }

        return d_opt, info
