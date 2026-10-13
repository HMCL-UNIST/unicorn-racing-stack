#!/usr/bin/env python3
# Fast GGV generator — parametric NLP (solver built once, V/g/alpha as parameters)
# Based on gg_diagram_generation/gen_gg_diagrams.py, restructured for speed.
# Key change: nlpsol() called once, not 16,875 times.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import yaml
import time
from casadi import *
from calc_max_slip_map import calc_max_slip_map
import multiprocessing
from joblib import Parallel, delayed


# Probe HSL ma27; fall back to MUMPS if libhsl.so is not loadable.
#       ma27 is 2~3x faster for our small NLPs but requires a user-built libhsl.
def _select_linear_solver(verbose=True):
    try:
        _x = MX.sym('x')
        _probe = nlpsol('hsl_probe', 'ipopt',
                        {'x': _x, 'f': (_x - 1.0) ** 2},
                        {'ipopt.linear_solver': 'ma27',
                         'ipopt.print_level': 0, 'print_time': 0})
        _probe(x0=0.0)
        if _probe.stats().get('success', False):
            if verbose:
                print('[fast_gg] linear_solver: ma27 (HSL)')
            return 'ma27'
    except Exception:
        pass
    if verbose:
        print('[fast_gg] linear_solver: mumps (HSL not available, fallback)')
    return 'mumps'

LINEAR_SOLVER = _select_linear_solver()

# parse arguments
parser = argparse.ArgumentParser(description='Fast GGV diagram generator (parametric NLP)')
parser.add_argument('--vehicle_name', type=str, default='rc_car_10th', help='Vehicle name')
parser.add_argument('--tuning', action='store_true', default=False,
                    help='Apply tuning_<vehicle>.yml override')
# decouple tuning file name from vehicle_name
#       Default: tuning_<vehicle_name>.yml (backward compatible)
#       Override: e.g. --tuning_name rc_car_10th
#                 → use tuning_rc_car_10th.yml regardless of --vehicle_name
#       Enables base × tuning combinations (multiple cars, shared tuning).
parser.add_argument('--tuning_name', type=str, default=None,
                    help='Tuning file suffix (default: same as --vehicle_name). '
                         'Allows decoupled base/tuning combinations.')
parser.add_argument('--fast', action='store_true', default=True,
                    help='Use reduced resolution for fast tuning (default: True)')
parser.add_argument('--full', action='store_true', default=False,
                    help='Use full resolution (override --fast)')
args, _ = parser.parse_known_args()

vehicle_name = args.vehicle_name
fast_mode = args.fast and not args.full

# ============================================================
# Resolution settings
# ============================================================
g_earth = 9.81

# raise V_min from 1.5 to 2.0 to avoid Pacejka low-speed singularity
#       Reason: at V<2, the slip angle regularization (eps_v=0.5) still intrudes
#       ~17%, producing an artificially shrunken envelope. The post-hoc floor
#       clamp worked around this but introduced a fit anomaly at V=1.5, g=10.5
#       (fast4 output showed ax_max=1.03 vs raw rho=5.98 — diamond NLP stuck
#       in a local minimum on the mixed-origin clamped envelope).
#       Setting V_min=2.0 means:
#         - NLP never computes the problematic V=1.5 slice
#         - Floor clamp effectively no-ops (no drop to trigger it)
#         - Diamond fit sees clean physics across the entire V range
#       Lookup coverage: planner may occasionally request V<2 (observed min
#       ≈1.64 on eng_0410_v5 track). The C++ FBGA binary clamps out-of-range
#       V lookups to the nearest v_list entry, so V=1.0~1.99 queries return
#       the V=2.0 envelope — equivalent to a hard floor at V=2, but much
#       cleaner than the original clamp hack.
# g range [1.81, 21.81] = step 2.0, g_N 6/11 → flat 9.81 on grid,
# covers weight=1.5 amplify at v_max=9 valley (19.68) clip-free.
# fast=full[::2] subset preserved.
# V uses fixed step (fast=1.0, full=0.5) → prefix-compatible across v_max.
if fast_mode:
    V_min = 2.0
    V_step = 1.0
    g_factor_min = 1.81 / g_earth
    g_factor_max = 21.81 / g_earth
    g_N = 6
    alpha_N_nlp = 60
    alpha_N_interp = 125
    print(f'[fast_gg] FAST mode: V_step={V_step} g_N={g_N} alpha_N={alpha_N_nlp}')
else:
    V_min = 2.0
    V_step = 0.5
    g_factor_min = 1.81 / g_earth
    g_factor_max = 21.81 / g_earth
    g_N = 11
    alpha_N_nlp = 125
    alpha_N_interp = 125
    print(f'[fast_gg] FULL mode: V_step={V_step} g_N={g_N} alpha_N={alpha_N_nlp}')

g_list = np.round(np.linspace(g_earth * g_factor_min, g_earth * g_factor_max, g_N), 6)
alpha_list = np.linspace(-0.5 * np.pi, 0.5 * np.pi, alpha_N_nlp)
alpha_list_interp = np.linspace(-np.pi, np.pi, alpha_N_interp)

# ============================================================
# Paths & parameters
# ============================================================
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'global_line', 'data')
vehicle_params_path = os.path.join(data_path, 'vehicle_params', 'params_' + vehicle_name + '.yml')
# output to fast_ggv_gen/output/ to NEVER overwrite original gg_diagrams
out_path = os.path.join(dir_path, 'output', vehicle_name)

num_cores = multiprocessing.cpu_count()

with open(vehicle_params_path, 'r') as stream:
    params = yaml.safe_load(stream)
vehicle_params = params['vehicle_params']
tire_params = params['tire_params']

if args.tuning:
    # tuning_name defaults to vehicle_name but can be overridden
    #       to allow base/tuning decoupling (e.g. different base, shared tuning).
    # tuning_path = os.path.join(data_path, 'vehicle_params', 'tuning_' + vehicle_name + '.yml')
    _tuning_name = args.tuning_name if args.tuning_name else vehicle_name
    tuning_path = os.path.join(data_path, 'vehicle_params', 'tuning_' + _tuning_name + '.yml')
    if os.path.exists(tuning_path):
        with open(tuning_path, 'r') as stream:
            tuning = yaml.safe_load(stream)
        if tuning:
            # include p_Dx_1/p_Dy_1 so rqt friction flows into NLP
            for key in ['lambda_mu_x', 'lambda_mu_y',
                        'p_Dx_1', 'p_Dy_1', 'p_Dx_2', 'p_Dy_2']:
                if key in tuning:
                    tire_params[key] = tuning[key]
            # extend tuning override keys to include NLP constraint params
            #       (was: only P_max, v_max, epsilon)
            # for key in ['P_max', 'v_max', 'epsilon']:
            #     if key in tuning:
            #         vehicle_params[key] = tuning[key]
            for key in ['P_max', 'v_max', 'epsilon',
                        'P_brake_max', 'ax_max_cap', 'ax_min_cap', 'ay_max_cap']:
                if key in tuning:
                    vehicle_params[key] = tuning[key]
            print(f'[fast_gg] Tuning override applied from {tuning_path}')

            # save merged (base + tuning override) params to output for traceability
            #       So each output/<vehicle>/ folder is self-contained and shows
            #       exactly which parameter set produced the rho.npy files.
            os.makedirs(out_path, exist_ok=True)
            merged_path = os.path.join(out_path, 'params_' + vehicle_name + '.yml')
            header = (
                "# Auto-generated merged params (base + tuning override)\n"
                f"# base:   {os.path.basename(vehicle_params_path)}\n"
                f"# tuning: {os.path.basename(tuning_path)}\n"
                f"# at:     {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                "# DO NOT EDIT — regenerated on every fast_gen_gg_diagrams run\n\n"
            )
            merged = {'vehicle_params': vehicle_params, 'tire_params': tire_params}
            with open(merged_path, 'w') as f:
                f.write(header)
                yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)
            print(f'[fast_gg] Merged params saved to {merged_path}')
    else:
        print(f'[fast_gg] WARNING: --tuning specified but {tuning_path} not found')

# V_max = v_max exactly — no overshoot so downstream (raceline NLP, FBGA) respects v_max.
# V_list built with fixed V_step → grows with v_max, prefix-compatible across vehicles.
_v_max_cfg = float(vehicle_params['v_max'])
V_max = _v_max_cfg
if V_max <= V_min:
    raise ValueError(f'[fast_gg] v_max ({_v_max_cfg}) must be > V_min ({V_min})')
V_list = np.arange(V_min, V_max + V_step / 2.0, V_step)
V_N = len(V_list)
print(f'[fast_gg] v_max={_v_max_cfg} → V_list={V_list.tolist()} (V_N={V_N})')

# calculate maximum slip maps
N_list, kappa_max_list, lambda_max_list = calc_max_slip_map(tire_params=tire_params)
kappa_max = interpolant("kappa_max", "bspline", [N_list], np.abs(kappa_max_list))
lambda_max = interpolant("lambda_max", "bspline", [N_list], np.abs(lambda_max_list))

# ============================================================
# Build parametric NLP (ONCE)
# ============================================================
print('[fast_gg] Building parametric NLP solver (one-time cost)...')
t_build_start = time.time()

# --- Parameters: V, g_force, alpha ---
p_V = MX.sym("p_V")
p_g = MX.sym("p_g")
p_alpha = MX.sym("p_alpha")
p = vertcat(p_V, p_g, p_alpha)

# --- Decision variables with scaling ---
mu_x_max = tire_params["p_Dx_1"] * tire_params["lambda_mu_x"]
a_max = mu_x_max * g_earth

a_x_n = MX.sym("a_x_n")
a_x_s = a_max
a_x = a_x_n * a_x_s
a_y_n = MX.sym("a_y_n")
a_y_s = a_max
a_y = a_y_n * a_y_s
u_n = MX.sym("u_n")
u_s = vehicle_params["v_max"]
u = u_n * u_s
v_n = MX.sym("v_n")
v_s = 1.0
v = v_n * v_s
omega_z_n = MX.sym("omega_z_n")
omega_z_s = a_max / u_s
omega_z = omega_z_n * omega_z_s
delta_n = MX.sym("delta_n")
delta_s = vehicle_params["delta_max"] / 5.0
delta = delta_n * delta_s

N_fl_n = MX.sym("N_fl_n")
N_fr_n = MX.sym("N_fr_n")
N_rl_n = MX.sym("N_rl_n")
N_rr_n = MX.sym("N_rr_n")
N_fl_s = N_fr_s = N_rl_s = N_rr_s = tire_params["N_0"] * 4
N_fl = N_fl_n * N_fl_s
N_fr = N_fr_n * N_fr_s
N_rl = N_rl_n * N_rl_s
N_rr = N_rr_n * N_rr_s

F_x_n = MX.sym("F_x_n")
F_x_s = vehicle_params["m"] * a_max
F_x = F_x_n * F_x_s
F_x_fl_n = MX.sym("F_x_fl_n")
F_x_fr_n = MX.sym("F_x_fr_n")
F_x_rl_n = MX.sym("F_x_rl_n")
F_x_rr_n = MX.sym("F_x_rr_n")
F_x_fl_s = F_x_fr_s = F_x_rl_s = F_x_rr_s = F_x_s / 2
F_x_fl = F_x_fl_n * F_x_fl_s
F_x_fr = F_x_fr_n * F_x_fr_s
F_x_rl = F_x_rl_n * F_x_rl_s
F_x_rr = F_x_rr_n * F_x_rr_s

F_y_fl_n = MX.sym("F_y_fl_n")
F_y_fr_n = MX.sym("F_y_fr_n")
F_y_rl_n = MX.sym("F_y_rl_n")
F_y_rr_n = MX.sym("F_y_rr_n")
F_y_fl_s = F_y_fr_s = F_y_rl_s = F_y_rr_s = F_x_s / 2
F_y_fl = F_y_fl_n * F_y_fl_s
F_y_fr = F_y_fr_n * F_y_fr_s
F_y_rl = F_y_rl_n * F_y_rl_s
F_y_rr = F_y_rr_n * F_y_rr_s

kappa_fl_n = MX.sym("kappa_fl_n")
kappa_fr_n = MX.sym("kappa_fr_n")
kappa_rl_n = MX.sym("kappa_rl_n")
kappa_rr_n = MX.sym("kappa_rr_n")
kappa_fl_s = kappa_fr_s = kappa_rl_s = kappa_rr_s = max(kappa_max_list) / 2
kappa_fl = kappa_fl_n * kappa_fl_s
kappa_fr = kappa_fr_n * kappa_fr_s
kappa_rl = kappa_rl_n * kappa_rl_s
kappa_rr = kappa_rr_n * kappa_rr_s

lambda_fl_n = MX.sym("lambda_fl_n")
lambda_fr_n = MX.sym("lambda_fr_n")
lambda_rl_n = MX.sym("lambda_rl_n")
lambda_rr_n = MX.sym("lambda_rr_n")
lambda_fl_s = lambda_fr_s = lambda_rl_s = lambda_rr_s = max(lambda_max_list) / 2
lambda_fl = lambda_fl_n * lambda_fl_s
lambda_fr = lambda_fr_n * lambda_fr_s
lambda_rl = lambda_rl_n * lambda_rl_s
lambda_rr = lambda_rr_n * lambda_rr_s

x_n = vertcat(
    a_x_n, a_y_n, u_n, v_n, omega_z_n, delta_n,
    N_fl_n, N_fr_n, N_rl_n, N_rr_n,
    F_x_n, F_x_fl_n, F_x_fr_n, F_x_rl_n, F_x_rr_n,
    F_y_fl_n, F_y_fr_n, F_y_rl_n, F_y_rr_n,
    kappa_fl_n, kappa_fr_n, kappa_rl_n, kappa_rr_n,
    lambda_fl_n, lambda_fr_n, lambda_rl_n, lambda_rr_n,
)

x_s = vertcat(
    a_x_s, a_y_s, u_s, v_s, omega_z_s, delta_s,
    N_fl_s, N_fr_s, N_rl_s, N_rr_s,
    F_x_s, F_x_fl_s, F_x_fr_s, F_x_rl_s, F_x_rr_s,
    F_y_fl_s, F_y_fr_s, F_y_rl_s, F_y_rr_s,
    kappa_fl_s, kappa_fr_s, kappa_rl_s, kappa_rr_s,
    lambda_fl_s, lambda_fr_s, lambda_rl_s, lambda_rr_s,
)

# --- Drive force distribution (AWD 50:50) ---
k_t = Function("k_t", [a_x_n], [0.5])

# --- Aerodynamic forces ---
F_D = 0.5 * vehicle_params["rho"] * vehicle_params["C_D_A"] * u**2
F_Lf = 0.5 * vehicle_params["rho"] * vehicle_params["C_Lf_A"] * u**2
F_Lr = 0.5 * vehicle_params["rho"] * vehicle_params["C_Lr_A"] * u**2

# --- Tire deflection ---
df_z_fl = (N_fl - tire_params["N_0"]) / tire_params["N_0"]
df_z_fr = (N_fr - tire_params["N_0"]) / tire_params["N_0"]
df_z_rl = (N_rl - tire_params["N_0"]) / tire_params["N_0"]
df_z_rr = (N_rr - tire_params["N_0"]) / tire_params["N_0"]

# --- Theoretical slips ---
sigma_x_fl = kappa_fl / (1 + kappa_fl)
sigma_x_fr = kappa_fr / (1 + kappa_fr)
sigma_x_rl = kappa_rl / (1 + kappa_rl)
sigma_x_rr = kappa_rr / (1 + kappa_rr)
sigma_y_fl = tan(lambda_fl) / (1 + kappa_fl)
sigma_y_fr = tan(lambda_fr) / (1 + kappa_fr)
sigma_y_rl = tan(lambda_rl) / (1 + kappa_rl)
sigma_y_rr = tan(lambda_rr) / (1 + kappa_rr)

eps_sigma = 1e-6
sigma_fl = sqrt(sigma_x_fl**2 + sigma_y_fl**2 + eps_sigma)
sigma_fr = sqrt(sigma_x_fr**2 + sigma_y_fr**2 + eps_sigma)
sigma_rl = sqrt(sigma_x_rl**2 + sigma_y_rl**2 + eps_sigma)
sigma_rr = sqrt(sigma_x_rr**2 + sigma_y_rr**2 + eps_sigma)

# --- Magic formula coefficients (longitudinal) ---
K_x_fl = N_fl * tire_params["p_Kx_1"] * exp(tire_params["p_Kx_3"] * df_z_fl)
K_x_fr = N_fr * tire_params["p_Kx_1"] * exp(tire_params["p_Kx_3"] * df_z_fr)
K_x_rl = N_rl * tire_params["p_Kx_1"] * exp(tire_params["p_Kx_3"] * df_z_rl)
K_x_rr = N_rr * tire_params["p_Kx_1"] * exp(tire_params["p_Kx_3"] * df_z_rr)

D_x_fl = (tire_params["p_Dx_1"] + tire_params["p_Dx_2"] * df_z_fl) * tire_params["lambda_mu_x"]
D_x_fr = (tire_params["p_Dx_1"] + tire_params["p_Dx_2"] * df_z_fr) * tire_params["lambda_mu_x"]
D_x_rl = (tire_params["p_Dx_1"] + tire_params["p_Dx_2"] * df_z_rl) * tire_params["lambda_mu_x"]
D_x_rr = (tire_params["p_Dx_1"] + tire_params["p_Dx_2"] * df_z_rr) * tire_params["lambda_mu_x"]

B_x_fl = K_x_fl / (tire_params["p_Cx_1"] * D_x_fl * N_fl)
B_x_fr = K_x_fr / (tire_params["p_Cx_1"] * D_x_fr * N_fr)
B_x_rl = K_x_rl / (tire_params["p_Cx_1"] * D_x_rl * N_rl)
B_x_rr = K_x_rr / (tire_params["p_Cx_1"] * D_x_rr * N_rr)

# --- Magic formula coefficients (lateral) ---
K_y_fl = tire_params["N_0"] * tire_params["p_Ky_1"] * sin(2 * arctan(N_fl / (tire_params["p_Ky_2"] * tire_params["N_0"])))
K_y_fr = tire_params["N_0"] * tire_params["p_Ky_1"] * sin(2 * arctan(N_fr / (tire_params["p_Ky_2"] * tire_params["N_0"])))
K_y_rl = tire_params["N_0"] * tire_params["p_Ky_1"] * sin(2 * arctan(N_rl / (tire_params["p_Ky_2"] * tire_params["N_0"])))
K_y_rr = tire_params["N_0"] * tire_params["p_Ky_1"] * sin(2 * arctan(N_rr / (tire_params["p_Ky_2"] * tire_params["N_0"])))

D_y_fl = (tire_params["p_Dy_1"] + tire_params["p_Dy_2"] * df_z_fl) * tire_params["lambda_mu_y"]
D_y_fr = (tire_params["p_Dy_1"] + tire_params["p_Dy_2"] * df_z_fr) * tire_params["lambda_mu_y"]
D_y_rl = (tire_params["p_Dy_1"] + tire_params["p_Dy_2"] * df_z_rl) * tire_params["lambda_mu_y"]
D_y_rr = (tire_params["p_Dy_1"] + tire_params["p_Dy_2"] * df_z_rr) * tire_params["lambda_mu_y"]

B_y_fl = K_y_fl / (tire_params["p_Cy_1"] * D_y_fl * N_fl)
B_y_fr = K_y_fr / (tire_params["p_Cy_1"] * D_y_fr * N_fr)
B_y_rl = K_y_rl / (tire_params["p_Cy_1"] * D_y_rl * N_rl)
B_y_rr = K_y_rr / (tire_params["p_Cy_1"] * D_y_rr * N_rr)

# ============================================================
# Constraints (built once, parametric in p_V, p_g, p_alpha)
# ============================================================
g_con = []
lbg = []
ubg = []

# --- alpha and velocity constraints (use PARAMETERS) ---
g_con += [p_alpha - arctan2(a_x, a_y)]
g_con += [p_V - sqrt(u**2 + v**2)]
lbg += [0, 0]
ubg += [0, 0]

# --- lateral slips ---
# low-speed slip angle regularization
# Pacejka tire model has a known singularity at low speed: slip angle
# λ = vy/vx diverges as vx→0, causing artificial grip loss.
# Standard fix (CarSim, MF-Tire, etc.): regularize denominator with
# sqrt(denom² + eps_v²) so λ stays bounded at low speed while leaving
# mid/high-speed results unchanged.
eps_v = 0.5  # regularization velocity [m/s]

denom_fl = sqrt((u + 0.5 * vehicle_params["T"] * omega_z)**2 + eps_v**2)
denom_fr = sqrt((u - 0.5 * vehicle_params["T"] * omega_z)**2 + eps_v**2)
denom_rl = sqrt((u + 0.5 * vehicle_params["T"] * omega_z)**2 + eps_v**2)
denom_rr = sqrt((u - 0.5 * vehicle_params["T"] * omega_z)**2 + eps_v**2)

g_con += [lambda_fl - delta + (v + omega_z * vehicle_params["a"]) / denom_fl]
g_con += [lambda_fr - delta + (v + omega_z * vehicle_params["a"]) / denom_fr]
g_con += [lambda_rl + (v - omega_z * vehicle_params["b"]) / denom_rl]
g_con += [lambda_rr + (v - omega_z * vehicle_params["b"]) / denom_rr]

# (original code before regularization)
# g_con += [lambda_fl - delta + (v + omega_z * vehicle_params["a"]) / (u + 0.5 * vehicle_params["T"] * omega_z)]
# g_con += [lambda_fr - delta + (v + omega_z * vehicle_params["a"]) / (u - 0.5 * vehicle_params["T"] * omega_z)]
# g_con += [lambda_rl + (v - omega_z * vehicle_params["b"]) / (u + 0.5 * vehicle_params["T"] * omega_z)]
# g_con += [lambda_rr + (v - omega_z * vehicle_params["b"]) / (u - 0.5 * vehicle_params["T"] * omega_z)]
lbg += [0, 0, 0, 0]
ubg += [0, 0, 0, 0]

# --- slip limits ---
g_con += [lambda_fl + lambda_max(N_fl), lambda_fr + lambda_max(N_fr),
          lambda_rl + lambda_max(N_rl), lambda_rr + lambda_max(N_rr)]
lbg += [0.0, 0.0, 0.0, 0.0]
ubg += [np.inf, np.inf, np.inf, np.inf]
g_con += [lambda_fl - lambda_max(N_fl), lambda_fr - lambda_max(N_fr),
          lambda_rl - lambda_max(N_rl), lambda_rr - lambda_max(N_rr)]
lbg += [-np.inf, -np.inf, -np.inf, -np.inf]
ubg += [0.0, 0.0, 0.0, 0.0]

g_con += [kappa_fl + kappa_max(N_fl), kappa_fr + kappa_max(N_fr),
          kappa_rl + kappa_max(N_rl), kappa_rr + kappa_max(N_rr)]
lbg += [0.0, 0.0, 0.0, 0.0]
ubg += [np.inf, np.inf, np.inf, np.inf]
g_con += [kappa_fl - kappa_max(N_fl), kappa_fr - kappa_max(N_fr),
          kappa_rl - kappa_max(N_rl), kappa_rr - kappa_max(N_rr)]
lbg += [-np.inf, -np.inf, -np.inf, -np.inf]
ubg += [0.0, 0.0, 0.0, 0.0]

# --- Magic formula as constraints (8 eqs: 4 wheels × Fx + Fy) ---
for (F_xi, N_i, sigma_xi, sigma_i, D_xi, B_xi) in [
    (F_x_fl, N_fl, sigma_x_fl, sigma_fl, D_x_fl, B_x_fl),
    (F_x_fr, N_fr, sigma_x_fr, sigma_fr, D_x_fr, B_x_fr),
    (F_x_rl, N_rl, sigma_x_rl, sigma_rl, D_x_rl, B_x_rl),
    (F_x_rr, N_rr, sigma_x_rr, sigma_rr, D_x_rr, B_x_rr),
]:
    g_con += [F_xi - N_i * sigma_xi / sigma_i * D_xi * sin(
        tire_params["p_Cx_1"] * arctan(B_xi * sigma_i - tire_params["p_Ex_1"] * (B_xi * sigma_i - arctan(B_xi * sigma_i)))
    )]

for (F_yi, N_i, sigma_yi, sigma_i, D_yi, B_yi) in [
    (F_y_fl, N_fl, sigma_y_fl, sigma_fl, D_y_fl, B_y_fl),
    (F_y_fr, N_fr, sigma_y_fr, sigma_fr, D_y_fr, B_y_fr),
    (F_y_rl, N_rl, sigma_y_rl, sigma_rl, D_y_rl, B_y_rl),
    (F_y_rr, N_rr, sigma_y_rr, sigma_rr, D_y_rr, B_y_rr),
]:
    g_con += [F_yi - N_i * sigma_yi / sigma_i * D_yi * sin(
        tire_params["p_Cy_1"] * arctan(B_yi * sigma_i - tire_params["p_Ey_1"] * (B_yi * sigma_i - arctan(B_yi * sigma_i)))
    )]

lbg += [0] * 8
ubg += [0] * 8

# --- Roll stiffness balance ---
g_con += [vehicle_params["m"] * a_y * vehicle_params["h"] / vehicle_params["T"] * vehicle_params["epsilon"]
          - 0.5 * (N_fl - N_fr)]
lbg += [0]
ubg += [0]

# --- Drive force distribution (AWD 50:50) ---
g_con += [F_x_fl - 0.5 * (1.0 - k_t(a_x)) * F_x]
g_con += [F_x_fr - 0.5 * (1.0 - k_t(a_x)) * F_x]
g_con += [F_x_rl - 0.5 * k_t(a_x) * F_x]
g_con += [F_x_rr - 0.5 * k_t(a_x) * F_x]
lbg += [0, 0, 0, 0]
ubg += [0, 0, 0, 0]

# --- Steady state equations ---
g_con += [vehicle_params["m"] * a_x - (F_x_fl + F_x_fr + F_x_rl + F_x_rr) + (F_y_fl + F_y_fr) * delta + F_D]
g_con += [vehicle_params["m"] * a_y - (F_y_fl + F_y_fr + F_y_rl + F_y_rr) - (F_x_fl + F_x_fr) * delta]
g_con += [vehicle_params["m"] * p_g + F_Lf + F_Lr - N_fl - N_fr - N_rl - N_rr]
g_con += [vehicle_params["m"] * a_y * vehicle_params["h"] - 0.5 * vehicle_params["T"] * (N_fl - N_fr + N_rl - N_rr)]
g_con += [vehicle_params["m"] * a_x * vehicle_params["h"]
          - vehicle_params["a"] * F_Lf + vehicle_params["b"] * F_Lr
          + vehicle_params["a"] * (N_fl + N_fr) - vehicle_params["b"] * (N_rl + N_rr)]
g_con += [0.5 * vehicle_params["T"] * (F_y_fl - F_y_fr) * delta
          - vehicle_params["a"] * (F_x_fl + F_x_fr) * delta
          + 0.5 * vehicle_params["T"] * (-F_x_fl + F_x_fr - F_x_rl + F_x_rr)
          - vehicle_params["a"] * (F_y_fl + F_y_fr)
          + vehicle_params["b"] * (F_y_rl + F_y_rr)]
g_con += [a_y - omega_z * u]
lbg += [0, 0, 0, 0, 0, 0, 0]
ubg += [0, 0, 0, 0, 0, 0, 0]

# --- Engine power limit ---
# add brake power limit (was: lbg = -inf, no brake power constraint)
#       Physical meaning: F_brake * V <= P_brake_max  (brake power upper bound)
#       At high speed the ESC can't recover/dissipate arbitrary power, so
#       max deceleration is limited. If P_brake_max is None/missing, lbg
#       stays at -inf (original behavior, unbounded brake power).
# g_con += [F_x * u]
# lbg += [-float('inf')]
# ubg += [vehicle_params["P_max"]]
_P_brake_max = vehicle_params.get("P_brake_max", None)
_brake_lb = -float('inf') if _P_brake_max is None else -float(_P_brake_max)
g_con += [F_x * u]
lbg += [_brake_lb]
ubg += [vehicle_params["P_max"]]
if _P_brake_max is not None:
    print(f'[fast_gg] Brake power limit active: P_brake_max = {_P_brake_max} W')

# --- Velocity limit ---
g_con += [u]
lbg += [0.0]
ubg += [vehicle_params['v_max']]

# --- Steering angle limit ---
g_con += [delta]
lbg += [-vehicle_params["delta_max"]]
ubg += [vehicle_params["delta_max"]]

# --- Positive normal forces ---
g_con += [N_fl, N_fr, N_rl, N_rr]
lbg += [0.0, 0.0, 0.0, 0.0]
ubg += [tire_params["N_max"]] * 4

# --- Cost function: maximize rho = sqrt(ax² + ay²) ---
f = -a_x**2 - a_y**2

# ============================================================
# Build solver ONCE
# ============================================================
nlp = {"x": x_n, "f": f, "g": vertcat(*g_con), "p": p}
opts = {
    "verbose": False, "ipopt.print_level": 0, "print_time": 0,
    "ipopt.max_iter": 500,
    "ipopt.tol": 1e-6,
    "ipopt.acceptable_tol": 1e-4,
    "ipopt.acceptable_iter": 10,
    "ipopt.hessian_approximation": 'limited-memory',
    "ipopt.warm_start_init_point": 'yes',
    "ipopt.linear_solver": LINEAR_SOLVER,  # ma27 if HSL available, else mumps
}
solver = nlpsol("solver", "ipopt", nlp, opts)

lbg_vec = vertcat(*lbg)
ubg_vec = vertcat(*ubg)

# build variable bounds for ax/ay caps (null = no cap, original behavior)
#       These are speed-independent engineering safety limits on ax, ay applied
#       directly as variable bounds (not constraints) — IPOPT handles these for
#       free. Bounds go on the SCALED variables a_x_n, a_y_n, so divide caps
#       by the corresponding scale factors a_x_s, a_y_s.
#       Index 0 = a_x_n, index 1 = a_y_n (see x_n vertcat order above).
_n_vars = x_n.shape[0]
lbx_vec = -np.inf * np.ones(_n_vars)
ubx_vec =  np.inf * np.ones(_n_vars)

_ax_max_cap = vehicle_params.get("ax_max_cap", None)
_ax_min_cap = vehicle_params.get("ax_min_cap", None)
_ay_max_cap = vehicle_params.get("ay_max_cap", None)

if _ax_max_cap is not None:
    ubx_vec[0] = float(_ax_max_cap) / a_x_s
    print(f'[fast_gg] ax_max_cap active: {_ax_max_cap} m/s^2')
if _ax_min_cap is not None:
    lbx_vec[0] = -float(_ax_min_cap) / a_x_s
    print(f'[fast_gg] ax_min_cap active: {_ax_min_cap} m/s^2')
if _ay_max_cap is not None:
    ubx_vec[1] =  float(_ay_max_cap) / a_y_s
    lbx_vec[1] = -float(_ay_max_cap) / a_y_s
    print(f'[fast_gg] ay_max_cap active: ±{_ay_max_cap} m/s^2')

t_build = time.time() - t_build_start
print(f'[fast_gg] Solver built in {t_build:.2f}s')


# ============================================================
# Solve functions (same physics, just calls solver with params)
# ============================================================
def calc_gg_points(V, g_force, alpha_list_local):
    """Solve GG points for given V, g_force across all alpha values."""
    wb = vehicle_params["a"] + vehicle_params["b"]
    mu_x_max_val = tire_params["p_Dx_1"] * tire_params["lambda_mu_x"]
    N_ij0 = vehicle_params["m"] * g_force / 4.0

    ax_points = []
    ay_points = []
    beta_points = []
    x0_prev = None

    for alpha in alpha_list_local:
        # initial guess (same logic as original)
        rho_guess = mu_x_max_val * g_force * 0.3
        ax0 = rho_guess * np.sin(alpha)
        ay0 = rho_guess * np.cos(alpha)
        omega_z0 = ay0 / max(V, 0.5)
        delta0 = np.arctan(omega_z0 * wb / max(V, 0.5))
        Fx0 = vehicle_params["m"] * max(ax0, 0.01)
        Fy_pw = vehicle_params["m"] * ay0 / 4.0

        x0 = vertcat(
            ax0, ay0, V, ay0 / max(V, 0.5) * 0.1, omega_z0, delta0,
            N_ij0, N_ij0, N_ij0, N_ij0,
            Fx0, 0.0, 0.0, max(Fx0, 0.01) / 2, max(Fx0, 0.01) / 2,
            Fy_pw, Fy_pw, Fy_pw, Fy_pw,
            0.01, 0.01, 0.01, 0.01,
            0.01 + abs(delta0) * 0.5, 0.01 + abs(delta0) * 0.5, 0.01, 0.01,
        )
        x0_n = x0 / x_s

        # warm start from previous solution if available
        if x0_prev is not None:
            x0_n = x0_prev

        # call solver with parameters
        # pass ax/ay cap variable bounds (was: lbx=-inf, ubx=inf)
        #       Defaults to ±inf vectors if no caps set, so behavior is
        #       identical to the original when all caps are null.
        # x_opt = solver(x0=x0_n, lbx=-np.inf, ubx=np.inf,
        #                lbg=lbg_vec, ubg=ubg_vec,
        #                p=vertcat(V, g_force, alpha))
        x_opt = solver(x0=x0_n, lbx=lbx_vec, ubx=ubx_vec,
                       lbg=lbg_vec, ubg=ubg_vec,
                       p=vertcat(V, g_force, alpha))

        if solver.stats()["success"]:
            x0_prev = x_opt["x"]
            ax_points.append(float(x_opt["x"][0] * a_x_s))
            ay_points.append(float(x_opt["x"][1] * a_y_s))
            beta_points.append(float(arctan(abs(x_opt["x"][3] * v_s) / abs(x_opt["x"][2] * u_s))))
        else:
            pass

    return np.array(ax_points), np.array(ay_points), np.array(beta_points)


def gen_gg_polar(ax_ay_pairs, alpha_list_out):
    """Convert (ax, ay) pairs to polar rho representation."""
    if ax_ay_pairs.ndim < 2 or ax_ay_pairs.shape[0] < 2:
        return np.zeros(len(alpha_list_out))

    ax_neg_ay_pairs = ax_ay_pairs.copy()
    ax_neg_ay_pairs[:, 1] = -ax_neg_ay_pairs[:, 1]
    ax_ay_mirrored = np.row_stack((ax_ay_pairs, ax_neg_ay_pairs[::-1]))

    alpha_points = [np.arctan2(ax, ay) for ax, ay in ax_ay_mirrored]
    ax_interp = np.interp(alpha_list_out, alpha_points, ax_ay_mirrored[:, 0], period=2 * np.pi)
    ay_interp = np.interp(alpha_list_out, alpha_points, ax_ay_mirrored[:, 1], period=2 * np.pi)

    return np.sqrt(ax_interp**2 + ay_interp**2)


def rotate_by_beta(ax_vf, ay_vf, beta):
    ax_vel = np.cos(beta) * np.array(ax_vf).squeeze() + np.sin(beta) * np.array(ay_vf).squeeze()
    ay_vel = -np.sin(beta) * np.array(ax_vf).squeeze() + np.cos(beta) * np.array(ay_vf).squeeze()
    return ax_vel, ay_vel


def calc_rho_for_V(V):
    """Compute rho for all g values at given V."""
    rho_veh = []
    rho_vel = []
    for g_force in g_list:
        ax_vf, ay_vf, beta = calc_gg_points(V, g_force, alpha_list)
        ax_velf, ay_velf = rotate_by_beta(ax_vf, ay_vf, beta)
        rho_veh.append(gen_gg_polar(np.column_stack((ax_vf, ay_vf)), alpha_list_interp))
        rho_vel.append(gen_gg_polar(np.column_stack((ax_velf, ay_velf)), alpha_list_interp))
    return rho_veh, rho_vel


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # V_list / V_N already defined above (after V_max load) using V_step.
    print(f'[fast_gg] Starting GGV computation ({V_N} velocities, {num_cores} cores)...')
    t_solve_start = time.time()

    import multiprocessing as mp
    mp.set_start_method('fork', force=True)
    pool = mp.Pool(processes=min(num_cores, V_N))
    processed_list = pool.map(calc_rho_for_V, V_list)
    pool.close()
    pool.join()

    t_solve = time.time() - t_solve_start

    rho_vehicle_frame = [tmp[0] for tmp in processed_list]
    rho_velocity_frame = [tmp[1] for tmp in processed_list]

    # low-speed rho floor clamp
    # Pacejka underestimates lateral grip at low speed due to slip angle
    # artifact (large geometric slip ≠ actual sliding at low speed).
    # For each g, scan high→low speed on the lateral axis (alpha≈0) to
    # detect where rho drops sharply (>10%), then clamp all lower speeds
    # to the value just before the drop. Only raises values (max op).
    DROP_THRESHOLD = 0.10  # 10% drop triggers clamp
    alpha_lat_idx = int(np.argmin(np.abs(alpha_list_interp)))  # alpha≈0

    for rho_data in [rho_vehicle_frame, rho_velocity_frame]:
        for g_idx in range(g_N):
            # extract lateral rho at each velocity
            rho_lat = np.array([rho_data[v_idx][g_idx][alpha_lat_idx]
                                for v_idx in range(V_N)])
            # scan high→low speed: find first sharp drop
            v_ref_idx = 0  # fallback: lowest speed (no clamp)
            for v_idx in range(V_N - 1, 0, -1):
                if rho_lat[v_idx] > 1e-6:
                    drop = (rho_lat[v_idx] - rho_lat[v_idx - 1]) / rho_lat[v_idx]
                    if drop > DROP_THRESHOLD:
                        # use v_idx + 1 as floor instead of v_idx
                        #       When a >10% drop is detected between v_idx and v_idx-1,
                        #       v_idx itself is the UPPER side of the drop — but
                        #       empirically it's still in the SHOULDER (transitioning
                        #       into plateau), not fully on the plateau. The next-up
                        #       sample (v_idx+1) lands on the truly healthy region.
                        #
                        #       Verified on rc_car_10th fast-mode data:
                        #         before: floor=rho[V=4.125]=5.943 (3.7% undershoot)
                        #         after : floor=rho[V=6.75 ]=6.144 (0.4% undershoot)
                        #
                        #       min() clips to V_N-1 for the edge case where the drop
                        #       occurs at the very top sample (extremely rare since
                        #       the scan starts from the top of the plateau).
                        # v_ref_idx = v_idx
                        v_ref_idx = min(v_idx + 1, V_N - 1)
                        break
            # apply floor clamp for all speeds below v_ref
            if v_ref_idx > 0:
                floor = rho_data[v_ref_idx][g_idx]  # rho at v_ref for all alpha
                for v_idx in range(v_ref_idx):
                    rho_data[v_idx][g_idx] = np.maximum(rho_data[v_idx][g_idx], floor)
                # updated log — v_ref-1 no longer meaningful after v_idx+1 shift
                # print(f'[fast_gg] Floor clamp: g={g_list[g_idx]:.2f}, '
                #       f'V_ref={V_list[v_ref_idx]:.2f} m/s, '
                #       f'clamped V < {V_list[v_ref_idx]:.2f} '
                #       f'(lat rho {rho_lat[v_ref_idx-1]:.2f} → {rho_lat[v_ref_idx]:.2f})')
                print(f'[fast_gg] Floor clamp: g={g_list[g_idx]:.2f}, '
                      f'V_ref={V_list[v_ref_idx]:.2f} m/s, '
                      f'clamped V < {V_list[v_ref_idx]:.2f} '
                      f'(lat rho at V_ref = {rho_lat[v_ref_idx]:.2f})')

    # (original: rho saved directly without clamp)

    for frame in ["vehicle_frame", "velocity_frame"]:
        os.makedirs(os.path.join(out_path, frame), exist_ok=True)
        np.save(os.path.join(out_path, frame, "v_list.npy"), V_list)
        np.save(os.path.join(out_path, frame, "g_list.npy"), g_list)
        np.save(os.path.join(out_path, frame, "alpha_list.npy"), alpha_list_interp)
        np.save(
            os.path.join(out_path, frame, "rho.npy"),
            np.asarray(rho_vehicle_frame) if frame == "vehicle_frame" else np.asarray(rho_velocity_frame),
        )

    t_total = time.time() - t_build_start
    n_nlp = V_N * g_N * len(alpha_list)
    print(f'\n[fast_gg] ===== DONE (standard) =====')
    print(f'[fast_gg] Solver build: {t_build:.2f}s')
    print(f'[fast_gg] NLP solve:    {t_solve:.2f}s  ({n_nlp} calls)')
    print(f'[fast_gg] Total:        {t_total:.2f}s')
    print(f'[fast_gg] Output: {out_path}')

# EOF
