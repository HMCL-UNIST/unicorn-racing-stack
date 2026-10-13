import os
import argparse
# yaml for tuning override (was not imported)
import yaml
from casadi import *
import multiprocessing
from joblib import Parallel, delayed

# add argparse for vehicle_name (was hardcoded to 'rc_car_10th')
parser = argparse.ArgumentParser(description='Generate diamond representation for GG diagrams')
parser.add_argument('--vehicle_name', type=str, default='rc_car_10th', help='Vehicle name (default: rc_car_10th)')
# add tuning CLI args (same as fast_gen_gg_diagrams.py) to enable
#       diamond post-process (gg_exp_scale, ax_max/min_scale, ay_scale).
#       --tuning_name decouples the tuning file name from --vehicle_name.
parser.add_argument('--tuning', action='store_true', default=False,
                    help='Apply tuning_<name>.yml post-process overrides')
parser.add_argument('--tuning_name', type=str, default=None,
                    help='Tuning file suffix (default: same as --vehicle_name)')
args, _ = parser.parse_known_args()
vehicle_name = args.vehicle_name
# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
# read from fast_ggv_gen/output/ (never touch original gg_diagrams)
gg_diagram_path = os.path.join(dir_path, 'output')

num_cores = multiprocessing.cpu_count()

# load diamond post-process params
#       Reads post-process keys from the UNIFIED params yml
#       (params_<vehicle_name>.yml), which contains all NLP + post-process
#       + raceline keys in one file. NLP/raceline keys are ignored here.
#       If --tuning flag is given, also tries tuning_<name>.yml as fallback
#       (backward compatibility with manual CLI usage).
#       All params default to None (= apply nothing = NLP fit as-is).
# (previous: read from tuning_<name>.yml only via --tuning flag)
tuning_post = {
    'gg_exp_scale':  None,
    'ax_max_scale':  None,
    'ax_min_scale':  None,
    'ay_scale':      None,
}

_data_path = os.path.join(dir_path, '..', 'global_line', 'data')

# primary: read from params_<vehicle_name>.yml (unified yml from gg_tuner)
_params_path = os.path.join(_data_path, 'vehicle_params',
                            'params_' + vehicle_name + '.yml')
_loaded_from = None
if os.path.exists(_params_path):
    with open(_params_path, 'r') as _stream:
        _params_raw = yaml.safe_load(_stream) or {}
    for _key in tuning_post:
        if _key in _params_raw and _params_raw[_key] is not None:
            tuning_post[_key] = float(_params_raw[_key])
    _loaded_from = _params_path

# fallback: if --tuning flag given, also check tuning_<name>.yml
#       (for backward compat with manual CLI: run_on_container.sh with TUNING=1)
if args.tuning and _loaded_from is None:
    _tuning_name = args.tuning_name if args.tuning_name else vehicle_name
    _tuning_path = os.path.join(_data_path, 'vehicle_params',
                                'tuning_' + _tuning_name + '.yml')
    if os.path.exists(_tuning_path):
        with open(_tuning_path, 'r') as _stream:
            _tuning_raw = yaml.safe_load(_stream) or {}
        for _key in tuning_post:
            if _key in _tuning_raw and _tuning_raw[_key] is not None:
                tuning_post[_key] = float(_tuning_raw[_key])
        _loaded_from = _tuning_path
    else:
        print(f'[gen_diamond] WARNING: --tuning specified but {_tuning_path} '
              f'not found')

# Only log params that actually deviate from the no-op value (1.0).
_active = {k: v for k, v in tuning_post.items()
           if v is not None and v != 1.0}
if _active:
    print(f'[gen_diamond] Post-process params from {_loaded_from}:')
    for _k, _v in _active.items():
        print(f'               {_k} = {_v}')


# x0_seed enables warm-start; also returns IPOPT success flag and final
#       objective value f_val so the multi-start wrapper can pick the best
#       healthy fit (default cold-start alone gets stuck in gg_exp=2 ax_max=0
#       local optimum on some V/g cells).
def gen_diamond_representation(alpha_list, rho_list, x0_seed=None):
    # diamond representation
    gg_exponent = MX.sym('gg_exponent')
    ax_min = MX.sym('ax_min')
    ax_max = MX.sym('ax_max')
    ay_max = MX.sym('ay_max')

    x = vertcat(
        gg_exponent,
        ax_min,
        ax_max,
        ay_max
    )
    lbx = vertcat(
        1.0,
        - np.interp(-np.pi / 2.0, alpha_list, rho_list),
        0.0,
        0.0
    )
    ubx = vertcat(
        2.0,
        0.0,
        np.interp(np.pi / 2.0, alpha_list, rho_list),
        np.interp(0.0, alpha_list, rho_list, 0.0)
    )

    f = 0
    g_const = []
    lbg = []
    ubg = []

    for alpha in np.linspace(
            -np.pi / 2.0,
            np.pi / 2.0,
            200,
    ):
        rho_max = np.interp(alpha, alpha_list, rho_list)

        ay = power(
            fabs(ax_min) ** gg_exponent / (
                        tan(fabs(alpha)) ** gg_exponent + (fabs(ax_min) / ay_max) ** gg_exponent),
            1.0 / gg_exponent
        )
        ax = ay * tan(fabs(alpha))
        rho_diamond = sqrt(ax ** 2 + ay ** 2)

        if alpha > 0.0:
            rho_diamond = fmin(
                rho_diamond,
                ax_max / sin(alpha)
            )

        g_const += [rho_max - rho_diamond]
        lbg += [0.0]
        ubg += [np.inf]

        f -= rho_diamond**2

    # x0 from seed when provided, default cold-start otherwise
    if x0_seed is None:
        x0 = vertcat(1.0, -5.1, 5.0, 5.0)
    else:
        x0 = vertcat(*x0_seed)
    nlp = {"x": x, "f": f, "g": vertcat(*g_const)}
    opts = {
        "verbose": False, "ipopt.print_level": 0, "print_time": 0,
        "ipopt.hessian_approximation": 'limited-memory',
    }
    solver = nlpsol("solver", "ipopt", nlp, opts)
    # capture full result dict so success flag + f_val are available
    sol = solver(x0=x0, lbx=lbx, ubx=ubx,
                 lbg=vertcat(*lbg), ubg=vertcat(*ubg))
    x_opt = sol['x']
    success = bool(solver.stats().get('success', False))
    f_val = float(sol['f'])

    gg_exponent = float(x_opt[0])
    ax_min = float(x_opt[1])
    ax_max = float(x_opt[2])
    ay_max = float(x_opt[3])

    return gg_exponent, ax_min, ax_max, ay_max, success, f_val


# multi-start wrapper - tries warm + cold + geometric seeds and picks
#       the best healthy fit. Healthy = IPOPT success AND not in the
#       (gg_exp~2, ax_max << physical_bound) local-optimum signature.
def gen_diamond_multi_start(alpha_list, rho_list, x0_warm=None):
    # Physical bounds for the fail-signature filter (same as inner NLP's ubx).
    ubx_ax_max = float(np.interp(np.pi / 2.0, alpha_list, rho_list))
    lbx_ax_min = -float(np.interp(-np.pi / 2.0, alpha_list, rho_list))
    ubx_ay_max = float(np.interp(0.0, alpha_list, rho_list, 0.0))

    seeds = []
    if x0_warm is not None:
        seeds.append(('warm', tuple(x0_warm)))
    seeds.append(('cold', (1.0, -5.1, 5.0, 5.0)))
    # Geometric seed: scaled to the actual rho envelope, gg_exp away from bound.
    seeds.append(('geom', (1.5,
                           0.9 * lbx_ax_min,
                           0.9 * ubx_ax_max,
                           0.9 * ubx_ay_max)))

    results = []  # list of (tag, vals_4tuple, success, f_val)
    for tag, seed in seeds:
        gg_e, ax_n, ax_x, ay_x, ok, f_val = gen_diamond_representation(
            alpha_list, rho_list, x0_seed=seed)
        results.append((tag, (gg_e, ax_n, ax_x, ay_x), ok, f_val))

    def is_healthy(r):
        _, (gg_e, _, ax_x, _), ok, _ = r
        if not ok:
            return False
        # Fail signature: stuck at gg_exp upper bound with collapsed ax_max.
        if gg_e >= 1.99 and ax_x < 0.3 * max(ubx_ax_max, 1e-6):
            return False
        return True

    healthy_results = [r for r in results if is_healthy(r)]
    pool = healthy_results if healthy_results else results
    # Best = most negative f_val (objective is -sum(rho_diamond^2)).
    best = min(pool, key=lambda r: r[3])
    return best[1]


def gen_diamond_representation_for_V(alpha_list, rho_list):
    gg_exponent_tmp = []
    ax_min_tmp = []
    ax_max_tmp = []
    ay_max_tmp = []

    # warm-start chain along g axis at fixed V
    x0_warm = None
    for rho in rho_list:
        # [TEMP] skip diamond fitting for all-zero rho entries
        # caused by gen_gg_diagrams solver failing at extreme V/g combos
        # TODO: fix by improving solver scaling or narrowing g range, then remove this
        if np.all(rho == 0):
            gg_exponent_tmp.append(1.0)
            ax_min_tmp.append(0.0)
            ax_max_tmp.append(0.0)
            ay_max_tmp.append(0.0)
            # Do not propagate warm-start through the zero gap.
            x0_warm = None
            continue
        gg_exponent, ax_min, ax_max, ay_max = gen_diamond_multi_start(
            alpha_list=alpha_list,
            rho_list=rho,
            x0_warm=x0_warm,
        )
        gg_exponent_tmp.append(gg_exponent)
        ax_min_tmp.append(ax_min)
        ax_max_tmp.append(ax_max)
        ay_max_tmp.append(ay_max)
        x0_warm = (gg_exponent, ax_min, ax_max, ay_max)

    return gg_exponent_tmp, ax_min_tmp, ax_max_tmp, ay_max_tmp


for frame in ['vehicle', 'velocity']:
    path = os.path.join(gg_diagram_path, vehicle_name, frame + '_frame')
    V_list = np.load(os.path.join(path, 'v_list.npy'))
    V_max = V_list.max()
    g_list = np.load(os.path.join(path, 'g_list.npy'))
    g_min = g_list.min()
    g_max = g_list.max()
    # polar coordinates
    alpha_list = np.load(os.path.join(path, 'alpha_list.npy'))
    rho_list = np.load(os.path.join(path, 'rho.npy'))

    processed_list = Parallel(
        n_jobs=num_cores
    )(
        delayed(gen_diamond_representation_for_V)(alpha_list, rho) for rho in rho_list
    )

    gg_exponent_list = [tmp[0] for tmp in processed_list]
    ax_min_list = [tmp[1] for tmp in processed_list]
    ax_max_list = [tmp[2] for tmp in processed_list]
    ay_max_list = [tmp[3] for tmp in processed_list]

    # V-axis safety net for residual bad cells
    #       Detects fit failures that survived multi-start (gg_exp at upper
    #       bound 2 with ax_max collapsed far below physical bound, OR the
    #       all-zero skip path from upstream rho==0). Each flagged cell is
    #       replaced by linear interpolation along V from the nearest
    #       healthy V cells at the same g.
    gg_arr     = np.asarray(gg_exponent_list, dtype=np.float64)
    ax_min_arr = np.asarray(ax_min_list,      dtype=np.float64)
    ax_max_arr = np.asarray(ax_max_list,      dtype=np.float64)
    ay_max_arr = np.asarray(ay_max_list,      dtype=np.float64)

    n_V, n_g = ax_max_arr.shape
    # Per-cell physical bound on ax_max (= rho at alpha=pi/2).
    bound_ax_max = np.zeros((n_V, n_g), dtype=np.float64)
    for _vi in range(n_V):
        for _gi in range(n_g):
            bound_ax_max[_vi, _gi] = float(
                np.interp(np.pi / 2.0, alpha_list, rho_list[_vi, _gi]))

    zero_skip = ((ax_max_arr == 0.0) & (ax_min_arr == 0.0)
                 & (ay_max_arr == 0.0) & (gg_arr == 1.0))
    fail_sig = (gg_arr >= 1.99) & (
        ax_max_arr < 0.3 * np.maximum(bound_ax_max, 1e-6))
    bad = zero_skip | fail_sig

    n_fixed = 0
    for vi, gi in zip(*np.where(bad)):
        col_bad = bad[:, gi]
        lo = vi - 1
        while lo >= 0 and col_bad[lo]:
            lo -= 1
        hi = vi + 1
        while hi < n_V and col_bad[hi]:
            hi += 1
        if lo < 0 and hi >= n_V:
            print(f'[gen_diamond] [{frame}_frame] V-axis sanitize: '
                  f'no healthy V at g_idx={gi}, leaving v_idx={vi} as-is')
            continue
        if lo < 0:
            new = (gg_arr[hi, gi], ax_min_arr[hi, gi],
                   ax_max_arr[hi, gi], ay_max_arr[hi, gi])
        elif hi >= n_V:
            new = (gg_arr[lo, gi], ax_min_arr[lo, gi],
                   ax_max_arr[lo, gi], ay_max_arr[lo, gi])
        else:
            t = (vi - lo) / float(hi - lo)
            new = (
                (1.0 - t) * gg_arr[lo, gi]     + t * gg_arr[hi, gi],
                (1.0 - t) * ax_min_arr[lo, gi] + t * ax_min_arr[hi, gi],
                (1.0 - t) * ax_max_arr[lo, gi] + t * ax_max_arr[hi, gi],
                (1.0 - t) * ay_max_arr[lo, gi] + t * ay_max_arr[hi, gi],
            )
        old_ax = ax_max_arr[vi, gi]
        gg_arr[vi, gi], ax_min_arr[vi, gi], ax_max_arr[vi, gi], ay_max_arr[vi, gi] = new
        print(f'[gen_diamond] [{frame}_frame] V-axis sanitize: '
              f'V={V_list[vi]:.2f} g={g_list[gi]:.2f} '
              f'ax_max {old_ax:.3f} -> {new[2]:.3f}')
        n_fixed += 1
    if n_fixed > 0:
        print(f'[gen_diamond] [{frame}_frame] V-axis sanitize: '
              f'{n_fixed} cell(s) replaced')

    gg_exponent_list = gg_arr.tolist()
    ax_min_list      = ax_min_arr.tolist()
    ax_max_list      = ax_max_arr.tolist()
    ay_max_list      = ay_max_arr.tolist()

    # apply diamond post-process scales (null = NLP fit as-is)
    #       - gg_exp_scale: multiplies fitted exponent, clipped to [1.0, 2.0].
    #         Preserves V/g variation while shifting toward diamond (<1) or
    #         ellipse (>1). Does NOT throw away NLP variation.
    #       - ax_max_scale / ax_min_scale: directional accel/brake scaling
    #         (can be used independently to bias brake-heavy vs accel-heavy).
    #       - ay_scale: corner capability scaling.
    #       Applied AFTER diamond fit, to both vehicle and velocity frames.
    gg_arr = np.asarray(gg_exponent_list, dtype=np.float64)
    ax_min_arr = np.asarray(ax_min_list,  dtype=np.float64)
    ax_max_arr = np.asarray(ax_max_list,  dtype=np.float64)
    ay_max_arr = np.asarray(ay_max_list,  dtype=np.float64)

    _applied_log = []
    if tuning_post['gg_exp_scale'] is not None:
        _s = tuning_post['gg_exp_scale']
        gg_before_min, gg_before_max = float(gg_arr.min()), float(gg_arr.max())
        gg_arr = np.clip(gg_arr * _s, 1.0, 2.0)
        _applied_log.append(
            f"gg_exp_scale={_s}: [{gg_before_min:.3f},{gg_before_max:.3f}] → "
            f"[{gg_arr.min():.3f},{gg_arr.max():.3f}]")
    if tuning_post['ax_max_scale'] is not None:
        _s = tuning_post['ax_max_scale']
        ax_max_arr = ax_max_arr * _s
        _applied_log.append(f"ax_max_scale={_s}")
    if tuning_post['ax_min_scale'] is not None:
        _s = tuning_post['ax_min_scale']
        ax_min_arr = ax_min_arr * _s   # ax_min is already negative
        _applied_log.append(f"ax_min_scale={_s}")
    if tuning_post['ay_scale'] is not None:
        _s = tuning_post['ay_scale']
        ay_max_arr = ay_max_arr * _s
        _applied_log.append(f"ay_scale={_s}")

    if _applied_log:
        print(f'[gen_diamond] [{frame}_frame] post-process applied: '
              + ', '.join(_applied_log))

    gg_exponent_list = gg_arr.tolist()
    ax_min_list      = ax_min_arr.tolist()
    ax_max_list      = ax_max_arr.tolist()
    ay_max_list      = ay_max_arr.tolist()

    out_path = os.path.join(gg_diagram_path, vehicle_name)
    np.save(os.path.join(out_path, frame + '_frame', "gg_exponent.npy"), gg_exponent_list)
    np.save(os.path.join(out_path, frame + '_frame', "ax_min.npy"), ax_min_list)
    np.save(os.path.join(out_path, frame + '_frame', "ax_max.npy"), ax_max_list)
    np.save(os.path.join(out_path, frame + '_frame', "ay_max.npy"), ay_max_list)

# EOF
