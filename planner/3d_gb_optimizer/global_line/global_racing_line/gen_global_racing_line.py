# work around the numpy 1.17.4 + pandas 2.0.3 version mismatch
# block the broken pandas in ~/.local + spoof the numpy version + inject BitGenerator
import sys as _sys
_sys.path[:] = [p for p in _sys.path if '/.local/' not in p]
import numpy as _np
_np.__version__ = '1.22.4'
if not hasattr(_np.random, 'BitGenerator'):
    from numpy.random.bit_generator import BitGenerator as _BG
    _np.random.BitGenerator = _BG

import os
import sys
import casadi as ca
import numpy as np
import yaml
import pandas as pd


# Probe HSL ma27; fall back to MUMPS if libhsl.so is not loadable.
#       ma27 is typically 1.5–3x faster than MUMPS for this NLP size.
#       Original MUMPS behavior is preserved when HSL is not installed —
#       no existing workflow breaks.
def _select_linear_solver(verbose=True):
    try:
        _x = ca.MX.sym('x')
        _probe = ca.nlpsol('hsl_probe', 'ipopt',
                           {'x': _x, 'f': (_x - 1.0) ** 2},
                           {'ipopt.linear_solver': 'ma27',
                            'ipopt.print_level': 0, 'print_time': 0})
        _probe(x0=0.0)
        if _probe.stats().get('success', False):
            if verbose:
                print('[gen_global_racing_line] linear_solver: ma27 (HSL)')
            return 'ma27'
    except Exception:
        pass
    if verbose:
        print('[gen_global_racing_line] linear_solver: mumps (HSL not available, fallback)')
    return 'mumps'

_LINEAR_SOLVER = _select_linear_solver()

params = {
    # 'track_name': 'experiment_3d_2_3d_smoothed.csv',
    # 'raceline_name': 'experiment_3d_2_3d_rc_car_timeoptimal.csv',
    # 'track_name': 'mount_panorama_3d_smoothed.csv',
    # 'raceline_name': 'mount_panorama_3d_dallaraAV21_timeoptimal.csv',
    'track_name': 'experiment_3d_2_3d_smoothed_waypoint1.csv',
    'raceline_name': 'experiment_3d_2_3d_smoothed_waypoint1_timeoptimal_curve.csv',
    'vehicle_name': 'rc_car_10th',
    'gg_vehicle_name': 'rc_car_10th',  # use rc_car_10th GG diagrams
    'safety_distance': 0.3,  # safety distance to track bounds in m
    'gg_mode': 'diamond',  # polar, diamond
    'gg_margin': 0.0,
    'neglect_w_omega_y': True,
    'neglect_w_omega_x': True,
    'neglect_euler': True,
    'neglect_centrifugal': True,
    'neglect_w_dot': False,
    'neglect_V_omega': False,
    'V_guess': 3.0,  # initial velocity guess
    # minimum velocity bound for raceline NLP
    #       Prevents the optimizer from planning V < V_min anywhere on the
    #       track. Consistent with fast_gen_gg_diagrams.py V_min=2.0 (which
    #       means the GGV lookup is only valid for V ≥ 2). Forces the
    #       optimizer to take wider lines at tight corners instead of
    #       crawling, and sidesteps low-speed GGV artifacts entirely.
    #         0.0 = unconstrained (original behavior)
    #         2.0 = current choice (matches GGV V_min)
    #       Infeasibility risk: if the track has a hairpin too tight for
    #       V≥V_min at the max available ay, the NLP will fail. Start
    #       lower (1.5 or 1.8) if feasibility is uncertain.
    'V_min': 2.0,
    # bridge mu scale baked into raceline (1.0=as-is, 0=flat, 1.5=amplify)
    'g_weight': 1.0,
    # bridge centerline / heading-lock constraints.
    # Yaml at <map>/bridge_sectors.yaml stores *positions only*
    # (Bridge{i}: {start, end} — smoothed-CSV row indices). The tuning
    # knobs below are GLOBAL — applied to every bridge in the yaml.
    # They come from rqt (Bridge group) and are forwarded via
    # run_pipeline.sh args. Default values reproduce legacy behavior:
    # no extension, no centerline pin, no heading bound. Missing yaml
    # OR bridge_constraints_enable=False ⇒ no added constraints.
    'bridge_constraints_enable': True,
    'bridge_yaml_dir': '',  # injected by run_pipeline.sh (MAP_DIR)
    'bridge_pre_m':        0.0,
    'bridge_post_m':       0.0,
    'bridge_force_center': False,
    'bridge_chi_max_rad':  -1.0,
    # per-sector safety_distance — friction-pattern rosparam tree at
    # /safety_sector_params/... populated by safety_sector_server.py.
    # When the rosparam tree exists and n_sectors > 0, the lateral
    # track-bound inflation in _state_bounds() becomes s-dependent
    # (per-sector wall margin with linear pre/post blend to the
    # default). Missing rosparam / n_sectors=0 ⇒ the per-point d_safe
    # collapses to the scalar `safety_distance`, so legacy behavior is
    # preserved bit-for-bit.
    'w_jx': 1e-2,  # cost weight for jerk x-direction
    'w_jy': 1e-2,  # cost weight for jerk y-direction
    # min time
    'w_dOmega_z': 0.0,  # cost weight for curvature in road plane (e.g. 1e3)
    'w_T': 1e0,  # cost weight for time (should be 1)

    # # min curve
    # 'w_dOmega_z': 1e-1,  # cost weight for curvature in road plane (e.g. 1e3)
    # 'w_T': 0.0,  # cost weight for time (should be 1)


    # optimization grid spacing (independent from smooth step_size)
    'step_size_opt': 0.2,  # [m] NLP grid spacing (coarser than smooth for performance)
    'RK4_steps': 1,
    'sol_opts': {
        "ipopt.max_iter": 2000,
        "ipopt.hessian_approximation": 'limited-memory',  # 'exact' or 'limited-memory'
        "ipopt.line_search_method": 'cg-penalty',  # 'filter', 'cg-penalty'
        # physically-justified tolerance relaxation for V_min=2.0 NLP
        #       Problem: after enforcing V_min=2.0, the raceline NLP develops
        #       near-degenerate active constraints at tight corners where both
        #       V=V_min and ay=ay_max are simultaneously binding. This causes
        #       dual multiplier ambiguity — solver primal-converges
        #       (inf_pr ~ 1e-10) but inf_du oscillates around 0.15 without exit.
        #
        #       Scale analysis (justification for each tolerance):
        #         - Objective J ≈ 20 s (lap time)
        #         - |dJ/dV| ≈ 1/V² ∈ [0.01, 0.25] over V ∈ [2, 10]
        #         - Default tol=1e-8 → excessive for this problem scale
        #         - tol=1e-4 → worst-case obj error ≈ 1e-4 · 20 ≈ 2 ms
        #         - dual_inf_tol=1e-1 → ~40% relative error on gradient scale
        #         - acceptable_dual_inf_tol=5.0 (observed: inf_du oscillates
        #           0.08~2.5 during active basin exploration — 1.0 kept
        #           resetting the acceptable counter whenever spikes hit 2+,
        #           so raised to 5.0 to absorb those spikes)
        #         - acceptable_iter=10 (default 15, slightly less conservative
        #           but still robust to noise-level oscillations)
        #       Constraint-violation tolerances kept TIGHT (1e-4) because
        #       primal violation means physically invalid trajectory (outside
        #       track, exceeding GGV envelope). Do not relax constr_viol_tol.
        # (previous)
        # "ipopt.acceptable_tol": 1e-4,
        # "ipopt.acceptable_dual_inf_tol": 1e-4,
        # "ipopt.constr_viol_tol": 1e-4,
        "ipopt.tol":                        1e-4,
        "ipopt.dual_inf_tol":               1e-1,
        "ipopt.constr_viol_tol":            1e-4,
        "ipopt.compl_inf_tol":              1e-4,
        "ipopt.acceptable_tol":             1e-3,
        "ipopt.acceptable_dual_inf_tol":    5.0,
        "ipopt.acceptable_constr_viol_tol": 1e-3,
        "ipopt.acceptable_iter":            10,
        # HSL ma27 if available (probed at import), else MUMPS fallback.
        "ipopt.linear_solver": _LINEAR_SOLVER,
    }
}

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
vehicle_params_path = os.path.join(data_path, 'vehicle_params', 'params_' + params['vehicle_name'] + '.yml')
gg_diagram_path = os.path.join(data_path, 'gg_diagrams', params.get('gg_vehicle_name', params['vehicle_name']), 'velocity_frame')
track_path = os.path.join(data_path, 'smoothed_track_data')
raceline_out_path = os.path.join(data_path, 'global_racing_lines')
sys.path.append(os.path.join(dir_path, '..', 'src'))

# load vehicle and tire parameters
with open(vehicle_params_path, 'r') as stream:
    params.update(yaml.safe_load(stream))

from track3D import Track3D
from ggManager import GGManager


# bridge sector adapter.
# Reads <map>/bridge_sectors.yaml (optional, positions-only) and the
# *global* tuning knobs forwarded from rqt, then projects them onto
# the NLP grid produced by track_handler.resample(...). Returns a list
# `bridge_state` of length track_handler.s.size where each entry is
# either None (no bridge ⇒ legacy bounds) or a dict
# {'force_center': bool, 'chi_max_rad': float}.
#
#      Yaml schema (positions only — tuning lives in rqt):
#        n_bridges: N
#        Bridge0: {start: <idx>, end: <idx>}
#        Bridge1: {start: <idx>, end: <idx>}
#        ...
#
#      Tuning args (global, from rqt → run_pipeline.sh args):
#        pre_m, post_m  : extend each bridge zone before/after [m of s]
#        force_center   : pin n ≈ 0 inside every bridge
#        chi_max_rad    : bound |chi| inside every bridge (<0 disables)
#
#      Index mapping: yaml stores start/end as smoothed-CSV row indices
#      (pre-resample). We resolve those to s [m] via the smoothed CSV's
#      s_m column, extend by ±(pre_m, post_m), then mark every NLP grid
#      step whose s falls inside the union of zones.
#
#      No-op cases (all silent — non-bridge maps work unchanged):
#        - enable=False
#        - yaml file missing / yaml has 0 bridges
#        - force_center=False AND chi_max_rad<0 (no actual constraint set)
#        - smoothed CSV missing (defensive)
def _build_bridge_state_table(track_handler, smoothed_csv_path,
                              bridge_yaml_dir, enable,
                              pre_m, post_m, force_center, chi_max_rad):
    n_grid = int(track_handler.s.size)
    none_table = [None] * n_grid
    if not enable:
        return none_table
    # both knobs at "off" ⇒ nothing to enforce even if yaml exists.
    # early-exit only when BOTH CLI fallbacks are disabled (no yaml
    # bridge can override since per-bridge entries can opt into one
    # knob even if the CLI fallback has it off — but in that case the
    # per-bridge fields are the truth, so we don't short-circuit here).
    # yaml-enable check below subsumes the all-disabled case anyway.
    if not bridge_yaml_dir:
        return none_table
    yaml_path = os.path.join(bridge_yaml_dir, 'bridge_sectors.yaml')
    if not os.path.exists(yaml_path):
        return none_table
    try:
        with open(yaml_path, 'r') as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        print(f'[gen_global_racing_line] bridge yaml parse failed '
              f'({yaml_path}): {e} — no-op')
        return none_table
    # yaml-level master enable overrides CLI. If yaml says off, no
    # bridges apply regardless of CLI args (yaml is single truth).
    if not bool(doc.get('bridge_constraints_enable', True)):
        return none_table
    n_bridges = int(doc.get('n_bridges', 0))
    if n_bridges <= 0:
        return none_table
    if not os.path.exists(smoothed_csv_path):
        print(f'[gen_global_racing_line] smoothed CSV not found '
              f'({smoothed_csv_path}) — bridge yaml ignored')
        return none_table
    try:
        df_smooth = pd.read_csv(smoothed_csv_path)
        s_smooth = df_smooth['s_m'].to_numpy()
    except Exception as e:
        print(f'[gen_global_racing_line] failed to read s_m from '
              f'{smoothed_csv_path}: {e} — bridge yaml ignored')
        return none_table
    s_total = float(s_smooth[-1] - s_smooth[0]) if len(s_smooth) > 1 else 0.0
    closed_loop = s_total > 0.0

    s_grid = track_handler.s
    pre_m_f = float(pre_m)
    post_m_f = float(post_m)
    # per-grid-step `ramp ∈ [0, 1]` blending legacy bound (ramp=0) to
    # full lock (ramp=1). NLP can solve discontinuous box constraints
    # fine — the ramp just makes the tuning meaning more intuitive:
    # pre_m/post_m become "ramp length" rather than "hard zone
    # extension", and the corridor narrows progressively so a small
    # pre_m still produces a smooth approach.
    # core (s_start..s_end):                 ramp = 1
    # pre  (s_start-pre_m..s_start):         ramp linear 0 → 1
    # post (s_end..s_end+post_m):            ramp linear 1 → 0
    # else:                                  ramp = 0 (entry None)
    # Overlapping bridges take the *max* ramp at each step.
    table = [None] * n_grid
    n_used = 0

    def _bridge_ramp(sk_in, core_start, core_end, pre, post):
        """Return ramp value at sk_in for one bridge's extended zone.
        Output is 1.0 in the hard core, linear 0→1 across pre, linear 1→0
        across post, and 0.0 outside the extended zone (which we should not
        be queried at).
        """
        if core_start <= sk_in <= core_end:
            return 1.0
        if (core_start - pre) <= sk_in < core_start and pre > 0.0:
            return (sk_in - (core_start - pre)) / pre
        if core_end < sk_in <= (core_end + post) and post > 0.0:
            return 1.0 - (sk_in - core_end) / post
        # pre==0 or post==0 boundary case — caller already filtered
        # `in_extended` above, so we're inside the extended zone but the
        # ramp length is zero ⇒ full lock immediately (legacy hard
        # behaviour preserved when pre/post are 0).
        return 1.0

    for i in range(n_bridges):
        b = doc.get(f'Bridge{i}')
        if not isinstance(b, dict):
            continue
        try:
            s_i = int(b['start'])
            e_i = int(b['end'])
        except (KeyError, ValueError, TypeError):
            continue
        s_i = max(0, min(s_i, len(s_smooth) - 1))
        e_i = max(0, min(e_i, len(s_smooth) - 1))
        if e_i < s_i:
            s_i, e_i = e_i, s_i
        # per-bridge tuning lives in yaml (single source of truth).
        # Function-level pre_m/post_m/force_center/chi_max_rad serve
        # as fallbacks when a particular bridge entry omits a field
        # (e.g. legacy yamls or hand-edits). All four are looked up
        # independently so a partial bridge dict still works.
        b_pre  = float(b.get('pre_m',        pre_m_f))
        b_post = float(b.get('post_m',       post_m_f))
        b_fc   = bool (b.get('force_center', force_center))
        b_chi  = float(b.get('chi_max_rad',  chi_max_rad))
        # skip a bridge whose tuning is fully disabled — same logic
        # as the early-exit at the start of this function, applied
        # per-bridge so a yaml mixing armed and disarmed bridges is
        # handled correctly.
        if (not b_fc) and b_chi < 0.0:
            continue
        core_start = float(s_smooth[s_i])
        core_end   = float(s_smooth[e_i])
        ext_start  = core_start - b_pre
        ext_end    = core_end   + b_post
        for k in range(n_grid):
            sk = float(s_grid[k])
            # place sk into the bridge's local s frame for closed-track
            # wrap-around. If the extended zone crosses s=0 / s=s_total,
            # shift sk by ±s_total to land it inside [ext_start, ext_end].
            sk_local = sk
            in_extended = (ext_start <= sk_local <= ext_end)
            if (not in_extended) and closed_loop:
                if ext_start < 0.0 and sk_local >= ext_start + s_total:
                    sk_local -= s_total
                    in_extended = (ext_start <= sk_local <= ext_end)
                elif ext_end > s_total and sk_local <= ext_end - s_total:
                    sk_local += s_total
                    in_extended = (ext_start <= sk_local <= ext_end)
            if not in_extended:
                continue
            r = _bridge_ramp(sk_local, core_start, core_end,
                             b_pre, b_post)
            cur = table[k]
            if cur is None:
                cur = {'force_center': bool(b_fc),
                       'chi_max_rad': float(b_chi),
                       'ramp': float(r)}
                table[k] = cur
            else:
                # overlapping bridges merge tightest: max ramp +
                # force_center OR-merge + chi_max picks the smaller
                # non-negative value (tighter clamp).
                if r > cur['ramp']:
                    cur['ramp'] = float(r)
                if b_fc:
                    cur['force_center'] = True
                if b_chi >= 0.0:
                    if cur['chi_max_rad'] < 0.0 or b_chi < cur['chi_max_rad']:
                        cur['chi_max_rad'] = float(b_chi)
        n_used += 1

    if n_used > 0:
        n_active = sum(1 for v in table if v is not None)
        n_core   = sum(1 for v in table if v is not None and v['ramp'] >= 0.999)
        print(f'[gen_global_racing_line] bridge constraints active on '
              f'{n_active}/{n_grid} grid points across {n_used} bridge(s); '
              f'core(ramp=1)={n_core}, ramp pts={n_active - n_core}; '
              f'CLI fallbacks: pre={pre_m:.2f}m, post={post_m:.2f}m, '
              f'force_center={force_center}, chi_max_rad={chi_max_rad:.3f}')
    return table


# per-sector safety_distance adapter — friction_sector_server pattern.
# Reads the live /safety_sector_params/... rosparam tree (populated by
# safety_sector_server.py from <map>/safety_sectors.yaml + rqt sliders)
# and returns a per-grid-step numpy array d_safe[idx] of length
# track_handler.s.size. _state_bounds() uses d_safe[idx] verbatim in
# place of the scalar safety_distance.
#
#      Rosparam tree (set by safety_sector_server.py):
#        /safety_sector_params/safety_distance_default  -> float
#        /safety_sector_params/n_sectors                -> int
#        /safety_sector_params/Sector{i}/{start,end,safety_distance,pre_m,post_m}
#
#      Sector start/end are smoothed-CSV row indices (closed interval). The
#      sector partition produced by safety_sector_slicing.py covers the full
#      track ([0, b1] [b1+1, b2] ... [b_{k-1}+1, N-1]) but this function does
#      not depend on that — disjoint or partial sector lists also work
#      (non-sector grid points keep the default).
#
#      Behavior:
#        - rosparam tree missing / n_sectors=0  ⇒  d_safe ≡ legacy_default
#          (= the scalar safety_distance from rqt). Legacy behavior preserved
#          bit-for-bit when no safety server is running.
#        - inside a sector core [s_start, s_end]: d_safe = sector value
#        - pre/post ramps: linear blend default ↔ sector value over
#          pre_m/post_m (in meters of s). Bridge ramp logic mirrored.
#        - overlapping sectors take the *larger* d_safe (more conservative).
#        - closed-loop wrap-around handled the same way as bridges.
# DO NOT use 'safety_distance' as a dict key in the helpers below.
# run_pipeline.sh injects raceline params with an anchorless regex
# r"'safety_distance'\s*:.*" that would silently clobber any matching
# line at exec-time (params dict line 53 is the only intended target).
# Use 'd_safe' for the per-sector value instead.
def _read_safety_sectors_rosparam(legacy_default):
    """Return (default_d, [sector_dict, ...]) from /safety_sector_params/...
    when the dyn cfg server is up. Returns (None, None) when the rosparam
    tree is missing — caller should fall back to yaml-on-disk.
    """
    try:
        import rospy as _rospy
    except Exception:
        return None, None
    try:
        n_sectors = int(_rospy.get_param(
            '/safety_sector_params/n_sectors', -1))
    except Exception:
        return None, None
    if n_sectors < 0:
        return None, None        # rosparam tree absent
    default_d = float(_rospy.get_param(
        '/safety_sector_params/safety_distance_default', legacy_default))
    sectors = []
    for i in range(n_sectors):
        try:
            sectors.append({
                'start': int(_rospy.get_param(
                    f'/safety_sector_params/Sector{i}/start')),
                'end': int(_rospy.get_param(
                    f'/safety_sector_params/Sector{i}/end')),
                'd_safe': float(_rospy.get_param(
                    f'/safety_sector_params/Sector{i}/safety_distance',
                    default_d)),
                'pre_m': float(_rospy.get_param(
                    f'/safety_sector_params/Sector{i}/pre_m', 0.0)),
                'post_m': float(_rospy.get_param(
                    f'/safety_sector_params/Sector{i}/post_m', 0.0)),
            })
        except Exception as e:
            print(f'[gen_global_racing_line] rosparam Sector{i} '
                  f'missing fields ({e}) — skipped')
    return default_d, sectors


def _read_safety_sectors_yaml(yaml_path, legacy_default):
    """Fallback path — read <map>/safety_sectors.yaml directly when the
    safety dyn cfg server is not running but use=True. Returns
    (default_d, [sector_dict, ...]) or (None, None) on failure.
    """
    if not os.path.exists(yaml_path):
        return None, None
    try:
        with open(yaml_path, 'r') as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        print(f'[gen_global_racing_line] yaml parse failed '
              f'({yaml_path}): {e}')
        return None, None
    default_d = float(doc.get('safety_distance_default', legacy_default))
    n_sectors = int(doc.get('n_sectors', 0))
    sectors = []
    for i in range(n_sectors):
        s = doc.get(f'Sector{i}')
        if not isinstance(s, dict):
            continue
        try:
            sectors.append({
                'start': int(s['start']),
                'end':   int(s['end']),
                'd_safe': float(s.get('safety_distance', default_d)),
                'pre_m':  float(s.get('pre_m',  0.0)),
                'post_m': float(s.get('post_m', 0.0)),
            })
        except Exception as e:
            print(f'[gen_global_racing_line] yaml Sector{i} malformed '
                  f'({e}) — skipped')
    return default_d, sectors


def _build_safety_table(track_handler, smoothed_csv_path, legacy_default):
    n_grid = int(track_handler.s.size)
    d_safe = np.full(n_grid, float(legacy_default), dtype=float)

    # master switch via rosparam set by gg_tuner_node right before the
    # raceline launch (Safety group's `safety_sector_use` checkbox).
    # False ⇒ legacy uniform margin regardless of sector data source.
    try:
        import rospy as _rospy
        use = bool(_rospy.get_param('/safety_sector_params/use', False))
    except Exception as e:
        print(f'[gen_global_racing_line] rospy.get_param unavailable '
              f'({e}); safety sectors ignored, using scalar safety_distance')
        return d_safe
    if not use:
        print(f'[gen_global_racing_line] safety_sector_use=False '
              f'— uniform safety_distance={legacy_default:.3f}m')
        return d_safe

    # two sources, rosparam first then yaml-on-disk.
    # Setting=ON  ⇒ rosparam tree exists (live slider values, may
    # differ from yaml if save_params not clicked).
    # Setting=OFF ⇒ rosparam tree wiped on server kill; fall through
    # to <map>/safety_sectors.yaml (last saved state).
    default_d, sectors = _read_safety_sectors_rosparam(legacy_default)
    source = 'rosparam'
    if sectors is None:
        yaml_path = os.path.join(
            os.path.dirname(smoothed_csv_path), 'safety_sectors.yaml')
        default_d, sectors = _read_safety_sectors_yaml(
            yaml_path, legacy_default)
        source = f'yaml({os.path.basename(yaml_path)})'
    if not sectors:
        print(f'[gen_global_racing_line] no safety sectors found '
              f'(use=True but neither rosparam nor yaml has data) — '
              f'uniform safety_distance={legacy_default:.3f}m')
        return d_safe

    d_safe[:] = default_d

    if not os.path.exists(smoothed_csv_path):
        print(f'[gen_global_racing_line] smoothed CSV not found '
              f'({smoothed_csv_path}) — safety sectors ignored')
        return d_safe
    try:
        df_smooth = pd.read_csv(smoothed_csv_path)
        s_smooth = df_smooth['s_m'].to_numpy()
    except Exception as e:
        print(f'[gen_global_racing_line] failed to read s_m from '
              f'{smoothed_csv_path}: {e} — safety sectors ignored')
        return d_safe
    s_total = float(s_smooth[-1] - s_smooth[0]) if len(s_smooth) > 1 else 0.0
    closed_loop = s_total > 0.0
    s_grid = track_handler.s
    n_used = 0
    for s in sectors:
        s_i = max(0, min(int(s['start']), len(s_smooth) - 1))
        e_i = max(0, min(int(s['end']),   len(s_smooth) - 1))
        if e_i < s_i:
            s_i, e_i = e_i, s_i
        d_sec = float(s['d_safe'])
        pre_m = float(s['pre_m'])
        post_m = float(s['post_m'])
        core_start = float(s_smooth[s_i])
        core_end   = float(s_smooth[e_i])
        ext_start  = core_start - pre_m
        ext_end    = core_end   + post_m
        for k in range(n_grid):
            sk_local = float(s_grid[k])
            in_ext = (ext_start <= sk_local <= ext_end)
            if (not in_ext) and closed_loop:
                if ext_start < 0.0 and sk_local >= ext_start + s_total:
                    sk_local -= s_total
                    in_ext = (ext_start <= sk_local <= ext_end)
                elif ext_end > s_total and sk_local <= ext_end - s_total:
                    sk_local += s_total
                    in_ext = (ext_start <= sk_local <= ext_end)
            if not in_ext:
                continue
            if core_start <= sk_local <= core_end:
                r = 1.0
            elif (core_start - pre_m) <= sk_local < core_start and pre_m > 0.0:
                r = (sk_local - (core_start - pre_m)) / pre_m
            elif core_end < sk_local <= (core_end + post_m) and post_m > 0.0:
                r = 1.0 - (sk_local - core_end) / post_m
            else:
                r = 1.0
            blended = (1.0 - r) * default_d + r * d_sec
            if blended > d_safe[k]:
                d_safe[k] = blended
        n_used += 1

    n_changed = int(np.sum(np.abs(d_safe - default_d) > 1e-9))
    print(f'[gen_global_racing_line] safety sectors active ({source}): '
          f'{n_changed}/{n_grid} grid points across {n_used} sector(s); '
          f'default={default_d:.3f}m, min={d_safe.min():.3f}m, '
          f'max={d_safe.max():.3f}m')
    return d_safe


def calc_global_raceline(
        track_name: str,
        vehicle_params: dict,
        gg_mode: str,
        gg_margin: float,
        safety_distance: float,
        w_T: float,
        w_jx: float,
        w_jy: float,
        w_dOmega_z: float,
        RK4_steps: int,
        V_guess: float,
        # V_min enforced as NLP lower bound on velocity state
        V_min: float,
        # bridge mu scale (forwarded to Track3D)
        g_weight: float,
        # bridge centerline / heading-lock — optional, off ⇒ no-op.
        # Tuning knobs are GLOBAL (applied to every bridge in yaml).
        bridge_constraints_enable: bool,
        bridge_yaml_dir: str,
        bridge_pre_m: float,
        bridge_post_m: float,
        bridge_force_center: bool,
        bridge_chi_max_rad: float,
        # per-sector safety_distance is consumed via the live rosparam
        # tree at /safety_sector_params/... (populated by
        # safety_sector_server.py). No function arg needed.
        neglect_w_omega_x: bool,
        neglect_w_omega_y: bool,
        neglect_euler: bool,
        neglect_centrifugal: bool,
        neglect_w_dot: bool,
        neglect_V_omega: bool,
        sol_opt: dict,
        out_path: str,
        step_size_opt: float = 0.2,
):
    track_handler = Track3D(
        path=os.path.join(track_path, track_name),
        # bake bridge mu scale into Track3D
        g_weight=g_weight,
    )

    # resample to optimization grid (coarser than smooth for NLP performance)
    track_handler.resample(step_size_opt)

    # load optional bridge sector yaml and build a per-NLP-step bound
    # override table. Yaml has positions only; tuning knobs are
    # globals forwarded from rqt. Missing yaml or both knobs at "off"
    # ⇒ override table stays all-None ⇒ bound expressions reduce to
    # the legacy formula (no behavioral change for tracks without
    # bridges).
    bridge_state = _build_bridge_state_table(
        track_handler=track_handler,
        smoothed_csv_path=os.path.join(track_path, track_name),
        bridge_yaml_dir=bridge_yaml_dir,
        enable=bridge_constraints_enable,
        pre_m=bridge_pre_m,
        post_m=bridge_post_m,
        force_center=bridge_force_center,
        chi_max_rad=bridge_chi_max_rad,
    )

    # build per-NLP-step d_safe[idx] from /safety_sector_params/...
    # rosparam tree. Tree missing / n_sectors=0 ⇒ d_safe is uniformly
    # the scalar `safety_distance` so legacy behavior is preserved
    # bit-for-bit when no safety server is running.
    d_safe_table = _build_safety_table(
        track_handler=track_handler,
        smoothed_csv_path=os.path.join(track_path, track_name),
        legacy_default=safety_distance,
    )

    gg_handler = GGManager(
        gg_path=gg_diagram_path,
        gg_margin=gg_margin
    )

    # Define state variables.
    # Velocity.
    V = ca.MX.sym('V')
    # Lateral displacement.
    n = ca.MX.sym('n')

    # Relative orientation w.r.t. reference line.
    chi = ca.MX.sym('chi')

    # Longitudinal acceleration.
    ax = ca.MX.sym('ax')

    # Lateral acceleration.
    ay = ca.MX.sym('ay')

    # Create state vector.
    x = ca.vertcat(V, n, chi, ax, ay)
    nx = x.shape[0]

    # Define control variables.
    # Longitudinal jerk.
    jx = ca.MX.sym('jx')

    # Lateral acceleration.
    jy = ca.MX.sym('jy')

    # Create control vector.
    u = ca.vertcat(jx, jy)
    nu = u.shape[0]

    s = ca.MX.sym('s')

    # Time-distance scaling factor (dt/ds).
    s_dot = (V * ca.cos(chi)) / (1.0 - n * track_handler.Omega_z_interpolator(s))
    # vertical velocity
    w = n * track_handler.Omega_x_interpolator(s) * s_dot

    # Differential equations for scaled point mass model.
    dV = 1.0 / s_dot * ax
    if not neglect_w_omega_y:
        dV += w * (track_handler.Omega_x_interpolator(s) * ca.sin(chi) - track_handler.Omega_y_interpolator(s) * ca.cos(chi))

    dn = 1.0 / s_dot * V * ca.sin(chi)

    dchi = 1.0 / s_dot * ay / V - track_handler.Omega_z_interpolator(s)
    if not neglect_w_omega_x:
        dchi += w * (track_handler.Omega_x_interpolator(s) * ca.cos(chi) + track_handler.Omega_y_interpolator(s) * ca.sin(chi)) / V

    dax = jx / s_dot
    day = jy / s_dot

    dOmega_z = track_handler.Omega_z_interpolator(s) + dchi  # curvature in road plane

    # Create ODE vector.
    dx = ca.vertcat(dV, dn, dchi, dax, day)

    # Objective function.
    L_t = w_T * 1.0 / s_dot
    L_reg = w_jx * (jx / s_dot) ** 2 + w_jy * (jy / s_dot) ** 2 + w_dOmega_z * dOmega_z ** 2

    # Discrete time dynamics using fixed step Runge-Kutta 4 integrator.
    M = RK4_steps  # RK4 steps per interval
    ds_rk = track_handler.ds / M  # Step size used for RK integration
    f = ca.Function('f', [x, u, s], [dx, L_t, L_reg])  # Function returning derivative and cost function
    X0 = ca.MX.sym('X0', nx)  # Input state.
    U = ca.MX.sym('U', nu)  # Input control.
    S0 = ca.MX.sym('S0')  # Input longitudinal position.
    X = X0  # Integrated derivative.
    S = S0  # Integrated longitudinal position.
    Q_t = 0  # Integrated cost function (time).
    Q_reg = 0  # Integrated cost function (regularization).
    for j in range(M):
        k1, k1_q_t, k1_q_reg = f(X, U, S)
        k2, k2_q_t, k2_q_reg = f(X + ds_rk / 2 * k1, U, S + ds_rk / 2)
        k3, k3_q_t, k3_q_reg = f(X + ds_rk / 2 * k2, U, S + ds_rk / 2)
        k4, k4_q_t, k4_q_reg = f(X + ds_rk * k3, U, S + ds_rk)
        X = X + ds_rk / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q_t = Q_t + ds_rk / 6 * (k1_q_t + 2 * k2_q_t + 2 * k3_q_t + k4_q_t)
        Q_reg = Q_reg + ds_rk / 6 * (k1_q_reg + 2 * k2_q_reg + 2 * k3_q_reg + k4_q_reg)
        S = S + ds_rk
    # Function to integrate derivative and costs.
    F = ca.Function('F', [X0, U, S0], [X, Q_t, Q_reg], ['x0', 'u', 's0'], ['xf', 'q_t', 'q_reg'])

    # Empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J_t = 0.0
    J_reg = 0.0
    g = []
    lbg = []
    ubg = []

    # factor state-bound construction so the same logic applies at the
    # initial step (k=0) and at every interior k+1 step below. Returns
    # (lb_list, ub_list, w0_list) of length nx. bridge_state[idx] is
    # None outside bridge zones (legacy bounds) or a dict with
    # force_center/chi_max_rad/half_width inside.
    def _state_bounds(idx):
        # per-sector safety_distance — d_safe_table[idx] equals the
        # scalar `safety_distance` everywhere when no safety yaml is
        # present (legacy behavior).
        d_safe = (float(d_safe_table[idx])
                  if idx < len(d_safe_table) else float(safety_distance))
        n_lb = (track_handler.w_tr_right[idx]
                + vehicle_params['total_width'] / 2.0 + d_safe)
        n_ub = (track_handler.w_tr_left[idx]
                - vehicle_params['total_width'] / 2.0 - d_safe)
        n_init = (track_handler.w_tr_left[idx]
                  + track_handler.w_tr_right[idx]) / 2.0
        chi_lb, chi_ub = -np.pi / 2.0, np.pi / 2.0
        bs = bridge_state[idx] if idx < len(bridge_state) else None
        if bs is not None:
            if bs.get('force_center', False):
                # pin n ≈ 0 (NLP reference frame center). NLP n=0 is
                # the smoothed reference path, which is *very close*
                # to (but not exactly) the geometric boundary midpoint
                # — typical residual offset is < 1 cm. Tried option B
                # (pin to (w_tr_l + w_tr_r)/2) but the moving target
                # curve made the NLP stiff and hit max-iter, so we
                # kept the cleaner n=0 pin. Tiny ±eps avoids solver
                # discomfort with hard equality bounds. Safety
                # distance is automatically satisfied — reference
                # path sits strictly inside [n_lb_legacy, n_ub_legacy].
                # With ramp r ∈ [0,1] the bound interpolates between
                # legacy (r=0) and tight ±eps (r=1) over pre_m/post_m.
                # eps ≥ NLP constraint tolerance (1e-4) by a safe
                # margin. With eps=1e-3 the solver hit the bound
                # every step and oscillated visibly (mm-scale wobble
                # consistent with the box being thinner than its
                # tolerance). 5e-3 (=±5 mm) is still well within
                # "centered" for a 1.8 m wide corridor (<0.3% of
                # half-corridor) but gives the solver room to find a
                # smooth interior path.
                eps = 5e-3
                r = float(bs.get('ramp', 1.0))
                n_lb = (1.0 - r) * n_lb + r * max(n_lb, -eps)
                n_ub = (1.0 - r) * n_ub + r * min(n_ub,  eps)
                n_init = (1.0 - r) * n_init  # r=1 ⇒ 0
            # force_center implies chi=0 (physically: "pinned to
            # centerline AND parallel to track"). Leaving chi free
            # while n is pinned makes the solver oscillate chi to
            # satisfy dynamics with bounded n — visible mm-scale
            # wobble. So whenever force_center is on we tighten
            # chi_max to 0 (taking the tighter of the user setting
            # and 0).
            chi_max = bs.get('chi_max_rad', -1.0)
            if bs.get('force_center', False):
                chi_max = 0.0 if chi_max < 0.0 else min(chi_max, 0.0)
            if chi_max >= 0.0:
                # chi clamp also ramps from legacy (r=0) to tight (r=1).
                r = float(bs.get('ramp', 1.0))
                chi_lb_target = max(chi_lb, -chi_max)
                chi_ub_target = min(chi_ub,  chi_max)
                chi_lb = (1.0 - r) * chi_lb + r * chi_lb_target
                chi_ub = (1.0 - r) * chi_ub + r * chi_ub_target
        return (
            [V_min,             n_lb, chi_lb, -np.inf, -np.inf],
            [gg_handler.V_max,  n_ub, chi_ub,  np.inf,  np.inf],
            [V_guess,           n_init, 0.0, 1e-6, 1e-6],
        )

    # Initial conditions
    Xk = ca.MX.sym('X0', nx)
    w += [Xk]
    # delegate to shared _state_bounds
    _lb0, _ub0, _w0_0 = _state_bounds(0)
    lbw += _lb0
    ubw += _ub0
    w0 += _w0_0

    # Formulate the NLP
    for k in range(0, track_handler.s.size):
        # Apparent accelerations
        ax_tilde, ay_tilde, g_tilde = track_handler.calc_apparent_accelerations(
            V=Xk[0],
            n=Xk[1],
            chi=Xk[2],
            ax=Xk[3],
            ay=Xk[4],
            s=k * track_handler.ds,
            h=vehicle_params['h'],
            neglect_w_omega_y=neglect_w_omega_y,
            neglect_w_omega_x=neglect_w_omega_x,
            neglect_euler=neglect_euler,
            neglect_centrifugal=neglect_centrifugal,
            neglect_w_dot=neglect_w_dot,
            neglect_V_omega=neglect_V_omega
        )

        # Constraints of gggv-Diagram.
        if gg_mode == 'polar':
            alpha = ca.arctan2(ax_tilde, ay_tilde)
            rho = ca.sqrt(ax_tilde ** 2 + ay_tilde ** 2)
            adherence_radius = gg_handler.gggv_interpolator(ca.vertcat(Xk[0], g_tilde, alpha))
            g += [adherence_radius - rho]
            lbg += [0.0]
            ubg += [np.inf]
        elif gg_mode == 'diamond':
            gg_exponent, ax_min, ax_max, ay_max = ca.vertsplit(gg_handler.acc_interpolator(ca.vertcat(Xk[0], g_tilde)))  # positive

            g += [ay_max - ca.fabs(ay_tilde)]
            lbg += [0.0]
            ubg += [np.inf]

            g += [ca.fabs(ax_min) * ca.power(
                    ca.fmax(
                        (1.0 - ca.power(
                            ca.fmin(ca.fabs(ay_tilde) / ay_max, 1.0),
                            gg_exponent
                        ))
                        , 1e-3
                    ),
                    1.0 / gg_exponent
                ) - ca.fabs(ax_tilde)
            ]
            lbg += [0.0]
            ubg += [np.inf]

            g += [ax_max - ax_tilde]
            lbg += [0.0]
            ubg += [np.inf]
        else:
            raise RuntimeError('Unknown gg_mode.')

        # If last iteration, don't add new control or states.
        if k == track_handler.s.size - 1:
            break

        # New NLP variable for the control.
        Uk = ca.MX.sym('U_' + str(k), nu)
        w += [Uk]
        lbw += [-np.inf] * nu
        ubw += [np.inf] * nu
        w0 += [0.0] * nu

        # Integrate till the end of the interval.
        Fk = F(x0=Xk, u=Uk, s0=k * track_handler.ds)
        Xk_end = Fk['xf']
        J_t = J_t + Fk['q_t']
        J_reg = J_reg + Fk['q_reg']

        # New NLP variable for state at end of interval.
        Xk = ca.MX.sym('X_' + str(k + 1), nx)
        w += [Xk]
        # delegate to shared _state_bounds. Applies bridge force_center / chi_max overrides at
        # grid index k+1 when bridge_state[k+1] is populated.
        _lb, _ub, _w0_k = _state_bounds(k + 1)
        lbw += _lb
        ubw += _ub
        w0 += _w0_k

        # Add equality constraint for continuity.
        g += [Xk_end - Xk]
        lbg += [0.0] * nx
        ubg += [0.0] * nx

    # Boundary constraint: start states = final states.
    g += [w[0] - Xk]
    lbg += [0.0] * nx
    ubg += [0.0] * nx

    # Concatenate NLP vectors.
    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    w0 = ca.vertcat(w0)
    lbw = ca.vertcat(lbw)
    ubw = ca.vertcat(ubw)
    lbg = ca.vertcat(lbg)
    ubg = ca.vertcat(ubg)

    # Create an NLP solver.
    nlp = {'f': J_t + J_reg, 'x': w, 'g': g}
    solver = ca.nlpsol('solver', 'ipopt', nlp, sol_opt)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    # Extract lap time.
    laptime = ca.Function('f_laptime', [w], [J_t])
    print(f'Laptime: {float(laptime(sol["x"]))}')
    # Extract regularization costs.
    regularization = ca.Function('f_regularization', [w], [J_reg])

    # Extract solution.
    w_opt = sol['x'].full().flatten()
    s_opt = track_handler.s
    v_opt = np.array(w_opt[0::nx + nu])
    n_opt = np.array(w_opt[1::nx + nu])
    chi_opt = np.array(w_opt[2::nx + nu])
    ax_opt = np.array(w_opt[3::nx + nu])
    ay_opt = np.array(w_opt[4::nx + nu])
    jx_opt = np.array(w_opt[5::nx + nu])
    jy_opt = np.array(w_opt[6::nx + nu])

    trajectory_data_frame = pd.DataFrame()
    trajectory_data_frame['s_opt'] = s_opt
    trajectory_data_frame['v_opt'] = v_opt
    trajectory_data_frame['n_opt'] = n_opt
    trajectory_data_frame['chi_opt'] = chi_opt
    trajectory_data_frame['ax_opt'] = ax_opt
    trajectory_data_frame['ay_opt'] = ay_opt
    trajectory_data_frame['jx_opt'] = np.concatenate([jx_opt, [0]])
    trajectory_data_frame['jy_opt'] = np.concatenate([jy_opt, [0]])
    trajectory_data_frame['laptime'] = float(laptime(sol["x"]))

    # Save solution.
    if out_path:
        trajectory_data_frame.to_csv(path_or_buf=out_path, sep=',', index=True, float_format='%.6f')

    return trajectory_data_frame

if __name__ == '__main__':
    os.makedirs(os.path.join(raceline_out_path), exist_ok=True)
    raceline = calc_global_raceline(
            track_name=params['track_name'],
            vehicle_params=params['vehicle_params'],
            gg_mode=params['gg_mode'],
            gg_margin=params['gg_margin'],
            safety_distance=params['safety_distance'],
            w_T=params['w_T'],
            w_jx=params['w_jx'],
            w_jy=params['w_jy'],
            w_dOmega_z=params['w_dOmega_z'],
            RK4_steps=params['RK4_steps'],
            V_guess=params['V_guess'],
            # pass V_min from params dict to NLP
            V_min=params['V_min'],
            # bridge mu scale baked into raceline
            g_weight=params['g_weight'],
            # bridge centerline / heading-lock — optional, off ⇒ no-op.
            # Tuning knobs come from rqt via run_pipeline.sh args.
            bridge_constraints_enable=params['bridge_constraints_enable'],
            bridge_yaml_dir=params['bridge_yaml_dir'],
            bridge_pre_m=params['bridge_pre_m'],
            bridge_post_m=params['bridge_post_m'],
            bridge_force_center=params['bridge_force_center'],
            bridge_chi_max_rad=params['bridge_chi_max_rad'],
            # per-sector safety_distance is read from /safety_sector_params/...
            # rosparam tree at NLP setup time — no arg here.
            neglect_w_omega_x=params['neglect_w_omega_x'],
            neglect_w_omega_y=params['neglect_w_omega_y'],
            neglect_euler=params['neglect_euler'],
            neglect_centrifugal=params['neglect_centrifugal'],
            neglect_w_dot=params['neglect_w_dot'],
            neglect_V_omega=params['neglect_V_omega'],
            sol_opt=params['sol_opts'],
            out_path=os.path.join(raceline_out_path, params['raceline_name']),
            step_size_opt=params['step_size_opt'],
    )
