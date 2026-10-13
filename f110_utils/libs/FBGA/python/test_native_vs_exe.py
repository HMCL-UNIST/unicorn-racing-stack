#!/usr/bin/env python3
# Numerical match test between GIGI_test_unicorn.exe (CSV
# round-trip) and the fbga_native pybind11 module. Runs the same synthetic
# input through both paths and asserts (v, ax) match within tolerance.
#
# Usage (inside Docker container):
#   python3 test_native_vs_exe.py
#
# This test does NOT depend on ROS — it's a pure FBGA library smoke test.

import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

# -- Paths (assumes this script lives under <race_stack>/f110_utils/libs/FBGA/python/) --
HERE = os.path.dirname(os.path.abspath(__file__))
FBGA_ROOT = os.path.dirname(HERE)
BIN_DIR = os.path.join(FBGA_ROOT, 'bin')
EXE = os.path.join(BIN_DIR, 'GIGI_test_unicorn.exe')

# Default params + gg paths used by FBGAStateMachine (see fbga_state_machine_node.py)
RACE_STACK = os.path.abspath(os.path.join(FBGA_ROOT, '..', '..', '..'))
GG_BIN_DEFAULT = os.path.join(
    RACE_STACK, 'planner', '3d_gb_optimizer', 'global_line', 'data',
    'gg_diagrams', 'rc_car_10th_latest', 'velocity_frame', 'gg.bin')
PARAMS_YML_DEFAULT = os.path.join(
    RACE_STACK, 'planner', '3d_gb_optimizer', 'global_line', 'data',
    'vehicle_params', 'params_rc_car_10th_latest.yml')


def make_params_txt(yml_path: str, out_txt: str):
    """Mirror fbga_runner._convert_params_yml — 5 scalars only."""
    import yaml
    with open(yml_path) as f:
        cfg = yaml.safe_load(f)
    vp = cfg['vehicle_params']
    tp = cfg['tire_params']
    with open(out_txt, 'w') as f:
        f.write(f"m={vp['m']}\n")
        f.write(f"P_max={vp['P_max']}\n")
        f.write(f"mu_x={tp['p_Dx_1']}\n")
        f.write(f"mu_y={tp['p_Dy_1']}\n")
        f.write(f"v_max={vp['v_max']}\n")


def synth_input(n=80, ds=0.1, kind='straight'):
    """Build a synthetic (s, kappa, g_tilde, mu, dmu_ds) input."""
    s = np.arange(n) * ds
    if kind == 'straight':
        kappa = np.zeros(n)
    elif kind == 'curve':
        kappa = 0.3 * np.sin(2 * np.pi * np.arange(n) / n)
    elif kind == 'chicane':
        kappa = 0.5 * np.sin(4 * np.pi * np.arange(n) / n)
    else:
        raise ValueError(kind)
    g_tilde = np.full(n, 9.806)
    mu = np.zeros(n)
    dmu_ds = np.zeros(n)
    return s, kappa, g_tilde, mu, dmu_ds


def run_exe(s, kappa, g_tilde, mu, dmu_ds, v0, slope_corr,
            params_txt, gg_bin):
    """Invoke the existing CLI exe via CSV. Returns (v, ax) arrays."""
    input_csv  = tempfile.NamedTemporaryFile(prefix='fbga_in_',  suffix='.csv', delete=False).name
    output_csv = tempfile.NamedTemporaryFile(prefix='fbga_out_', suffix='.csv', delete=False).name
    try:
        with open(input_csv, 'w') as f:
            f.write('s,kappa,g_tilde,mu,dmu_ds\n')
            for i in range(len(s)):
                f.write(f'{s[i]:.10f},{kappa[i]:.10f},{g_tilde[i]:.10f},'
                        f'{mu[i]:.10f},{dmu_ds[i]:.10f}\n')
        cmd = [EXE, '--model', 'lookup',
               '--input', input_csv,
               '--params', params_txt,
               '--gg', gg_bin,
               '--output', output_csv,
               '--v0', f'{v0:.6f}',
               '--slope-corr', f'{slope_corr:.6f}']
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
        v, ax = [], []
        with open(output_csv) as f:
            for line in f:
                if line.startswith('#') or line.startswith('s,'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    v.append(float(parts[1]))
                    ax.append(float(parts[2]))
        return np.array(v), np.array(ax)
    finally:
        for p in (input_csv, output_csv):
            try: os.remove(p)
            except OSError: pass


def main():
    print(f"FBGA root      : {FBGA_ROOT}")
    print(f"exe            : {EXE}")
    print(f"gg.bin         : {GG_BIN_DEFAULT}")
    print(f"params yml     : {PARAMS_YML_DEFAULT}")

    if not os.path.exists(EXE):
        print(f"FAIL: exe missing ({EXE})")
        sys.exit(1)
    if not os.path.exists(GG_BIN_DEFAULT):
        print(f"FAIL: gg.bin missing ({GG_BIN_DEFAULT})")
        sys.exit(1)
    if not os.path.exists(PARAMS_YML_DEFAULT):
        print(f"FAIL: params.yml missing ({PARAMS_YML_DEFAULT})")
        sys.exit(1)

    # Make sure native module is importable from BIN_DIR.
    if BIN_DIR not in sys.path:
        sys.path.insert(0, BIN_DIR)
    import fbga_native
    print(f"fbga_native    : {fbga_native.__file__}")

    # Build params.txt the same way fbga_runner does.
    params_txt = os.path.join(tempfile.gettempdir(), 'fbga_test_params.txt')
    make_params_txt(PARAMS_YML_DEFAULT, params_txt)

    # Instantiate native once.
    native = fbga_native.FBGANative(params_path=params_txt, gg_path=GG_BIN_DEFAULT)
    print(f"native.v_max   : {native.v_max():.3f}")
    print(f"native.n_v/n_g/n_s = {native.n_v()}/{native.n_g()}/{native.n_s()}")
    print()

    atol = 1e-6   # FWBW evalV uses internal spline; CSV round-trip costs ~1e-10
    rtol = 1e-9
    overall_ok = True
    cases = [
        ('straight, v0=1, slope=1.0',  dict(kind='straight', v0=1.0, slope_corr=1.0)),
        ('curve,    v0=1, slope=1.0',  dict(kind='curve',    v0=1.0, slope_corr=1.0)),
        ('chicane,  v0=2, slope=1.0',  dict(kind='chicane',  v0=2.0, slope_corr=1.0)),
        ('chicane,  v0=2, slope=1.5',  dict(kind='chicane',  v0=2.0, slope_corr=1.5)),
        ('straight, v0=3, slope=0.8',  dict(kind='straight', v0=3.0, slope_corr=0.8)),
    ]
    for label, opt in cases:
        s, kappa, g_tilde, mu, dmu_ds = synth_input(n=80, ds=0.1, kind=opt['kind'])
        t0 = time.perf_counter()
        v_exe, ax_exe = run_exe(s, kappa, g_tilde, mu, dmu_ds,
                                opt['v0'], opt['slope_corr'],
                                params_txt, GG_BIN_DEFAULT)
        t1 = time.perf_counter()
        v_nat, ax_nat = native.solve(s, kappa, g_tilde, mu, dmu_ds,
                                     v0=opt['v0'], slope_corr=opt['slope_corr'])
        t2 = time.perf_counter()
        exe_ms = (t1 - t0) * 1000.0
        nat_ms = (t2 - t1) * 1000.0
        max_dv  = float(np.max(np.abs(v_exe  - v_nat)))
        max_dax = float(np.max(np.abs(ax_exe - ax_nat)))
        ok = (max_dv <= atol) and (max_dax <= atol)
        overall_ok = overall_ok and ok
        status = 'PASS' if ok else 'FAIL'
        print(f"[{status}] {label}")
        print(f"    exe   {exe_ms:6.1f} ms  v=[{v_exe.min():.4f},{v_exe.max():.4f}]")
        print(f"    nat   {nat_ms:6.1f} ms  v=[{v_nat.min():.4f},{v_nat.max():.4f}]   speedup x{exe_ms/max(nat_ms,1e-3):.1f}")
        print(f"    Δv max={max_dv:.3e}  Δax max={max_dax:.3e}  (atol={atol})")
        print()

    # Hot-loop timing: how fast is native when warm?
    print("=== Hot loop (100 solves, chicane, n=80) ===")
    s, kappa, g_tilde, mu, dmu_ds = synth_input(n=80, ds=0.1, kind='chicane')
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        native.solve(s, kappa, g_tilde, mu, dmu_ds, v0=2.0, slope_corr=1.0)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    print(f"native solve_ms  p50={np.median(times):.2f}  p95={np.percentile(times,95):.2f}  "
          f"max={times.max():.2f}  min={times.min():.2f}")

    print()
    print("OVERALL:", "PASS" if overall_ok else "FAIL")
    sys.exit(0 if overall_ok else 2)


if __name__ == '__main__':
    main()
