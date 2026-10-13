# FBGA C++ runner, with a pybind11 native module path (fbga_native).
#   When fbga_native.so is built (libs/FBGA/bin/), run_once skips the
#   subprocess+CSV round-trip and calls FWBW in-process. params/gg.bin are
#   loaded ONCE in FBGANative ctor instead of every solve.
#   Env var FBGA_DISABLE_NATIVE=1 forces the legacy subprocess path.
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import struct
import numpy as np
import yaml

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY_SG = True
except ImportError:
    HAS_SCIPY_SG = False


# Best-effort import of fbga_native.so. The .so lives next to
# the exe under libs/FBGA/bin/. We add that directory to sys.path so a plain
# `import fbga_native` works without site-packages install.
def _try_import_fbga_native(fbga_bin_path: str):
    """Return the fbga_native module or None. fbga_bin_path is the exe path;
    the .so is expected as <dirname(exe)>/fbga_native.cpython-*.so."""
    if os.environ.get('FBGA_DISABLE_NATIVE', '') == '1':
        return None
    so_dir = os.path.dirname(os.path.abspath(fbga_bin_path))
    if so_dir not in sys.path:
        sys.path.insert(0, so_dir)
    try:
        import fbga_native  # noqa: F401
        return fbga_native
    except Exception:
        return None


class FBGARunner:
    """Python wrapper around libs/FBGA/bin/GIGI_test_unicorn.exe.

    Mirrors fbga_velocity_planner.py so behavior stays consistent between
    the global-pass node and any local-pass caller. No rospy dependency
    (inject a logger if you want messages).
    """

    def __init__(self, fbga_bin, gg_bin, params_yml, *,
                 enable_mu=True, bridge_effect=1.0,
                 g_tilde_soft_beta=20.0,
                 smooth_mu=True,  mu_window=21,    mu_polyorder=3,
                 smooth_kappa=True, kappa_window=21, kappa_polyorder=3,
                 alpha=0.5, max_iter=50, tol=0.05,
                 slope_brake_margin=0.0, slope_brake_vmax=5.0,
                 gg_scale=1.0,
                 tag='latest',
                 logger=None):
        self._log = logger if logger is not None else (lambda *a, **k: None)

        # tag isolates the scaled gg.bin destination (and params.txt)
        # so multiple FBGARunner instances (per state_machine mode) don't
        # collide on the same disk file.
        self.tag = str(tag)

        self.fbga_bin       = fbga_bin
        self.gg_bin_source  = gg_bin          # original (untouched)
        self.params_yml     = params_yml
        self.gg_scale       = float(gg_scale)

        for name, path in [('fbga_bin', self.fbga_bin),
                           ('params_yml', self.params_yml)]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"FBGA {name} not found: {path}")

        if not os.path.exists(self.gg_bin_source):
            self._generate_gg_bin(self.gg_bin_source)

        # Materialise a scaled copy (ax_max, ay_max scaled by gg_scale)
        # under <vehicle>_fbga_<tag>/ alongside the source vehicle folder.
        self.gg_bin = self._materialize_scaled_gg(self.gg_bin_source, self.gg_scale)

        self.params_txt = os.path.join(tempfile.gettempdir(),
                                       f'fbga_params_{self.tag}.txt')
        self.v_max = 12.0
        self._convert_params_yml(self.params_yml)

        self.enable_mu             = bool(enable_mu)
        self.bridge_effect         = float(bridge_effect)
        self.g_tilde_soft_beta     = float(g_tilde_soft_beta)
        self.smooth_mu             = bool(smooth_mu) and HAS_SCIPY_SG
        self.mu_smooth_window      = int(mu_window) | 1
        self.mu_smooth_polyorder   = int(mu_polyorder)
        self.smooth_kappa          = bool(smooth_kappa) and HAS_SCIPY_SG
        self.kappa_smooth_window   = int(kappa_window) | 1
        self.kappa_smooth_polyorder = int(kappa_polyorder)
        self.alpha                 = float(alpha)
        self.max_iter              = int(max_iter)
        self.tol                   = float(tol)
        self.slope_brake_margin    = float(slope_brake_margin)
        self.slope_brake_vmax      = float(slope_brake_vmax)

        self.g_min, self.g_max = self._read_g_range()

        # Try the native pybind11 module. If available, run_once
        # calls native.solve() directly (no subprocess, no CSV). Otherwise
        # fall back silently to the exe path.
        self._native_mod = _try_import_fbga_native(self.fbga_bin)
        self._native_inst = None
        if self._native_mod is not None:
            try:
                self._native_inst = self._native_mod.FBGANative(
                    params_path=self.params_txt,
                    gg_path=self.gg_bin)
                self._log(f"[FBGARunner] using fbga_native (no subprocess)  "
                          f"so={getattr(self._native_mod, '__file__', '?')}")
            except Exception as e:
                self._log(f"[FBGARunner] fbga_native init failed, "
                          f"falling back to exe: {e}")
                self._native_inst = None
        else:
            self._log("[FBGARunner] fbga_native module unavailable, using exe")

        self._log(f"[FBGARunner] ready bin={self.fbga_bin} gg={self.gg_bin}")

    # gg.bin / params helpers (identical to fbga_velocity_planner.py)
    def _generate_gg_bin(self, bin_path):
        npy_dir = os.path.dirname(bin_path)
        self._log(f"[FBGARunner] gg.bin not found, generating from {npy_dir}")
        v_list = np.load(os.path.join(npy_dir, 'v_list.npy')).astype(np.float64)
        g_list = np.load(os.path.join(npy_dir, 'g_list.npy')).astype(np.float64)
        ax_max = np.load(os.path.join(npy_dir, 'ax_max.npy')).astype(np.float64)
        ax_min = np.load(os.path.join(npy_dir, 'ax_min.npy')).astype(np.float64)
        ay_max = np.load(os.path.join(npy_dir, 'ay_max.npy')).astype(np.float64)
        gg_exp = np.load(os.path.join(npy_dir, 'gg_exponent.npy')).astype(np.float64)
        nv, ng = len(v_list), len(g_list)
        with open(bin_path, 'wb') as f:
            f.write(struct.pack('II', nv, ng))
            for arr in [v_list, g_list, ax_max, ax_min, ay_max, gg_exp]:
                arr.tofile(f)
        self._log(f"[FBGARunner] gg.bin generated nv={nv} ng={ng}")

    # Write a scaled copy of the GGV into
    # <vehicle>_fbga_latest/velocity_frame/ (always overwrite). ax_max/ay_max
    # are multiplied by gg_scale; everything else is copied unchanged. The
    # source folder is never modified.
    def _materialize_scaled_gg(self, source_gg_bin, scale):
        # Always snapshot (incl. scale==1.0) to isolate runtime source edits.

        src_velocity = os.path.dirname(source_gg_bin)        # .../<vehicle>/velocity_frame
        src_vehicle  = os.path.dirname(src_velocity)         # .../<vehicle>
        gg_diagrams  = os.path.dirname(src_vehicle)          # .../gg_diagrams
        src_name     = os.path.basename(src_vehicle)         # rc_car_10th_latest

        dst_name     = f"{src_name}_fbga_{self.tag}"         # rc_car_10th_latest_fbga_<tag>
        dst_velocity = os.path.join(gg_diagrams, dst_name, 'velocity_frame')
        os.makedirs(dst_velocity, exist_ok=True)

        # Load full npy set from source.
        v_list = np.load(os.path.join(src_velocity, 'v_list.npy')).astype(np.float64)
        g_list = np.load(os.path.join(src_velocity, 'g_list.npy')).astype(np.float64)
        ax_max = np.load(os.path.join(src_velocity, 'ax_max.npy')).astype(np.float64)
        ax_min = np.load(os.path.join(src_velocity, 'ax_min.npy')).astype(np.float64)
        ay_max = np.load(os.path.join(src_velocity, 'ay_max.npy')).astype(np.float64)
        gg_exp = np.load(os.path.join(src_velocity, 'gg_exponent.npy')).astype(np.float64)

        # Apply gg_scale to ax_max and ay_max only.
        ax_max_s = ax_max * scale
        ay_max_s = ay_max * scale

        # Overwrite full set in dst (so the folder is self-contained).
        np.save(os.path.join(dst_velocity, 'v_list.npy'),      v_list)
        np.save(os.path.join(dst_velocity, 'g_list.npy'),      g_list)
        np.save(os.path.join(dst_velocity, 'ax_max.npy'),      ax_max_s)
        np.save(os.path.join(dst_velocity, 'ax_min.npy'),      ax_min)
        np.save(os.path.join(dst_velocity, 'ay_max.npy'),      ay_max_s)
        np.save(os.path.join(dst_velocity, 'gg_exponent.npy'), gg_exp)

        # Rebuild gg.bin from the scaled npy.
        nv, ng = len(v_list), len(g_list)
        dst_bin = os.path.join(dst_velocity, 'gg.bin')
        with open(dst_bin, 'wb') as f:
            f.write(struct.pack('II', nv, ng))
            for arr in [v_list, g_list, ax_max_s, ax_min, ay_max_s, gg_exp]:
                arr.tofile(f)

        self._log(f"[FBGARunner] scaled gg materialised at {dst_velocity} "
                  f"(gg_scale={scale:.3f})")
        return dst_bin

    def _convert_params_yml(self, yml_path):
        with open(yml_path) as f:
            cfg = yaml.safe_load(f)
        vp = cfg['vehicle_params']
        tp = cfg['tire_params']
        with open(self.params_txt, 'w') as f:
            f.write(f"m={vp['m']}\n")
            f.write(f"P_max={vp['P_max']}\n")
            f.write(f"mu_x={tp['p_Dx_1']}\n")
            f.write(f"mu_y={tp['p_Dy_1']}\n")
            f.write(f"v_max={vp['v_max']}\n")
        self.v_max = float(vp['v_max'])

    def _read_g_range(self):
        file_size = os.path.getsize(self.gg_bin)
        with open(self.gg_bin, 'rb') as f:
            nv, ng = struct.unpack('II', f.read(8))
            expected_2d = 8 + 8 * (nv + ng) + 4 * 8 * nv * ng
            if file_size > expected_2d:
                # 3D format: ns header after (nv, ng)
                _ns = struct.unpack('I', f.read(4))[0]
            v_list = np.frombuffer(f.read(nv * 8), dtype=np.float64)
            g_list = np.frombuffer(f.read(ng * 8), dtype=np.float64)
        return float(g_list.min()), float(g_list.max())

    # math helpers
    def compute_g_tilde(self, mu, v, dmu_ds):
        gt = 9.81 * np.cos(mu) - v ** 2 * dmu_ds
        # softplus lower bound at g_min (smooth replacement of np.clip)
        beta = self.g_tilde_soft_beta
        gt_soft = self.g_min + np.log1p(np.exp(beta * (gt - self.g_min))) / beta
        return np.minimum(gt_soft, self.g_max)

    def initial_speed_estimate(self, kappa, mu, dmu_ds):
        ay_max = 4.5
        v_max = 12.0
        radius = np.where(np.abs(kappa) > 1e-4, 1.0 / np.abs(kappa), 1e4)
        v_lat = np.clip(np.sqrt(ay_max * radius), 0, v_max)
        v_vert = np.full_like(kappa, v_max)
        crest = dmu_ds > 1e-4
        v_vert[crest] = np.clip(
            np.sqrt(9.81 * np.cos(mu[crest]) / dmu_ds[crest]), 0.5, v_max)
        return np.minimum(v_lat, v_vert)

    def _apply_kappa_smoothing(self, kappa):
        # Method renamed from 'smooth_kappa' to avoid shadowing the
        # like-named bool attribute set in __init__.
        if not self.smooth_kappa or not HAS_SCIPY_SG:
            return kappa
        win = self.kappa_smooth_window | 1
        if len(kappa) < win:
            return kappa
        poly = max(1, min(self.kappa_smooth_polyorder, win - 1))
        pad = win // 2
        k_pad = np.concatenate([kappa[-pad:], kappa, kappa[:pad]])
        return savgol_filter(k_pad, win, poly)[pad:-pad]

    def _apply_mu_smoothing(self, mu, ds_grid):
        """Returns (mu_smoothed, dmu_ds). Uses SG filter when enabled."""
        n = len(mu)
        if self.smooth_mu and HAS_SCIPY_SG and n >= self.mu_smooth_window:
            win = self.mu_smooth_window | 1
            poly = max(1, min(self.mu_smooth_polyorder, win - 1))
            pad = win // 2
            mu_pad = np.concatenate([mu[-pad:], mu, mu[:pad]])
            mu_s = savgol_filter(mu_pad, win, poly)[pad:-pad]
            dmu_ds = savgol_filter(mu_pad, win, poly,
                                   deriv=1, delta=ds_grid)[pad:-pad]
            return mu_s, dmu_ds
        # central diff fallback (open path: pad with endpoints, not wrap).
        if n > 1:
            mu_pad = np.concatenate([[mu[0]], mu, [mu[-1]]])
            dmu_ds = (mu_pad[2:] - mu_pad[:-2]) / (2.0 * ds_grid)
        else:
            dmu_ds = np.zeros_like(mu)
        return mu, dmu_ds

    def run_once(self, s, kappa, g_tilde, mu, dmu_ds, v0, timeout_s=10.0):
        """One FBGA invocation. Returns (v, ax) arrays or None on failure.
        Native path first (no subprocess); the exe is the fallback."""
        # -- native (in-process) path --
        if self._native_inst is not None:
            try:
                v, ax = self._native_inst.solve(
                    np.asarray(s,       dtype=np.float64),
                    np.asarray(kappa,   dtype=np.float64),
                    np.asarray(g_tilde, dtype=np.float64),
                    np.asarray(mu,      dtype=np.float64),
                    np.asarray(dmu_ds,  dtype=np.float64),
                    float(v0),
                    float(self.bridge_effect),   # slope_corr
                )
                return v, ax
            except Exception as e:
                self._log(f"[FBGARunner] native solve raised {type(e).__name__}: {e}; "
                          f"falling back to exe for this call")
                # fall through to exe path

        # -- exe path (subprocess + CSV) --
        input_csv  = os.path.join(tempfile.gettempdir(), 'fbga_input.csv')
        output_csv = os.path.join(tempfile.gettempdir(), 'fbga_output.csv')

        with open(input_csv, 'w') as f:
            if self.enable_mu:
                f.write('s,kappa,g_tilde,mu,dmu_ds\n')
                for i in range(len(s)):
                    f.write(f'{s[i]:.6f},{kappa[i]:.8f},{g_tilde[i]:.6f},'
                            f'{mu[i]:.8f},{dmu_ds[i]:.8f}\n')
            else:
                f.write('s,kappa,g_tilde\n')
                for i in range(len(s)):
                    f.write(f'{s[i]:.6f},{kappa[i]:.8f},{g_tilde[i]:.6f}\n')

        cmd = [
            self.fbga_bin,
            '--model', 'lookup',
            '--input', input_csv,
            '--params', self.params_txt,
            '--gg', self.gg_bin,
            '--output', output_csv,
            '--v0', f'{v0:.4f}',
            '--slope-corr', f'{self.bridge_effect:.4f}',
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                           timeout=timeout_s)
        except subprocess.CalledProcessError as e:
            self._log(f"[FBGARunner] exe rc={e.returncode}: {e.stderr[:300]}")
            return None
        except subprocess.TimeoutExpired:
            self._log("[FBGARunner] exe timeout")
            return None
        except Exception as e:
            self._log(f"[FBGARunner] exe error: {e}")
            return None

        v_out, ax_out = [], []
        with open(output_csv) as f:
            for line in f:
                if line.startswith('#') or line.startswith('s,'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    v_out.append(float(parts[1]))
                    ax_out.append(float(parts[2]))
        try:
            os.remove(input_csv); os.remove(output_csv)
        except OSError:
            pass
        return np.array(v_out), np.array(ax_out)

    def solve_open(self, s, kappa, mu, *, v_seed=None, v_start=None,
                   max_iter=None, tol=None):
        """Local-cycle solve on an OPEN path (no lap stacking).

        s, kappa, mu : length-N arrays (s monotone)
        v_seed       : optional warm-start shape, length N
        v_start      : optional start-speed boundary [m/s] (e.g. current speed);
                       overrides v_seed[0] as the v0 anchor when given
        returns (v, ax) length N, or None on failure
        """
        n = len(s)
        if n < 5:
            return None
        max_iter = self.max_iter if max_iter is None else int(max_iter)
        tol      = self.tol      if tol      is None else float(tol)

        ds_grid = float(s[1] - s[0]) if n > 1 else 0.1
        mu_s, dmu_ds = self._apply_mu_smoothing(np.asarray(mu, dtype=float), ds_grid)
        kappa_s = self._apply_kappa_smoothing(np.asarray(kappa, dtype=float))

        if v_seed is not None and np.any(np.asarray(v_seed) > 0.1):
            v_prev = np.asarray(v_seed, dtype=float).copy()
        else:
            v_prev = self.initial_speed_estimate(kappa_s, mu_s, dmu_ds)

        # anchor the warm-start head to the given start speed
        if v_start is not None:
            v_prev[0] = float(v_start)

        v_new = ax_new = None
        for _ in range(max_iter):
            g_tilde = self.compute_g_tilde(mu_s, v_prev, dmu_ds)
            v0 = max(float(v_start) if v_start is not None else float(v_prev[0]), 1.0)
            res = self.run_once(np.asarray(s, dtype=float),
                                kappa_s, g_tilde, mu_s, dmu_ds, v0)
            if res is None:
                return None
            v_new, ax_new = res
            if v_new.shape[0] != n:
                return None

            nan_mask = np.isnan(v_new)
            if nan_mask.any():
                if nan_mask.sum() / n > 0.05:
                    return None
                valid = np.where(~nan_mask)[0]
                v_new[nan_mask] = np.interp(np.where(nan_mask)[0],
                                            valid, v_new[valid])

            if np.max(np.abs(v_new - v_prev)) < tol:
                break
            v_prev = self.alpha * v_new + (1.0 - self.alpha) * v_prev

        if v_new is None:
            return None

        nm = np.isnan(ax_new)
        if nm.any():
            valid_ax = np.where(~nm)[0]
            if len(valid_ax) == 0:
                return None
            ax_new[nm] = np.interp(np.where(nm)[0], valid_ax, ax_new[valid_ax])

        v_new = np.minimum(v_new, self.v_max)
        return v_new, ax_new

    def refresh_from_disk(self):
        """Re-pack gg.bin from .npy + rebuild native instance.
        Used by /fbga/reload after gg_tuner regenerates GGV. No launch restart.
        Returns True on success. On native rebuild failure, the old instance
        is kept (atomic swap)."""
        try:
            self.gg_bin = self._materialize_scaled_gg(self.gg_bin_source, self.gg_scale)
            self._convert_params_yml(self.params_yml)
            self.g_min, self.g_max = self._read_g_range()
        except Exception as e:
            self._log(f"[FBGARunner] refresh_from_disk: materialise failed: {e}")
            return False

        if self._native_mod is None:
            self._log("[FBGARunner] refresh_from_disk: exe mode, files updated")
            return True

        try:
            new_inst = self._native_mod.FBGANative(
                params_path=self.params_txt, gg_path=self.gg_bin)
        except Exception as e:
            self._log(f"[FBGARunner] refresh_from_disk: native rebuild failed: {e}")
            return False
        self._native_inst = new_inst   # atomic swap; refcount keeps old alive for in-flight solve
        self._log(f"[FBGARunner] refresh_from_disk: native rebuilt (gg={self.gg_bin})")
        return True

    def reload(self, *, gg_bin=None, params_yml=None, **kwargs):
        """Hot-reload paths and runtime knobs (mirrors fbga_velocity_planner reload_cb)."""
        gg_changed     = gg_bin is not None
        params_changed = params_yml is not None and os.path.exists(params_yml)
        if gg_changed:
            self.gg_bin = gg_bin
            if not os.path.exists(self.gg_bin):
                self._generate_gg_bin(self.gg_bin)
            self.g_min, self.g_max = self._read_g_range()
        if params_changed:
            self.params_yml = params_yml
            self._convert_params_yml(params_yml)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Rebuild the native instance when gg or params changed so
        # the in-memory tables stay in sync with the files on disk.
        if (gg_changed or params_changed) and self._native_mod is not None:
            try:
                self._native_inst = self._native_mod.FBGANative(
                    params_path=self.params_txt, gg_path=self.gg_bin)
                self._log("[FBGARunner] native reloaded with new gg/params")
            except Exception as e:
                self._log(f"[FBGARunner] native reload failed: {e}; "
                          f"continuing with prior instance")

        self._log("[FBGARunner] reloaded")
