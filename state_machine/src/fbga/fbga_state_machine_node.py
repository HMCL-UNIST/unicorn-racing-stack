#!/usr/bin/env python3
# FBGA-enabled state_machine (subclass; parent owns transitions).
# No 2.5d: cbs store raw planner wpnts (curvature-scaled vx = fallback). Main loop
# solves FBGA once per new msg stamp, anchored to cur_vs, and caches it. get_*_wpts
# applies that cache (latch, re-anchored to cur_vs) to the published window.
# Per-mode v_scale / gg_scale tuned live via rqt_reconfigure (FBGATuner.cfg).

import os
import sys
import importlib.util

import numpy as np
import rospy
import yaml
from rospkg import RosPack
from std_srvs.srv import Trigger, TriggerResponse
from dynamic_reconfigure.server import Server as DynServer


# Load 3d_state_machine_node.py via importlib (filename starts with a digit).
_SM_SRC = os.path.join(RosPack().get_path('state_machine'), 'src')
if _SM_SRC not in sys.path:
    sys.path.insert(0, _SM_SRC)

_sm_path = os.path.join(_SM_SRC, '3d_state_machine_node.py')
_spec = importlib.util.spec_from_file_location('sm3d', _sm_path)
sm3d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm3d)

from states_types import StateType
from fbga_runner import FBGARunner
from state_machine.cfg import FBGATunerConfig


# the four FBGA-controlled modes. Order matters for YAML round-trip.
FBGA_MODES = ['recovery', 'avoidance', 'static_avoidance', 'start']


class FBGAStateMachine(sm3d.StateMachine):
    """Recomputes vx_mps for RECOVERY / OVERTAKE / START via real C++ FBGA.

    Each mode has its own FBGARunner instance (own scaled gg.bin), so
    gg_scale can differ per mode. v_scale is a cheap output multiplier
    applied at overlay time. Both are live-tunable via rqt.
    """

    def __init__(self, name):
        # FBGA attrs initialised BEFORE super().__init__ so parent-registered
        # subscribers don't trip on missing fields when they fire mid-init.
        self._fbga = {}                          ## per-mode FBGARunner dict
        self.fbga_states = set()
        self._fbga_max_fail_streak = 100000
        self._fbga_fail = {}
        self._fbga_disabled_states = set()

        # per-mode v_scale - pending = slider value, active = currently
        # applied multiplier (used at overlay). Apply commits pending->active.
        # v_scale used to apply immediately; now it is staged like gg_scale so
        # nothing changes while dragging sliders - only the apply button does.
        self._fbga_v_scale         = {m: 1.0 for m in FBGA_MODES}   ## active
        self._fbga_v_scale_pending = {m: 1.0 for m in FBGA_MODES}   ## slider stage
        # per-mode gg_scale - pending = slider value, active = currently
        # materialised in the runner's gg.bin. Apply commits pending->active.
        self._fbga_gg_scale_pending = {m: 1.0 for m in FBGA_MODES}
        self._fbga_gg_scale_active  = {m: 1.0 for m in FBGA_MODES}

        ## per-mode FBGA cache / latch: last good solve (s_raw/v/ax) + stamp + t.
        self._fbga_overlay = {m: {'stamp': None, 'v': None, 'ax': None,
                                  's_raw': None, 't': None, 'L': 0.0}
                              for m in FBGA_MODES}

        super().__init__(name)

        # which states actually drive FBGA. The set is independent of
        # per-mode tuning (you can disable a state here while still having
        # a slider sitting in rqt for it).
        states_param = rospy.get_param('~fbga_states',
                                       ['RECOVERY', 'OVERTAKE', 'START'])
        self.fbga_states = {StateType[s] for s in states_param}
        self._fbga_max_fail_streak = int(rospy.get_param('~fbga_max_fail_streak', 100000))
        ## latch: hold last FBGA up to TTL [s]; re-anchor start within ax [m/s^2].
        self._fbga_latch_ttl    = float(rospy.get_param('~fbga_latch_ttl', 0.5))
        self._fbga_latch_ax_max = float(rospy.get_param('~fbga_latch_ax_max', 5.0))

        ## Resolve FBGA binary / gg / params paths once. Empty rosparam falls
        ## back to canonical race_stack defaults so launch files can stay blank.
        race_stack = os.path.abspath(os.path.join(_SM_SRC, '..', '..'))
        default_fbga_bin = os.path.join(race_stack, 'f110_utils', 'libs',
            'FBGA', 'bin', 'GIGI_test_unicorn.exe')
        default_gg_bin = os.path.join(race_stack, 'planner', '3d_gb_optimizer',
            'global_line', 'data', 'gg_diagrams', 'rc_car_10th_latest',
            'velocity_frame', 'gg.bin')
        default_params_yml = os.path.join(race_stack, 'planner',
            '3d_gb_optimizer', 'global_line', 'data', 'vehicle_params',
            'params_rc_car_10th_latest.yml')

        fbga_bin   = rospy.get_param('~fbga_bin',        '') or default_fbga_bin
        gg_bin     = rospy.get_param('~fbga_gg_bin',     '') or default_gg_bin
        params_yml = rospy.get_param('~fbga_params_yml', '') or default_params_yml

        ## Spawn one FBGARunner per mode. Each writes its own scaled gg.bin
        ## under <vehicle>_fbga_<mode>/ so refresh_from_disk on one mode
        ## doesn't clobber another's table.
        bridge_effect = float(rospy.get_param('~bridge_effect', 1.0))
        fbga_alpha    = float(rospy.get_param('~fbga_alpha',    0.5))
        fbga_max_iter = int(  rospy.get_param('~fbga_max_iter', 5))
        fbga_tol      = float(rospy.get_param('~fbga_tol',      0.05))
        enable_mu     = bool( rospy.get_param('~fbga_enable_mu', True))
        for mode in FBGA_MODES:
            try:
                self._fbga[mode] = FBGARunner(
                    fbga_bin=fbga_bin,
                    gg_bin=gg_bin,
                    params_yml=params_yml,
                    enable_mu=enable_mu,
                    bridge_effect=bridge_effect,
                    alpha=fbga_alpha,
                    max_iter=fbga_max_iter,
                    tol=fbga_tol,
                    gg_scale=1.0,
                    tag=mode,
                    logger=rospy.loginfo,
                )
                rospy.loginfo(f'[FBGAStateMachine] runner[{mode}] ready')
            except Exception as e:
                rospy.logwarn(
                    f'[FBGAStateMachine] runner[{mode}] init failed: {e}')

        ## /fbga/reload service - refresh ALL modes' GGV without restart.
        self._reload_srv = rospy.Service(
            '/fbga/reload', Trigger, self._fbga_reload_cb)

        # YAML save/load path is per-map so different tracks keep
        # independent tuning. Falls back to ~/.ros if /map is unset.
        # /map can be clobbered into a dict when a perception node is
        # launched as name="map" (its private params land under /map/*),
        # which made os.path.join crash with a dict. Resolve the map name
        # robustly: accept /map only if it's a string, otherwise recover it
        # from another node's private 'map' param, else fall back to ~/.ros.
        map_name = self._resolve_map_name()
        self._map_name = map_name   # keep for health-check logging
        if map_name:
            self._yaml_path = os.path.join(
                RosPack().get_path('stack_master'),
                'maps', map_name, 'fbga_tuner.yaml')
        else:
            self._yaml_path = os.path.join(
                os.path.expanduser('~'), '.ros', 'fbga_tuner.yaml')
        # log the map-name resolution path once, right here.
        self._check_map_src()

        ## dynamic_reconfigure : live per-mode v_scale / gg_scale.
        ## Note: dyn_reconfigure first call fires with defaults, which may
        ## clobber any rosparams loaded from yaml. We deal with that by
        ## auto-loading the yaml AFTER the server is up (below).
        self._dyn_srv = DynServer(FBGATunerConfig, self._fbga_dyn_cb)

    # -- map-name resolution ---------------------------------------
    # robustly resolve the current map name as a plain string.
    # Records how it was resolved in self._map_src so the main loop can
    # periodically warn/err whether we are on the normal or fallback path.
    def _resolve_map_name(self):
        ## 1) /map as a string is the normal case.
        try:
            m = rospy.get_param('/map', None)
        except Exception:
            m = None
        if isinstance(m, str) and m.strip():
            self._map_src = 'direct'   # /map was a clean string
            self._map_src_detail = '/map'
            return m.strip()

        ## 2) /map got clobbered into a dict (e.g. a perception node named
        ##    "map" parked its private params under /map/*). Recover the map
        ##    name from another node that stores it as a private 'map' param.
        for p in ('/global_republisher/map', '/bridge_sector_node_3d/map',
                  '/friction/map', '/gg_tuner/map'):
            try:
                v = rospy.get_param(p, None)
            except Exception:
                v = None
            if isinstance(v, str) and v.strip():
                self._map_src = 'fallback'   # recovered from another node
                self._map_src_detail = p
                return v.strip()   # _check_map_src() logs this once

        ## 3) nothing usable -> caller falls back to ~/.ros.
        self._map_src = 'none'   # could not resolve at all
        self._map_src_detail = f'/map type={type(m).__name__}'
        return None   # _check_map_src() logs this once

    # -- map-source health check (once, at startup) ----------------
    # log the resolution path exactly ONCE so we can confirm the
    # map name was found correctly without spamming the console.
    # 'direct'   -> /map was a clean string (normal).
    # 'fallback' -> map name recovered from a sibling node (WARN).
    # 'none'     -> no map name at all (~/.ros yaml) (ERROR).
    def _check_map_src(self):
        src = getattr(self, '_map_src', 'direct')
        detail = getattr(self, '_map_src_detail', '?')
        if src == 'direct':
            ## node log_level is WARN, so use logwarn to make the
            ## "did we find the right map name?" line actually visible.
            rospy.logwarn(
                f'[FBGAStateMachine] map OK: name="{self._map_name}" '
                f'via {detail} -> {self._yaml_path}')
        elif src == 'fallback':
            rospy.logwarn(
                f'[FBGAStateMachine] map via FALLBACK: /map was clobbered, '
                f'using name="{self._map_name}" recovered from {detail} '
                f'-> {self._yaml_path}')
        else:  # 'none'
            rospy.logerr(
                f'[FBGAStateMachine] map UNRESOLVED ({detail}); '
                f'tuner yaml falls back to {self._yaml_path}')

    # -- reload service --------------------------------------------
    def _fbga_reload_cb(self, req):
        ok_all = True
        msgs = []
        for mode, runner in self._fbga.items():
            try:
                ok = runner.refresh_from_disk()
                ok_all &= ok
                msgs.append(f'{mode}:{"ok" if ok else "fail"}')
                self._fbga_overlay[mode] = {'stamp': None, 'v': None, 'ax': None}
            except Exception as e:
                rospy.logwarn(f"[FBGAStateMachine] reload[{mode}] exc: {e}")
                ok_all = False
                msgs.append(f'{mode}:exc')
        return TriggerResponse(success=ok_all, message=', '.join(msgs))

    # -- dynamic_reconfigure callback ------------------------------
    def _fbga_dyn_cb(self, config, level):
        ## Sliders only STAGE pending values - nothing is applied while dragging.
        for mode in FBGA_MODES:
            self._fbga_v_scale_pending[mode]  = float(config[f'{mode}_v_scale'])
            self._fbga_gg_scale_pending[mode] = float(config[f'{mode}_gg_scale'])

        ## apply : commit ALL pending v_scale AND gg_scale at once. Auto-off.
        if config['apply']:
            self._fbga_commit_pending_v_scales()
            self._fbga_commit_pending_gg_scales()
            config['apply'] = False

        ## save_param : dump current v/gg + active gg to yaml. Auto-off.
        if config['save_param']:
            self._fbga_save_yaml()
            config['save_param'] = False

        ## load_param : read yaml, push into config so rqt UI updates, also
        ## refresh runners since gg_scale may have changed. Auto-off.
        if config['load_param']:
            loaded = self._fbga_load_yaml()
            if loaded is not None:
                for mode in FBGA_MODES:
                    if mode in loaded:
                        v = float(loaded[mode].get('v_scale', 1.0))
                        g = float(loaded[mode].get('gg_scale', 1.0))
                        config[f'{mode}_v_scale']  = v
                        config[f'{mode}_gg_scale'] = g
                        self._fbga_v_scale_pending[mode]  = v
                        self._fbga_gg_scale_pending[mode] = g
                ## load = explicit user action -> commit both immediately
                self._fbga_commit_pending_v_scales()
                self._fbga_commit_pending_gg_scales()
            config['load_param'] = False

        return config

    def _fbga_commit_pending_v_scales(self):
        """Apply pending->active v_scale for any mode whose value changed."""
        for mode in FBGA_MODES:
            pending = self._fbga_v_scale_pending[mode]
            active  = self._fbga_v_scale[mode]
            if abs(pending - active) < 1e-6:
                continue
            self._fbga_v_scale[mode] = pending
            rospy.logwarn(
                f'[FBGAStateMachine] {mode} v_scale: {active:.3f} -> {pending:.3f}')

    def _fbga_commit_pending_gg_scales(self):
        """Apply pending->active for any mode whose gg_scale differs."""
        for mode in FBGA_MODES:
            pending = self._fbga_gg_scale_pending[mode]
            active  = self._fbga_gg_scale_active[mode]
            if abs(pending - active) < 1e-6:
                continue
            runner = self._fbga.get(mode)
            if runner is None:
                continue
            try:
                runner.gg_scale = pending
                ok = runner.refresh_from_disk()
                if ok:
                    self._fbga_gg_scale_active[mode] = pending
                    self._fbga_overlay[mode] = {'stamp': None, 'v': None, 'ax': None}
                    rospy.logwarn(
                        f'[FBGAStateMachine] {mode} gg_scale: {active:.3f} -> {pending:.3f}')
                else:
                    rospy.logwarn(
                        f'[FBGAStateMachine] {mode} gg_scale refresh returned False')
            except Exception as e:
                rospy.logwarn(
                    f'[FBGAStateMachine] {mode} gg_scale refresh exc: {e}')

    # -- YAML save / load ------------------------------------------
    def _fbga_save_yaml(self):
        data = {}
        ## save the ACTIVE (committed) values, not the staged sliders.
        for mode in FBGA_MODES:
            data[mode] = {
                'v_scale':  float(self._fbga_v_scale[mode]),
                'gg_scale': float(self._fbga_gg_scale_active[mode]),
            }
        try:
            os.makedirs(os.path.dirname(self._yaml_path), exist_ok=True)
            with open(self._yaml_path, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            rospy.logwarn(f'[FBGAStateMachine] saved -> {self._yaml_path}')
        except Exception as e:
            rospy.logwarn(f'[FBGAStateMachine] save failed: {e}')

    def _fbga_load_yaml(self):
        if not os.path.exists(self._yaml_path):
            rospy.logwarn(f'[FBGAStateMachine] no yaml at {self._yaml_path}')
            return None
        try:
            with open(self._yaml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            rospy.logwarn(f'[FBGAStateMachine] loaded <- {self._yaml_path}')
            return data
        except Exception as e:
            rospy.logwarn(f'[FBGAStateMachine] load failed: {e}')
            return None

    # -- source cbs : store raw wpnts (planner vx = no-FBGA fallback) --
    def recovery_wpnts_cb(self, data):
        self.recovery_wpnts = data

    def avoidance_cb(self, data):
        self.avoidance_wpnts = data

    def static_avoidance_cb(self, data):
        self.static_avoidance_wpnts = data

    # -- main-loop entry : solve + cache FBGA per new msg stamp --
    def _fbga_step(self):
        self._fbga_overlay_latest()

    def _fbga_overlay_latest(self):
        st = getattr(self, 'cur_state', None)
        if st not in self.fbga_states or st in self._fbga_disabled_states:
            return

        ## (wpnts_list, stamp, mode_key)
        targets = []
        if st == StateType.RECOVERY:
            if self.recovery_wpnts is not None and len(self.recovery_wpnts.wpnts) > 0:
                targets.append((self.recovery_wpnts.wpnts,
                                self.recovery_wpnts.header.stamp, 'recovery'))
        elif st == StateType.OVERTAKE:
            if self.avoidance_wpnts is not None and len(self.avoidance_wpnts.wpnts) > 0:
                targets.append((self.avoidance_wpnts.wpnts,
                                self.avoidance_wpnts.header.stamp, 'avoidance'))
            if self.static_avoidance_wpnts is not None and len(self.static_avoidance_wpnts.wpnts) > 0:
                targets.append((self.static_avoidance_wpnts.wpnts,
                                self.static_avoidance_wpnts.header.stamp, 'static_avoidance'))
        elif st == StateType.START:
            if self.cur_start_wpnts.is_init and len(self.cur_start_wpnts.list) > 0:
                targets.append((self.cur_start_wpnts.list,
                                self.cur_start_wpnts.stamp, 'start'))
        else:
            return

        for wpnts, stamp, key in targets:
            self._fbga_solve_and_cache(wpnts, stamp, key, st)

    def _fbga_solve_and_cache(self, wpnts, cur_stamp, key, st):
        # Solve per new msg stamp; cache profile. Applied in get_*_wpts (latch).
        n = len(wpnts)
        if n < 5:
            return
        runner = self._fbga.get(key)
        if runner is None:
            return

        cache = self._fbga_overlay[key]
        if cache['stamp'] == cur_stamp and cache['v'] is not None:
            return  # already solved this msg

        s      = np.array([w.s_m         for w in wpnts])
        s_raw  = s.copy()                        # raw frenet s for latch sampling
        kappa  = np.array([w.kappa_radpm for w in wpnts])
        mu     = np.array([getattr(w, 'mu_rad', 0.0) for w in wpnts])
        v_seed = np.array([w.vx_mps     for w in wpnts])

        ## unwrap s if path crosses start/finish (FBGA needs monotone s)
        track_length = float(getattr(self, 'track_length', 0.0)) or 0.0
        if track_length <= 0.0 and self.gb_wpnts is not None:
            track_length = float(self.gb_wpnts.wpnts[-1].s_m) + self.wpnt_dist
        ds_raw = np.diff(s)
        jumps = np.where(ds_raw < -track_length * 0.5)[0]
        if len(jumps) > 0 and track_length > 0.0:
            for j in jumps:
                s[j + 1:] = s[j + 1:] + track_length

        ds = np.diff(s)
        rospy.logwarn_throttle(1.0,
            f'[FBGA-DIAG] {st.name}/{key} input n={n} '
            f's=[{s.min():.2f},{s.max():.2f}] ds=[{ds.min():.3f},{ds.max():.3f}] '
            f'v_seed=[{v_seed.min():.2f},{v_seed.max():.2f}] jumps={len(jumps)}')

        if ds.min() <= 0:
            self._fbga_register_fail(st, key, f's not monotone (ds.min={ds.min():.4f})')
            return

        t0 = rospy.Time.now()
        try:
            ## v_start = current speed anchor; v_seed = warm-start shape.
            res = runner.solve_open(
                s, kappa, mu,
                v_seed=v_seed if np.any(v_seed > 0.1) else None,
                v_start=float(self.cur_vs),
            )
        except Exception as exc:
            self._fbga_register_fail(st, key,
                f'solve_open exc {type(exc).__name__}: {exc}')
            return
        solve_ms = (rospy.Time.now() - t0).to_sec() * 1000.0

        if res is None:
            self._fbga_register_fail(st, key,
                f'solve_open None after {solve_ms:.0f}ms')
            return
        v_new, ax_new = res
        if v_new.shape[0] != n or ax_new.shape[0] != n:
            self._fbga_register_fail(st, key,
                f'len mismatch v={v_new.shape[0]} ax={ax_new.shape[0]} expected={n}')
            return

        v_max = self.pars["veh_params"]["v_max"]
        v_new = np.minimum(v_new, v_max)

        ## v_end clamp at GB join (mirrors legacy L1537)
        try:
            gi = int(wpnts[-1].s_m / self.wpnt_dist)
            v_end_gb = self.gb_wpnts.wpnts[gi % len(self.gb_wpnts.wpnts)].vx_mps
            v_new[-1] = min(v_new[-1], v_end_gb)
        except Exception:
            pass

        ## store latch (sampled by s in get_*_wpts; t gates the TTL)
        cache['stamp'] = cur_stamp
        cache['v']     = v_new
        cache['ax']    = ax_new
        cache['s_raw'] = s_raw
        cache['t']     = rospy.Time.now()
        cache['L']     = track_length
        self._fbga_fail[st] = 0

        rospy.logwarn_throttle(1.0,
            f'[FBGA-DIAG] {st.name}/{key} OK solve_ms={solve_ms:.0f} '
            f'v=[{v_new.min():.2f},{v_new.max():.2f}]')

    def _fbga_register_fail(self, st, key, reason):
        self._fbga_fail[st] = self._fbga_fail.get(st, 0) + 1
        rospy.logwarn_throttle(1.0,
            f'[FBGA-DIAG] {st.name}/{key} FAIL '
            f'({self._fbga_fail[st]}/{self._fbga_max_fail_streak}): {reason}')
        if self._fbga_fail[st] >= self._fbga_max_fail_streak:
            rospy.logwarn(
                f'[FBGAStateMachine] {st.name} FBGA failed '
                f'{self._fbga_fail[st]}x -> disable for episode')
            self._fbga_disabled_states.add(st)

    # -- published-window getters : apply FBGA latch before publish --
    # states.py-only; runs before behavior_strategy so the controller sees it.
    # No fresh latch -> raw planner vx stands.
    def get_recovery_wpts(self):
        wpts = super().get_recovery_wpts()
        self._fbga_apply_latch(wpts, 'recovery')
        return wpts

    def get_splini_wpts(self):
        wpts = super().get_splini_wpts()
        key = 'static_avoidance' if getattr(self, 'static_overtaking_mode', False) else 'avoidance'
        self._fbga_apply_latch(wpts, key)
        return wpts

    def get_start_wpts(self):
        wpts = super().get_start_wpts()
        self._fbga_apply_latch(wpts, 'start')
        return wpts

    def _fbga_apply_latch(self, wpts, key):
        if not wpts:
            return
        st = getattr(self, 'cur_state', None)
        if st not in self.fbga_states or st in self._fbga_disabled_states:
            return
        cache = self._fbga_overlay.get(key)
        if not cache or cache['v'] is None or cache['s_raw'] is None or cache['t'] is None:
            return                                   # never solved -> raw vx stands
        if (rospy.Time.now() - cache['t']).to_sec() > self._fbga_latch_ttl:
            return                                   # stale -> raw vx stands

        s_lat = cache['s_raw']; v_lat = cache['v']; ax_lat = cache['ax']
        L = cache['L'] or 0.0
        scale = self._fbga_v_scale.get(key, 1.0)
        ds_tol = max(0.15, 1.5 * self.wpnt_dist)

        # nearest-s (circular) sample of the latched profile onto the window
        q = np.array([w.s_m for w in wpts])
        d = np.abs(s_lat[None, :] - q[:, None])
        if L > 0:
            d = np.minimum(d, L - d)
        idx = d.argmin(axis=1)
        dmin = d[np.arange(len(q)), idx]
        v_tgt = v_lat[idx] * scale
        ax_tgt = ax_lat[idx]

        # re-anchor start to current speed, then track v_tgt within accel limit
        a = self._fbga_latch_ax_max
        v_out = v_tgt.copy()
        v_out[0] = max(float(self.cur_vs), 0.0)
        for i in range(1, len(wpts)):
            dxy = float(np.hypot(wpts[i].x_m - wpts[i - 1].x_m,
                                 wpts[i].y_m - wpts[i - 1].y_m))
            v_up = np.sqrt(v_out[i - 1] ** 2 + 2.0 * a * dxy)
            v_dn = np.sqrt(max(v_out[i - 1] ** 2 - 2.0 * a * dxy, 0.0))
            v_out[i] = min(max(float(v_tgt[i]), v_dn), v_up)

        for i, w in enumerate(wpts):
            if dmin[i] > ds_tol:
                continue                             # beyond FBGA segment -> keep raw
            w.vx_mps  = float(v_out[i])
            w.ax_mps2 = float(ax_tgt[i])


if __name__ == "__main__":
    name = "state_machine"
    rospy.init_node(name, anonymous=False, log_level=rospy.WARN)

    sm = FBGAStateMachine(name)

    rospy.on_shutdown(sm.on_shutdown)

    loop_rate = rospy.Rate(sm.rate_hz)
    while not rospy.is_shutdown():
        sm._fbga_step()
        sm.loop()
        loop_rate.sleep()
