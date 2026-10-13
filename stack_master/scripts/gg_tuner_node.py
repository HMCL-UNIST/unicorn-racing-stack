#!/usr/bin/env python3

import os
import sys
import copy
import json
import shutil
import struct
import subprocess
import threading
import time
from datetime import datetime

import numpy as np
import rospkg
import rospy
import yaml
from std_msgs.msg import String
from dynamic_reconfigure.server import Server
from stack_master.cfg import GGTunerConfig
# Trigger service for FBGA hot-reload + viewer-driven save_version
from std_srvs.srv import Trigger, TriggerResponse
# bridge yaml IO — single source of truth for bridge positions + tuning.
# Add the bridge_sector_tuner_3d package src dir to sys.path so we can
# import its shared helper without needing it on PYTHONPATH at launch.
_BRIDGE_TUNER_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '..', 'f110_utils', 'nodes', '3d_bridge_sector_tuner', 'src')
if os.path.isdir(_BRIDGE_TUNER_SRC) and _BRIDGE_TUNER_SRC not in sys.path:
    sys.path.insert(0, _BRIDGE_TUNER_SRC)
try:
    import bridge_yaml_io
except ImportError:
    bridge_yaml_io = None  # graceful no-op if helper missing
# safety sector tuner — friction-pattern node. safety_sector_server.py
# (safety_setting.launch, spawned by the safety_sector_setting toggle)
# owns the /safety_sector_params/... rosparam tree and the yaml sync.


class GGTunerNode:

    TIRE_KEYS = ['lambda_mu_x', 'lambda_mu_y', 'p_Dx_1', 'p_Dy_1', 'p_Dx_2', 'p_Dy_2']
    VEHICLE_KEYS = ['P_max', 'v_max', 'epsilon',
                    'P_brake_max', 'ax_max_cap', 'ax_min_cap', 'ay_max_cap']
    CAP_KEYS = ['P_brake_max', 'ax_max_cap', 'ax_min_cap', 'ay_max_cap']
    POST_KEYS = ['gg_exp_scale', 'ax_max_scale', 'ax_min_scale', 'ay_scale']
    RACELINE_KEYS = ['V_min', 'safety_distance', 'w_T', 'w_jx', 'w_jy', 'w_dOmega_z']
    ALL_TUNING_KEYS = TIRE_KEYS + VEHICLE_KEYS + POST_KEYS + RACELINE_KEYS

    def __init__(self):
        rospy.loginfo("[GGTuner] Initializing...")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        race_stack_root = os.path.dirname(os.path.dirname(script_dir))
        self.race_stack_root = race_stack_root

        self.data_path = os.path.join(
            race_stack_root, 'planner', '3d_gb_optimizer', 'global_line', 'data')
        self.maps_dir = os.path.join(race_stack_root, 'stack_master', 'maps')

        self.fast_ggv_dir = os.path.join(
            race_stack_root, 'planner', '3d_gb_optimizer', 'fast_ggv_gen')
        self.fast_ggv_script = os.path.join(self.fast_ggv_dir, 'run_on_container.sh')
        self.fast_ggv_output_dir = os.path.join(self.fast_ggv_dir, 'output')

        self.base_vehicle = rospy.get_param("~base_vehicle", "rc_car_10th")

        base_yml_path = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + self.base_vehicle + '.yml')
        backup_yml_path = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + self.base_vehicle + '_backup.yml')

        loaded_path = None
        for path in [base_yml_path, backup_yml_path]:
            if os.path.exists(path) and not os.path.islink(path) or \
               (os.path.islink(path) and os.path.exists(os.readlink(path) if not os.path.isabs(os.readlink(path)) else path)):
                try:
                    with open(path, 'r') as f:
                        self.base_params = yaml.safe_load(f)
                    loaded_path = path
                    break
                except (yaml.YAMLError, OSError):
                    continue
        if loaded_path is None:
            rospy.logerr(f"[GGTuner] Base params not found: tried {base_yml_path} and {backup_yml_path}")
            raise FileNotFoundError(base_yml_path)
        rospy.loginfo(f"[GGTuner] Base params loaded: {loaded_path}")

        # scan available maps
        self.available_maps = self._scan_maps_dir()
        rospy.loginfo(f"[GGTuner] Available maps ({len(self.available_maps)}): "
                      f"{self.available_maps}")

        if not os.path.exists(self.fast_ggv_script):
            rospy.logwarn(f"[GGTuner] fast_ggv script missing: {self.fast_ggv_script}")

        self.status_pub = rospy.Publisher(
            '/gg_compute_status', String, queue_size=5, latch=True)
        self.status_pub.publish(f"READY: {self.base_vehicle}")

        # publish GGV diamond results as JSON for rqt_gg_viewer
        self.results_pub = rospy.Publisher(
            '/gg_results', String, queue_size=1, latch=True)

        self.pipeline_thread = None
        self.pipeline_lock = threading.Lock()
        self.fbga_proc = None
        # velopt subprocess handle (Stage 4 alternative engine)
        self.velopt_proc = None
        # sector_slicing roslaunch handle
        self.sector_slicing_proc = None
        self._last_sector_slicing_state = False
        # sector respawn handle (apply_new_sectors auto-clear trigger)
        self.sector_respawn_proc = None
        # bridge_tuner roslaunch handle (separate from sector_slicing
        # because bridges live on the smoothed centerline, not on the
        # raceline waypoints, so it doesn't need /global_waypoints).
        self.bridge_tuner_proc = None
        self._last_bridge_tuner_state = False
        # track last map name so we can re-sync rqt sliders from yaml
        # when the user picks a different map (yaml is per-map).
        self._last_bridge_yaml_map = None
        # safety sector — 3 independent checkboxes:
        # - safety_sector_tuner   : trigger, slicing GUI only (auto-clear)
        # - safety_sector_setting : persistent toggle, dyn cfg server
        # - safety_sector_use     : raceline application (no spawn)
        # Setting needs edge-detect so we can spawn/kill server on
        # transitions; tuner is a trigger like run_bridge_tuner.
        self.safety_tuner_proc = None
        self.safety_setting_proc = None
        self._last_safety_setting_state = False
        # friction sector — clone of safety pattern (3 checkboxes).
        self.friction_tuner_proc = None
        self.friction_setting_proc = None
        self._last_friction_setting_state = False

        rospy.on_shutdown(self._shutdown_cleanup)

        # viewer-driven save_version service
        self.save_version_srv = rospy.Service(
            '/gg_tuner/save_version', Trigger, self._save_version_cb)

        # restore selected version → _latest (target via rosparam)
        self.restore_version_srv = rospy.Service(
            '/gg_tuner/restore_version', Trigger, self._restore_version_cb)

        self.srv = Server(GGTunerConfig, self.reconfigure_cb)
        rospy.loginfo("[GGTuner] Ready. Use rqt_reconfigure → /gg_tuner")
        # on startup, pull bridge tuning from the current map's yaml
        # so the rqt sliders reflect what's actually on disk (yaml is
        # the single source of truth). Done after Server creation so
        # update_configuration takes effect.
        try:
            initial_map = self.srv.config.get('map', '')
        except Exception:
            initial_map = ''
        if initial_map:
            self._sync_bridge_cfg_from_yaml(initial_map)
        # safety sector server (safety_setting.launch) owns the
        # /safety_sector_params/... rosparam tree and the yaml sync.
        # Nothing to sync here in gg_tuner.

        # startup latch /gg_results from disk _latest so viewer
        # gets current immediately (publisher alone leaves latch empty).
        latest_vf_dir = os.path.join(
            self.data_path, 'gg_diagrams',
            f'{self.base_vehicle}_latest', 'velocity_frame')
        if os.path.isdir(latest_vf_dir):
            self._publish_gg_results(f'{self.base_vehicle}_latest')
            rospy.loginfo(f"[GGTuner] startup latch: {self.base_vehicle}_latest")
        else:
            rospy.loginfo(f"[GGTuner] startup latch skipped (no _latest)")

    # bridge yaml ↔ rqt cfg synchronisation helpers.
    # yaml is the single source of truth; rqt is just an editor that
    # pre-populates from yaml on map switch and writes back on apply.
    def _map_dir(self, map_name):
        return os.path.join(self.maps_dir, map_name) if map_name else ''

    def _sync_bridge_cfg_from_yaml(self, map_name, config=None):
        """Read <map>/bridge_sectors.yaml and push values into rqt cfg.
        Tuning is global per-rqt-slider; we pull from the first bridge if
        present (all bridges share the same tuning under the current UI).

        When called from inside reconfigure_cb, pass the in-flight
        `config` so we mutate it directly. Calling
        srv.update_configuration from inside the callback re-enters
        the server's lock and deadlocks (rqt sliders freeze).
        Startup path (config=None) is safe — it's outside the lock.
        """
        if bridge_yaml_io is None or not map_name:
            return
        map_dir = self._map_dir(map_name)
        if not os.path.isdir(map_dir):
            return
        enable, bridges = bridge_yaml_io.load_bridges(map_dir)
        upd = {'bridge_constraints_enable': bool(enable)}
        if bridges:
            b = bridges[0]
            upd.update({
                'bridge_pre_m':        float(b['pre_m']),
                'bridge_post_m':       float(b['post_m']),
                'bridge_force_center': bool(b['force_center']),
                'bridge_chi_max_rad':  float(b['chi_max_rad']),
            })
        try:
            if config is not None:
                for k, v in upd.items():
                    config[k] = v
            else:
                self.srv.update_configuration(upd)
            rospy.loginfo(
                f"[GGTuner] bridge cfg synced from yaml ({map_name}): "
                f"{len(bridges)} bridge(s), enable={enable}")
        except Exception as e:
            rospy.logwarn(f"[GGTuner] bridge cfg sync failed: {e}")
        self._last_bridge_yaml_map = map_name

    def _dump_bridge_cfg_to_yaml(self, run_opts):
        """Persist current rqt bridge values into <map>/bridge_sectors.yaml.
        Bridge positions are preserved; only the global tuning and master
        enable flag get overwritten. Called on every apply."""
        if bridge_yaml_io is None:
            return
        map_dir = self._map_dir(run_opts.get('map', ''))
        if not os.path.isdir(map_dir):
            return
        try:
            bridge_yaml_io.apply_global_tuning(
                map_dir,
                pre_m=run_opts['bridge_pre_m'],
                post_m=run_opts['bridge_post_m'],
                force_center=run_opts['bridge_force_center'],
                chi_max_rad=run_opts['bridge_chi_max_rad'],
                constraints_enable=run_opts['bridge_constraints_enable'],
            )
        except Exception as e:
            rospy.logwarn(f"[GGTuner] bridge yaml save failed: {e}")


    # viewer-driven save_version
    def _save_version_cb(self, req):
        if self.pipeline_thread is not None and self.pipeline_thread.is_alive():
            return TriggerResponse(success=False, message='pipeline running')
        try:
            ok = self._snapshot_latest_to_version()
            if ok:
                v_name = f'{self.base_vehicle}_v{self._next_version() - 1}'
                return TriggerResponse(success=True, message=v_name)
            return TriggerResponse(success=False, message='snapshot failed')
        except Exception as e:
            rospy.logerr(f'[GGTuner] save_version exception: {e}')
            return TriggerResponse(success=False, message=str(e)[:100])

    # restore selected version → _latest
    def _restore_version_cb(self, req):
        if self.pipeline_thread is not None and self.pipeline_thread.is_alive():
            return TriggerResponse(success=False, message='pipeline running')
        target = rospy.get_param('/gg_tuner/restore_target', '')
        if not target:
            return TriggerResponse(success=False, message='no target')
        prefix = self.base_vehicle + '_v'
        if not (target.startswith(prefix) and target[len(prefix):].isdigit()):
            return TriggerResponse(
                success=False, message=f'invalid target: {target}')
        try:
            ok = self._restore_to_latest(target)
            if not ok:
                return TriggerResponse(success=False, message='restore failed')
            self.status_pub.publish(f'RESTORED: {target} -> latest')
            self._publish_gg_results(f'{self.base_vehicle}_latest')
            return TriggerResponse(
                success=True, message=f'{target} -> latest')
        except Exception as e:
            rospy.logerr(f'[GGTuner] restore_version exception: {e}')
            return TriggerResponse(success=False, message=str(e)[:100])

    def _round_tuning(self, tuning_dict):
        return {k: round(v, 4) for k, v in sorted(tuning_dict.items())}

    def _find_cached(self, tuning_dict):
        rounded = self._round_tuning(tuning_dict)
        gg_dir = os.path.join(self.data_path, 'gg_diagrams')
        prefix = self.base_vehicle + '_v'
        if not os.path.exists(gg_dir):
            return None
        for name in sorted(os.listdir(gg_dir)):
            if not name.startswith(prefix):
                continue
            full = os.path.join(gg_dir, name)
            if os.path.islink(full):
                continue
            suffix = name[len(prefix):]
            if not suffix.isdigit():
                continue
            meta_path = os.path.join(full, 'params_used.json')
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                saved = {k: round(v, 4) for k, v in sorted(meta['tuning'].items())}
                if saved == rounded:
                    return name
            except (json.JSONDecodeError, KeyError):
                continue
        return None

    def _next_version(self):
        gg_dir = os.path.join(self.data_path, 'gg_diagrams')
        prefix = self.base_vehicle + '_v'
        max_ver = 0
        if os.path.exists(gg_dir):
            for name in os.listdir(gg_dir):
                if not name.startswith(prefix):
                    continue
                suffix = name[len(prefix):]
                if not suffix.isdigit():
                    continue
                max_ver = max(max_ver, int(suffix))
        return max_ver + 1
    
    def _merge_all_params(self, tuning_dict):
        merged = copy.deepcopy(self.base_params)
        # NLP: tire
        for key in self.TIRE_KEYS:
            if key in tuning_dict:
                merged['tire_params'][key] = tuning_dict[key]
        # NLP: vehicle (caps: 0.0 → None)
        for key in self.VEHICLE_KEYS:
            if key not in tuning_dict:
                continue
            val = tuning_dict[key]
            if key in self.CAP_KEYS and float(val) <= 0.0:
                merged['vehicle_params'][key] = None
            else:
                merged['vehicle_params'][key] = val
        # Post-process + raceline → top-level
        for key in self.POST_KEYS + self.RACELINE_KEYS:
            if key in tuning_dict:
                merged[key] = float(tuning_dict[key])
        return merged

    def _save_params_yml(self, vehicle_name, merged_params):
        yml_path = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + vehicle_name + '.yml')
        with open(yml_path, 'w') as f:
            yaml.dump(merged_params, f, default_flow_style=False, allow_unicode=True)
        # copy to latest (real file, not symlink)
        latest_path = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + self.base_vehicle + '_latest.yml')
        shutil.copy2(yml_path, latest_path)
        rospy.loginfo(f"[GGTuner] Params saved: {yml_path} → latest copied")
        return yml_path

    def _save_meta(self, vehicle_name, tuning_dict):
        out_dir = os.path.join(self.data_path, 'gg_diagrams', vehicle_name)
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            'vehicle_name': vehicle_name,
            'base_vehicle': self.base_vehicle,
            'tuning': tuning_dict,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        meta_path = os.path.join(out_dir, 'params_used.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        rospy.loginfo(f"[GGTuner] Meta saved: {meta_path}")

    def _write_latest_params_yml(self, merged_params):
        latest_yml = os.path.join(
            self.data_path, 'vehicle_params',
            f'params_{self.base_vehicle}_latest.yml')
        with open(latest_yml, 'w') as f:
            yaml.dump(merged_params, f, default_flow_style=False,
                      allow_unicode=True)
        rospy.loginfo(f"[GGTuner] latest yml written: {latest_yml}")
        return latest_yml

    # overlay rqt RACELINE_KEYS onto existing latest yml in-place
    def _update_raceline_keys_in_yml(self, vehicle_name, tuning):
        yml_path = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + vehicle_name + '.yml')
        if not os.path.exists(yml_path):
            rospy.logerr(f"[GGTuner] raceline-keys overlay: yml missing: {yml_path}")
            return False
        try:
            with open(yml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as e:
            rospy.logerr(f"[GGTuner] raceline-keys overlay: yml load failed: {e}")
            return False
        overlaid = {}
        for k in self.RACELINE_KEYS:
            if k in tuning:
                data[k] = float(tuning[k])
                overlaid[k] = data[k]
        try:
            with open(yml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except OSError as e:
            rospy.logerr(f"[GGTuner] raceline-keys overlay: yml write failed: {e}")
            return False
        rospy.loginfo(f"[GGTuner] raceline keys merged into {os.path.basename(yml_path)}: {overlaid}")
        return True

    def _restore_to_latest(self, cached_name):
        latest_name = f'{self.base_vehicle}_latest'
        src_gg = os.path.join(self.data_path, 'gg_diagrams', cached_name)
        dst_gg = os.path.join(self.data_path, 'gg_diagrams', latest_name)
        src_yml = os.path.join(self.data_path, 'vehicle_params',
                               f'params_{cached_name}.yml')
        dst_yml = os.path.join(self.data_path, 'vehicle_params',
                               f'params_{latest_name}.yml')
        if not os.path.exists(src_gg):
            rospy.logerr(f"[GGTuner] cached gg_diagrams missing: {src_gg}")
            return False
        if os.path.islink(dst_gg):
            os.unlink(dst_gg)
        elif os.path.exists(dst_gg):
            shutil.rmtree(dst_gg)
        shutil.copytree(src_gg, dst_gg)
        if os.path.exists(src_yml):
            shutil.copy2(src_yml, dst_yml)
        rospy.loginfo(f"[GGTuner] restored {cached_name} → latest")
        return True

    def _snapshot_latest_to_version(self, tuning_dict=None):
        ver = self._next_version()
        snapshot_name = f'{self.base_vehicle}_v{ver}'
        rospy.loginfo(f"[GGTuner] ===== SAVE snapshot: {snapshot_name} =====")
        self.status_pub.publish(f"SAVING: {snapshot_name}")

        latest_name = f'{self.base_vehicle}_latest'
        latest_gg = os.path.join(self.data_path, 'gg_diagrams', latest_name)
        latest_yml = os.path.join(self.data_path, 'vehicle_params',
                                  f'params_{latest_name}.yml')
        if not os.path.exists(latest_gg):
            rospy.logerr(f"[GGTuner] SAVE failed: {latest_gg} missing")
            self.status_pub.publish("SAVE_FAILED: no latest gg")
            return False
        if not os.path.exists(latest_yml):
            rospy.logerr(f"[GGTuner] SAVE failed: {latest_yml} missing")
            self.status_pub.publish("SAVE_FAILED: no latest yml")
            return False

        dst_gg = os.path.join(self.data_path, 'gg_diagrams', snapshot_name)
        real_src = (os.path.realpath(latest_gg)
                    if os.path.islink(latest_gg) else latest_gg)
        shutil.copytree(real_src, dst_gg)
        dst_yml = os.path.join(self.data_path, 'vehicle_params',
                               f'params_{snapshot_name}.yml')
        shutil.copy2(latest_yml, dst_yml)

        latest_meta_path = os.path.join(latest_gg, 'params_used.json')
        meta = None
        if os.path.exists(latest_meta_path):
            try:
                with open(latest_meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                meta = None
        if meta is None:
            meta = {
                'vehicle_name': snapshot_name,
                'base_vehicle': self.base_vehicle,
                'tuning': tuning_dict or {},
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
        meta['vehicle_name'] = snapshot_name
        meta['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        meta_path = os.path.join(dst_gg, 'params_used.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        rospy.loginfo(f"[GGTuner] SAVED: {snapshot_name}")
        self.status_pub.publish(f"SAVED: {snapshot_name}")
        return True

    def _scan_maps_dir(self):
        if not os.path.exists(self.maps_dir):
            return []
        return sorted([n for n in os.listdir(self.maps_dir)
                       if os.path.isdir(os.path.join(self.maps_dir, n))])

    def _generate_gg_bin(self, npy_dir):
        bin_path = os.path.join(npy_dir, 'gg.bin')
        try:
            v_list = np.load(os.path.join(npy_dir, 'v_list.npy')).astype(np.float64)
            g_list = np.load(os.path.join(npy_dir, 'g_list.npy')).astype(np.float64)
            ax_max = np.load(os.path.join(npy_dir, 'ax_max.npy')).astype(np.float64)
            ax_min = np.load(os.path.join(npy_dir, 'ax_min.npy')).astype(np.float64)
            ay_max = np.load(os.path.join(npy_dir, 'ay_max.npy')).astype(np.float64)
            gg_exp = np.load(os.path.join(npy_dir, 'gg_exponent.npy')).astype(np.float64)
        except (OSError, FileNotFoundError) as e:
            rospy.logerr(f"[GGTuner] gg.bin gen failed (npy missing): {e}")
            return False

        nv, ng = len(v_list), len(g_list)
        try:
            with open(bin_path, 'wb') as f:
                f.write(struct.pack('II', nv, ng))
                for arr in [v_list, g_list, ax_max, ax_min, ay_max, gg_exp]:
                    arr.tofile(f)
        except OSError as e:
            rospy.logerr(f"[GGTuner] gg.bin write failed: {e}")
            return False
        rospy.loginfo(
            f"[GGTuner] gg.bin written: {bin_path} (nv={nv}, ng={ng}, "
            f"size={os.path.getsize(bin_path)} B)")
        return True

    def _copy_to_gg_diagrams(self, vehicle_name):
        src = os.path.join(self.fast_ggv_output_dir, vehicle_name)
        dst = os.path.join(self.data_path, 'gg_diagrams', vehicle_name)
        if not os.path.exists(src):
            rospy.logerr(f"[GGTuner] fast_ggv output missing: {src}")
            return False
        meta_backup = None
        meta_path = os.path.join(dst, 'params_used.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta_backup = f.read()
            except OSError:
                pass
        if os.path.exists(dst) or os.path.islink(dst):
            if os.path.islink(dst):
                os.unlink(dst)
            else:
                shutil.rmtree(dst)
        shutil.copytree(src, dst)
        if meta_backup is not None:
            with open(meta_path, 'w') as f:
                f.write(meta_backup)
        rospy.loginfo(f"[GGTuner] Copied: {vehicle_name} → gg_diagrams/")
        for frame in ('velocity_frame', 'vehicle_frame'):
            frame_dir = os.path.join(dst, frame)
            if os.path.isdir(frame_dir):
                self._generate_gg_bin(frame_dir)
        return True

    # symlinks for DIRECTORIES only (params latest is a real file copy)
    def _update_dir_symlinks(self, vehicle_name):
        latest_name = f'{self.base_vehicle}_latest'
        targets = [
            (self.fast_ggv_output_dir, latest_name, vehicle_name),
            (os.path.join(self.data_path, 'gg_diagrams'), latest_name, vehicle_name),
        ]
        for parent, link_name, target in targets:
            link_path = os.path.join(parent, link_name)
            target_path = os.path.join(parent, target)
            if not os.path.exists(target_path):
                continue
            try:
                if os.path.islink(link_path) or os.path.exists(link_path):
                    os.unlink(link_path)
                os.symlink(target, link_path)
                rospy.loginfo(f"[GGTuner] symlink: {link_name} → {target}")
            except OSError as e:
                rospy.logwarn(f"[GGTuner] symlink failed: {e}")

    def _run_and_stream(self, cmd, tag, timeout=600, env=None):
        rospy.loginfo(f"[GGTuner] [{tag}] cmd: {' '.join(cmd)}")
        run_env = None
        if env is not None:
            run_env = os.environ.copy()
            run_env.update(env)
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=run_env)
        except (FileNotFoundError, OSError) as e:
            rospy.logerr(f"[GGTuner] [{tag}] Popen failed: {e}")
            return False
        start_time = time.time()
        try:
            for line in iter(proc.stdout.readline, ''):
                if time.time() - start_time > timeout:
                    rospy.logerr(f"[GGTuner] [{tag}] timeout ({timeout}s)")
                    proc.kill()
                    return False
                line = line.rstrip()
                if line:
                    rospy.loginfo(f"[{tag}] {line}")
        except Exception as e:
            rospy.logerr(f"[GGTuner] [{tag}] stream error: {e}")
            proc.kill()
            return False
        proc.wait()
        ok = (proc.returncode == 0)
        if ok:
            rospy.loginfo(f"[GGTuner] [{tag}] done (rc=0)")
        else:
            rospy.logerr(f"[GGTuner] [{tag}] failed (rc={proc.returncode})")
        return ok

    # fast_ggv — no --tuning, unified params yml already has everything
    def _run_fast_ggv(self, vehicle_name, full_resolution=False):
        rospy.loginfo(f"[GGTuner] [fast_ggv] starting: {vehicle_name}")
        self.status_pub.publish(f"GGV_COMPUTING: {vehicle_name}")
        if not os.path.exists(self.fast_ggv_script):
            rospy.logerr(f"[GGTuner] fast_ggv script missing: {self.fast_ggv_script}")
            return False
        resolution = '--full' if full_resolution else '--fast'
        cmd = ['bash', self.fast_ggv_script, vehicle_name, resolution]
        return self._run_and_stream(cmd, tag='fast_ggv', timeout=600)

    # raceline — passes safety_distance + g_weight from rqt
    def _run_raceline(self, vehicle_name, map_name, safety_distance=0.20,
                      g_weight=1.0,
                      # bridge centerline / heading-lock.
                      # Yaml at <map>/bridge_sectors.yaml stores
                      # *positions only* (start/end indices). The
                      # tuning knobs (pre/post extension, force_center,
                      # chi_max) come from rqt and are forwarded via
                      # roslaunch args to gen_global_racing_line.py.
                      # Missing yaml ⇒ silent no-op.
                      bridge_constraints_enable=True,
                      bridge_pre_m=0.0, bridge_post_m=0.0,
                      bridge_force_center=False, bridge_chi_max_rad=-1.0):
        if map_name not in self.available_maps:
            rospy.logerr(f"[GGTuner] map '{map_name}' not found. "
                         f"Available: {self.available_maps}")
            self.status_pub.publish(f"FAILED_RACELINE: invalid map")
            return False
        self.status_pub.publish(f"RACELINE_STARTED: {vehicle_name}")
        rospy.loginfo(f"[GGTuner] [raceline] map={map_name}, vehicle={vehicle_name}, "
                      f"safety_distance={safety_distance}, g_weight={g_weight:.3f}, "
                      f"bridge_constraints_enable={bridge_constraints_enable}, "
                      f"bridge_pre_m={bridge_pre_m:.2f}, "
                      f"bridge_post_m={bridge_post_m:.2f}, "
                      f"bridge_force_center={bridge_force_center}, "
                      f"bridge_chi_max_rad={bridge_chi_max_rad:.3f}")
        cmd = [
            'roslaunch', 'stack_master', '3d_global_line.launch',
            f'map:={map_name}',
            f'vehicle:={vehicle_name}',
            f'safety_distance:={safety_distance}',
            f'g_weight:={float(g_weight)}',
            # forward bridge args to run_pipeline.sh → gen_global_racing_line.py
            f'bridge_constraints_enable:={str(bool(bridge_constraints_enable)).lower()}',
            f'bridge_pre_m:={float(bridge_pre_m)}',
            f'bridge_post_m:={float(bridge_post_m)}',
            f'bridge_force_center:={str(bool(bridge_force_center)).lower()}',
            f'bridge_chi_max_rad:={float(bridge_chi_max_rad)}',
            'start_from:=5',
        ]
        ok = self._run_and_stream(cmd, tag='raceline', timeout=900)
        if ok:
            self.status_pub.publish(f"RACELINE_DONE: {vehicle_name}")
        else:
            self.status_pub.publish(f"FAILED_RACELINE: {vehicle_name}")
        return ok

    def _run_fbga_planner(self, vehicle_name, enable_mu=True, force_restart=False,
                          bridge_effect=1.0,
                          g_tilde_soft_beta=20.0,
                          smooth_mu=True, mu_smooth_window=21, mu_smooth_polyorder=3,
                          smooth_kappa=True, kappa_smooth_window=21, kappa_smooth_polyorder=3,
                          slope_brake_margin=0.0, slope_brake_vmax=5.0):
        rospy.loginfo(
            f"[GGTuner] [fbga] update: {vehicle_name}, enable_mu={enable_mu}, "
            f"bridge_effect={bridge_effect:.3f}, soft_beta={g_tilde_soft_beta:.1f}, "
            f"smooth_mu={smooth_mu}/win{mu_smooth_window}, "
            f"smooth_kappa={smooth_kappa}/win{kappa_smooth_window}, "
            f"brake_margin={slope_brake_margin:.2f}m, force_restart={force_restart}")
        self.status_pub.publish(f"FBGA_STARTED: {vehicle_name}")

        gg_bin = os.path.join(
            self.data_path, 'gg_diagrams', vehicle_name,
            'velocity_frame', 'gg.bin')
        params_yml = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + vehicle_name + '.yml')
        if not os.path.exists(params_yml):
            rospy.logerr(f"[GGTuner] params yml missing: {params_yml}")
            self.status_pub.publish(f"FAILED_FBGA: {vehicle_name}")
            return False

        rospy.set_param('/fbga_planner/gg_bin', gg_bin)
        rospy.set_param('/fbga_planner/params_yml', params_yml)
        rospy.set_param('/fbga_planner/enable_mu', bool(enable_mu))
        # forward all live FBGA knobs (picked up by reload_cb)
        rospy.set_param('/fbga_planner/bridge_effect', float(bridge_effect))
        rospy.set_param('/fbga_planner/g_tilde_soft_beta', float(g_tilde_soft_beta))
        rospy.set_param('/fbga_planner/smooth_mu', bool(smooth_mu))
        rospy.set_param('/fbga_planner/mu_smooth_window', int(mu_smooth_window))
        rospy.set_param('/fbga_planner/mu_smooth_polyorder', int(mu_smooth_polyorder))
        rospy.set_param('/fbga_planner/smooth_kappa', bool(smooth_kappa))
        rospy.set_param('/fbga_planner/kappa_smooth_window', int(kappa_smooth_window))
        rospy.set_param('/fbga_planner/kappa_smooth_polyorder', int(kappa_smooth_polyorder))
        rospy.set_param('/fbga_planner/slope_brake_margin', float(slope_brake_margin))
        rospy.set_param('/fbga_planner/slope_brake_vmax', float(slope_brake_vmax))

        if not force_restart:
            try:
                rospy.wait_for_service('/fbga/reload', timeout=0.5)
                reload_srv = rospy.ServiceProxy('/fbga/reload', Trigger)
                resp = reload_srv()
                if resp.success:
                    rospy.loginfo(
                        f"[GGTuner] [fbga] hot-reloaded: {resp.message}")
                    self.status_pub.publish(f"FBGA_DONE: {vehicle_name}")
                    return True
                else:
                    rospy.logwarn(
                        f"[GGTuner] [fbga] reload returned failure: "
                        f"{resp.message} → cold start")
            except (rospy.ROSException, rospy.ServiceException) as e:
                rospy.loginfo(
                    f"[GGTuner] [fbga] reload unavailable ({e}) → cold start")

        try:
            subprocess.run(['rosnode', 'kill', '/fbga_planner'],
                           capture_output=True, timeout=5, check=False)
            time.sleep(0.3)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        if self.fbga_proc is not None and self.fbga_proc.poll() is None:
            try:
                self.fbga_proc.terminate()
                self.fbga_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.fbga_proc.kill()
        self.fbga_proc = None

        cmd = [
            'rosrun', 'stack_master', 'fbga_velocity_planner.py',
            '__name:=fbga_planner',
            f'_gg_bin:={gg_bin}',
            f'_params_yml:={params_yml}',
            f'_enable_mu:={str(enable_mu).lower()}',
            # pass all FBGA knobs on cold start
            f'_bridge_effect:={float(bridge_effect)}',
            f'_g_tilde_soft_beta:={float(g_tilde_soft_beta)}',
            f'_smooth_mu:={str(smooth_mu).lower()}',
            f'_mu_smooth_window:={int(mu_smooth_window)}',
            f'_mu_smooth_polyorder:={int(mu_smooth_polyorder)}',
            f'_smooth_kappa:={str(smooth_kappa).lower()}',
            f'_kappa_smooth_window:={int(kappa_smooth_window)}',
            f'_kappa_smooth_polyorder:={int(kappa_smooth_polyorder)}',
            f'_slope_brake_margin:={float(slope_brake_margin)}',
            f'_slope_brake_vmax:={float(slope_brake_vmax)}',
        ]
        rospy.loginfo(f"[GGTuner] [fbga] cold start: {' '.join(cmd)}")
        try:
            self.fbga_proc = subprocess.Popen(cmd)
            return True
        except OSError as e:
            rospy.logerr(f"[GGTuner] Failed to start FBGA: {e}")
            self.status_pub.publish(f"FAILED_FBGA: {vehicle_name}")
            return False

    def _kill_fbga_planner(self):
        rospy.loginfo("[GGTuner] [fbga] kill requested (checkbox off)")
        try:
            subprocess.run(['rosnode', 'kill', '/fbga_planner'],
                           capture_output=True, timeout=5, check=False)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        if self.fbga_proc is not None and self.fbga_proc.poll() is None:
            try:
                self.fbga_proc.terminate()
                self.fbga_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.fbga_proc.kill()
        self.fbga_proc = None
        rospy.loginfo("[GGTuner] [fbga] killed")

    # sector_slicing spawn/kill (3d_sector_slicing.launch)
    # spawns the speed / overtaking / static-obstacle sector slicers for the
    # selected map. Trigger-style; close the GUI windows to finish.
    def _run_sector_slicing(self, map_name):
        if self.sector_slicing_proc is not None and self.sector_slicing_proc.poll() is None:
            rospy.loginfo("[GGTuner] [sectors] already running")
            return True
        if map_name not in self.available_maps:
            rospy.logerr(f"[GGTuner] [sectors] invalid map '{map_name}'")
            self.status_pub.publish(f"FAILED_SECTORS: invalid map")
            return False
        cmd = [
            'roslaunch', 'stack_master', '3d_sector_slicing.launch',
            f'map:={map_name}',
        ]
        rospy.loginfo(f"[GGTuner] [sectors] launching: {' '.join(cmd)}")
        try:
            self.sector_slicing_proc = subprocess.Popen(cmd)
            self.status_pub.publish(f"SECTORS_RUNNING: {map_name}")
            return True
        except OSError as e:
            rospy.logerr(f"[GGTuner] [sectors] launch failed: {e}")
            self.status_pub.publish(f"FAILED_SECTORS: {map_name}")
            return False

    def _kill_sector_slicing(self):
        rospy.loginfo("[GGTuner] [sectors] kill requested (checkbox off)")
        # kill the nodes in the launch
        for node in ('/sector_node_3d', '/ot_sector_node_3d',
                     '/static_obs_sector_node_3d', '/friction_sector_node'):
            try:
                subprocess.run(['rosnode', 'kill', node],
                               capture_output=True, timeout=3, check=False)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        if self.sector_slicing_proc is not None and self.sector_slicing_proc.poll() is None:
            try:
                self.sector_slicing_proc.terminate()
                self.sector_slicing_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.sector_slicing_proc.kill()
        self.sector_slicing_proc = None
        rospy.loginfo("[GGTuner] [sectors] killed")

    # apply_new_sectors — kill 5 sector consumer nodes and respawn via
    # 3d_sector_respawn.launch so a fresh sector slicing yaml takes
    # effect without restarting 3d_base_system.
    def _apply_new_sectors(self, map_name):
        if map_name not in self.available_maps:
            rospy.logerr(f"[GGTuner] [apply_sectors] invalid map '{map_name}'")
            self.status_pub.publish("FAILED_APPLY_SECTORS: invalid map")
            return False

        rospy.loginfo("[GGTuner] [apply_sectors] killing 5 nodes")
        targets = ('/dyn_sector_tuner/speed',
                   '/dyn_sector_tuner/overtake',
                   '/dyn_sector_tuner/static_obs',
                   '/velocity_scaler',
                   '/ot_interpolator')
        for n in targets:
            try:
                subprocess.run(['rosnode', 'kill', n],
                               capture_output=True, timeout=3, check=False)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        if self.sector_respawn_proc is not None and \
                self.sector_respawn_proc.poll() is None:
            try:
                self.sector_respawn_proc.terminate()
                self.sector_respawn_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.sector_respawn_proc.kill()
        self.sector_respawn_proc = None

        time.sleep(0.3) 

        # rebuild 3 sector_tuner_3d packages so the new yaml in
        # each pkg's cfg/ regenerates the dyn_reconfigure Config
        # that the respawned servers will import. Run in parallel
        # and wait for all to finish before respawn.
        rebuild_pkgs = ('sector_tuner_3d',
                        'overtaking_sector_tuner_3d',
                        'static_obstacle_sector_tuner_3d')
        rebuild_procs = []
        for pkg in rebuild_pkgs:
            try:
                pkg_path = rospkg.RosPack().get_path(pkg)
            except rospkg.ResourceNotFound:
                rospy.logerr(f"[GGTuner] [apply_sectors] pkg not found: {pkg}")
                continue
            shell = os.path.join(pkg_path, 'scripts/finish_sector_3d.sh')
            if not os.path.exists(shell):
                rospy.logerr(f"[GGTuner] [apply_sectors] missing: {shell}")
                continue
            rospy.loginfo(f"[GGTuner] [apply_sectors] rebuild: {pkg}")
            rebuild_procs.append(subprocess.Popen([shell]))
        for p in rebuild_procs:
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                rospy.logerr("[GGTuner] [apply_sectors] rebuild timeout")
                p.kill()

        cmd = ['roslaunch', 'stack_master', '3d_sector_respawn.launch',
               f'map:={map_name}']
        rospy.loginfo(f"[GGTuner] [apply_sectors] respawn: {' '.join(cmd)}")
        try:
            self.sector_respawn_proc = subprocess.Popen(cmd)
            self.status_pub.publish(f"SECTORS_APPLIED: {map_name}")
            return True
        except OSError as e:
            rospy.logerr(f"[GGTuner] [apply_sectors] launch failed: {e}")
            self.status_pub.publish(f"FAILED_APPLY_SECTORS: {map_name}")
            return False

    # VelOpt-only save/load — scoped to vo_* keys only AND per-map.
    # Single file at stack_master/maps/<map>/velopt_params.yml, overwritten
    # on each save (no version history). Map is read from the D_Raceline
    # `map` slider (config.map).
    def _velopt_params_path(self, map_name):
        if not map_name:
            return None
        d = os.path.join(self.maps_dir, map_name)
        if not os.path.isdir(d):
            return None
        return os.path.join(d, 'velopt_params.yml')

    def _save_velopt_params_yaml(self, config):
        map_name = str(getattr(config, 'map', '') or '')
        fpath = self._velopt_params_path(map_name)
        if fpath is None:
            rospy.logwarn(
                f"[GGTuner] save_velopt_params: invalid map '{map_name}' "
                f"(maps/{map_name} not found)")
            self.status_pub.publish(
                f"VELOPT_PARAMS_SAVE_BAD_MAP: {map_name}")
            return None
        params = {}
        for k, v in dict(config).items():
            if not k.startswith('vo_'):
                continue
            if isinstance(v, bool):
                params[k] = bool(v)
            elif isinstance(v, (int, float)):
                params[k] = float(v) if isinstance(v, float) else int(v)
            else:
                params[k] = v
        if not params:
            rospy.logwarn("[GGTuner] save_velopt_params: no vo_* keys in cfg")
            self.status_pub.publish("VELOPT_PARAMS_SAVE_EMPTY")
            return None
        payload = {
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'map': map_name,
            'source': '/gg_tuner dynamic_reconfigure (velopt-only)',
            'params': params,
        }
        with open(fpath, 'w') as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=True)
        rospy.loginfo(
            f"[GGTuner] save_velopt_params: {len(params)} keys → {fpath}")
        self.status_pub.publish(f"VELOPT_PARAMS_SAVED_{map_name}")
        return fpath

    def _load_velopt_params_yaml_into(self, config):
        map_name = str(getattr(config, 'map', '') or '')
        fpath = self._velopt_params_path(map_name)
        if fpath is None:
            rospy.logwarn(
                f"[GGTuner] load_velopt_params: invalid map '{map_name}' "
                f"(maps/{map_name} not found)")
            self.status_pub.publish(
                f"VELOPT_PARAMS_LOAD_BAD_MAP: {map_name}")
            return None
        if not os.path.isfile(fpath):
            rospy.logwarn(
                f"[GGTuner] load_velopt_params: {fpath} not found")
            self.status_pub.publish(
                f"VELOPT_PARAMS_LOAD_NONE: {map_name}")
            return None
        with open(fpath) as f:
            payload = yaml.safe_load(f) or {}
        params = payload.get('params') if isinstance(payload, dict) else None
        if not isinstance(params, dict) or not params:
            rospy.logwarn(
                f"[GGTuner] load_velopt_params: no params field in {fpath}")
            self.status_pub.publish(
                f"VELOPT_PARAMS_LOAD_EMPTY: {map_name}")
            return fpath
        applied = 0
        skipped = 0
        for k, v in params.items():
            if not k.startswith('vo_'):
                continue
            try:
                config[k] = v
                applied += 1
            except (KeyError, TypeError) as e:
                rospy.logwarn(
                    f"[GGTuner] load_velopt_params: skip {k} "
                    f"({type(v).__name__}): {e}")
                skipped += 1
        rospy.loginfo(
            f"[GGTuner] load_velopt_params: {fpath} → "
            f"applied={applied} skipped={skipped}")
        self.status_pub.publish(
            f"VELOPT_PARAMS_LOADED_{map_name}_n{applied}")
        return fpath

    # bridge_tuner spawn/kill — mirrors _run_sector_slicing pattern.
    # Launches bridge_sector_tuner_3d/bridge_tuner.launch which opens a
    # matplotlib GUI on the *smoothed centerline CSV* (pre-opt). User
    # marks bridge start/end and the GUI saves bridge_sectors.yaml.
    # Tuning sliders live in the rqt Bridge group; yaml is the single
    # source of truth. Kill is invoked when the rqt trigger auto-clears
    # (we keep the function for an explicit shutdown path).
    def _run_bridge_tuner(self, map_name):
        if self.bridge_tuner_proc is not None and \
                self.bridge_tuner_proc.poll() is None:
            rospy.loginfo("[GGTuner] [bridge] already running")
            return True
        if map_name not in self.available_maps:
            rospy.logerr(f"[GGTuner] [bridge] invalid map '{map_name}'")
            self.status_pub.publish(f"FAILED_BRIDGE: invalid map")
            return False
        cmd = [
            'roslaunch', 'bridge_sector_tuner_3d', 'bridge_tuner.launch',
            f'map:={map_name}',
        ]
        rospy.loginfo(f"[GGTuner] [bridge] launching: {' '.join(cmd)}")
        try:
            self.bridge_tuner_proc = subprocess.Popen(cmd)
            self.status_pub.publish(f"BRIDGE_TUNER_RUNNING: {map_name}")
            return True
        except OSError as e:
            rospy.logerr(f"[GGTuner] [bridge] launch failed: {e}")
            self.status_pub.publish(f"FAILED_BRIDGE: {map_name}")
            return False

    def _kill_bridge_tuner(self):
        rospy.loginfo("[GGTuner] [bridge] kill requested (checkbox off)")
        try:
            subprocess.run(['rosnode', 'kill', '/bridge_sector_node_3d'],
                           capture_output=True, timeout=3, check=False)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        if self.bridge_tuner_proc is not None and \
                self.bridge_tuner_proc.poll() is None:
            try:
                self.bridge_tuner_proc.terminate()
                self.bridge_tuner_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.bridge_tuner_proc.kill()
        self.bridge_tuner_proc = None
        rospy.loginfo("[GGTuner] [bridge] killed")

    # safety slicing GUI — auto-clear trigger (bridge_tuner pattern).
    # Spawns safety_tuner.launch (slicing only). User marks sectors,
    # clicks Done → yaml + pkg cfg + rebuild. GUI exits on its own.
    def _run_safety_sector_tuner(self, map_name):
        if self.safety_tuner_proc is not None and \
                self.safety_tuner_proc.poll() is None:
            rospy.loginfo("[GGTuner] [safety-tuner] already running")
            return True
        if map_name not in self.available_maps:
            rospy.logerr(f"[GGTuner] [safety-tuner] invalid map '{map_name}'")
            self.status_pub.publish(f"FAILED_SAFETY_TUNER: invalid map")
            return False
        cmd = [
            'roslaunch', 'safety_sector_tuner_3d', 'safety_tuner.launch',
            f'map:={map_name}',
        ]
        rospy.loginfo(f"[GGTuner] [safety-tuner] launching: {' '.join(cmd)}")
        try:
            self.safety_tuner_proc = subprocess.Popen(cmd)
            self.status_pub.publish(f"SAFETY_TUNER_RUNNING: {map_name}")
            return True
        except OSError as e:
            rospy.logerr(f"[GGTuner] [safety-tuner] launch failed: {e}")
            self.status_pub.publish(f"FAILED_SAFETY_TUNER: {map_name}")
            return False

    # safety sector setting — persistent toggle, server only.
    # Spawns safety_setting.launch (dyn cfg server under
    # /dyn_sector_tuner/safety). Edge-detected ON/OFF.
    def _start_safety_sector_setting(self, map_name):
        if self.safety_setting_proc is not None and \
                self.safety_setting_proc.poll() is None:
            rospy.loginfo("[GGTuner] [safety-setting] already running")
            return True
        if map_name not in self.available_maps:
            rospy.logerr(f"[GGTuner] [safety-setting] invalid map '{map_name}'")
            self.status_pub.publish(f"FAILED_SAFETY_SETTING: invalid map")
            return False
        cmd = [
            'roslaunch', 'safety_sector_tuner_3d', 'safety_setting.launch',
            f'map:={map_name}',
        ]
        rospy.loginfo(f"[GGTuner] [safety-setting] launching: {' '.join(cmd)}")
        try:
            self.safety_setting_proc = subprocess.Popen(cmd)
            self.status_pub.publish(f"SAFETY_SETTING_ON: {map_name}")
            return True
        except OSError as e:
            rospy.logerr(f"[GGTuner] [safety-setting] launch failed: {e}")
            self.status_pub.publish(f"FAILED_SAFETY_SETTING: {map_name}")
            return False

    def _stop_safety_sector_setting(self):
        rospy.loginfo("[GGTuner] [safety-setting] stop requested (toggle off)")
        try:
            subprocess.run(['rosnode', 'kill', '/dyn_sector_tuner/safety'],
                           capture_output=True, timeout=3, check=False)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        if self.safety_setting_proc is not None and \
                self.safety_setting_proc.poll() is None:
            try:
                self.safety_setting_proc.terminate()
                self.safety_setting_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.safety_setting_proc.kill()
        self.safety_setting_proc = None
        # also wipe the rosparam tree so a later safety_sector_use=ON
        # with setting=OFF cleanly falls through to the yaml-on-disk
        # path (instead of seeing stale rosparam from this session).
        try:
            rospy.delete_param('/safety_sector_params')
        except KeyError:
            pass
        self.status_pub.publish("SAFETY_SETTING_OFF")
        rospy.loginfo("[GGTuner] [safety-setting] stopped")

    # friction sector — clone of safety pattern.
    def _run_friction_sector_tuner(self, map_name):
        if self.friction_tuner_proc is not None and \
                self.friction_tuner_proc.poll() is None:
            rospy.loginfo("[GGTuner] [friction-tuner] already running")
            return True
        if map_name not in self.available_maps:
            rospy.logerr(f"[GGTuner] [friction-tuner] invalid map '{map_name}'")
            self.status_pub.publish(f"FAILED_FRICTION_TUNER: invalid map")
            return False
        cmd = [
            'roslaunch', 'friction_sector_tuner', 'friction_tuner.launch',
            f'map:={map_name}',
        ]
        rospy.loginfo(f"[GGTuner] [friction-tuner] launching: {' '.join(cmd)}")
        try:
            self.friction_tuner_proc = subprocess.Popen(cmd)
            self.status_pub.publish(f"FRICTION_TUNER_RUNNING: {map_name}")
            return True
        except OSError as e:
            rospy.logerr(f"[GGTuner] [friction-tuner] launch failed: {e}")
            self.status_pub.publish(f"FAILED_FRICTION_TUNER: {map_name}")
            return False

    def _start_friction_sector_setting(self, map_name):
        if self.friction_setting_proc is not None and \
                self.friction_setting_proc.poll() is None:
            rospy.loginfo("[GGTuner] [friction-setting] already running")
            return True
        if map_name not in self.available_maps:
            rospy.logerr(f"[GGTuner] [friction-setting] invalid map '{map_name}'")
            self.status_pub.publish(f"FAILED_FRICTION_SETTING: invalid map")
            return False
        cmd = [
            'roslaunch', 'friction_sector_tuner', 'friction_setting.launch',
            f'map:={map_name}',
        ]
        rospy.loginfo(f"[GGTuner] [friction-setting] launching: {' '.join(cmd)}")
        try:
            self.friction_setting_proc = subprocess.Popen(cmd)
            self.status_pub.publish(f"FRICTION_SETTING_ON: {map_name}")
            return True
        except OSError as e:
            rospy.logerr(f"[GGTuner] [friction-setting] launch failed: {e}")
            self.status_pub.publish(f"FAILED_FRICTION_SETTING: {map_name}")
            return False

    def _stop_friction_sector_setting(self):
        rospy.loginfo("[GGTuner] [friction-setting] stop requested (toggle off)")
        try:
            subprocess.run(['rosnode', 'kill', '/dyn_sector_tuner/friction'],
                           capture_output=True, timeout=3, check=False)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        if self.friction_setting_proc is not None and \
                self.friction_setting_proc.poll() is None:
            try:
                self.friction_setting_proc.terminate()
                self.friction_setting_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.friction_setting_proc.kill()
        self.friction_setting_proc = None
        # wipe rosparam so use=ON without setting falls through to yaml.
        try:
            rospy.delete_param('/friction_map_params')
        except KeyError:
            pass
        self.status_pub.publish("FRICTION_SETTING_OFF")
        rospy.loginfo("[GGTuner] [friction-setting] stopped")

    # Stage 4 alternative — 3d_optimized_vel_planner (NLP velocity).
    # Mirrors fbga pattern: hot-reload via /velopt/reload, fallback cold start.
    # bridge_effect is shared with fbga (Track3D inside velopt takes
    # g_weight=bridge_effect → mu / Omega scale handled in track3D.py).
    def _run_velopt_planner(self, vehicle_name, map_name, velopt_opts,
                            bridge_effect=1.0, force_restart=False,
                            # per-point friction scaling toggle.
                            # When True, base GGV is scaled in the NLP
                            # by friction_sector mu — no per-friction
                            # GGV directories required.
                            use_friction_scaling=False):
        rospy.loginfo(
            f"[GGTuner] [velopt] update: vehicle={vehicle_name}, map={map_name}, "
            f"opts={velopt_opts}, bridge_effect={bridge_effect:.3f}, "
            f"use_friction_scaling={use_friction_scaling}, "
            f"force_restart={force_restart}")
        self.status_pub.publish(f"VELOPT_STARTED: {vehicle_name}")

        params_yml = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + vehicle_name + '.yml')
        if not os.path.exists(params_yml):
            rospy.logerr(f"[GGTuner] params yml missing: {params_yml}")
            self.status_pub.publish(f"FAILED_VELOPT: {vehicle_name}")
            return False
        raceline_csv = os.path.join(
            self.maps_dir, map_name,
            f'{map_name}_3d_{vehicle_name}_timeoptimal.csv')
        if not os.path.exists(raceline_csv):
            rospy.logerr(f"[GGTuner] raceline csv missing: {raceline_csv}")
            self.status_pub.publish(f"FAILED_VELOPT: {vehicle_name}")
            return False

        rospy.set_param('/vel_opt_3d/map',           map_name)
        rospy.set_param('/vel_opt_3d/racecar',       vehicle_name)
        rospy.set_param('/vel_opt_3d/V_min',         float(velopt_opts['V_min']))
        rospy.set_param('/vel_opt_3d/gg_margin',     float(velopt_opts['gg_margin']))
        # step_size_opt no longer set from rqt (velopt uses its default)
        rospy.set_param('/vel_opt_3d/w_T',           float(velopt_opts['w_T']))
        rospy.set_param('/vel_opt_3d/w_jx',          float(velopt_opts['w_jx']))
        # asymmetric jerk + tanh-saturated curvature/transition weighting
        rospy.set_param('/vel_opt_3d/w_jx_acc_ratio',      float(velopt_opts['w_jx_acc_ratio']))
        rospy.set_param('/vel_opt_3d/w_jx_brk_ratio',      float(velopt_opts['w_jx_brk_ratio']))
        rospy.set_param('/vel_opt_3d/w_jx_curv_alpha_acc', float(velopt_opts['w_jx_curv_alpha_acc']))
        rospy.set_param('/vel_opt_3d/w_jx_curv_alpha_brk', float(velopt_opts['w_jx_curv_alpha_brk']))
        rospy.set_param('/vel_opt_3d/w_jx_curv_k_alpha',   float(velopt_opts['w_jx_curv_k_alpha']))
        rospy.set_param('/vel_opt_3d/w_jx_curv_beta_acc',  float(velopt_opts['w_jx_curv_beta_acc']))
        rospy.set_param('/vel_opt_3d/w_jx_curv_beta_brk',  float(velopt_opts['w_jx_curv_beta_brk']))
        rospy.set_param('/vel_opt_3d/w_jx_curv_k_beta',    float(velopt_opts['w_jx_curv_k_beta']))
        # legacy w_jx_curv_alpha removed from rqt; vel_opt_3d default 0 → fallback inactive.
        # direct ax_pos^2 corner penalty
        rospy.set_param('/vel_opt_3d/w_ax_corner_acc',     float(velopt_opts['w_ax_corner_acc']))
        rospy.set_param('/vel_opt_3d/w_ax_corner_k',       float(velopt_opts['w_ax_corner_k']))
        rospy.set_param('/vel_opt_3d/bridge_effect',       float(bridge_effect))
        # per-point friction scaling toggle (bypasses multi-GGV dir).
        rospy.set_param('/vel_opt_3d/use_friction_scaling',
                        bool(use_friction_scaling))
        # extra per-direction mu scale. Default 1.0 ≡ legacy isotropic
        # scaling. Re-read by vel_opt every hot-reload / cold-start.
        # rospy.set_param('/vel_opt_3d/mu_scale_x',
        #                 float(velopt_opts.get('mu_scale_x', 1.0)))
        # rospy.set_param('/vel_opt_3d/mu_scale_y',
        #                 float(velopt_opts.get('mu_scale_y', 1.0)))

        if not force_restart:
            try:
                rospy.wait_for_service('/velopt/reload', timeout=0.5)
                reload_srv = rospy.ServiceProxy('/velopt/reload', Trigger)
                resp = reload_srv()
                if resp.success:
                    rospy.loginfo(
                        f"[GGTuner] [velopt] hot-reloaded: {resp.message}")
                    self.status_pub.publish(f"VELOPT_DONE: {vehicle_name}")
                    return True
                else:
                    rospy.logwarn(
                        f"[GGTuner] [velopt] reload returned failure: "
                        f"{resp.message} → cold start")
            except (rospy.ROSException, rospy.ServiceException) as e:
                rospy.loginfo(
                    f"[GGTuner] [velopt] reload unavailable ({e}) → cold start")

        try:
            subprocess.run(['rosnode', 'kill', '/vel_opt_3d'],
                           capture_output=True, timeout=5, check=False)
            time.sleep(0.3)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        if self.velopt_proc is not None and self.velopt_proc.poll() is None:
            try:
                self.velopt_proc.terminate()
                self.velopt_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.velopt_proc.kill()
        self.velopt_proc = None

        cmd = [
            'rosrun', 'stack_master', '3d_optimized_vel_planner.py',
            '__name:=vel_opt_3d',
            f'_map:={map_name}',
            f'_racecar:={vehicle_name}',
            f'_V_min:={float(velopt_opts["V_min"])}',
            f'_gg_margin:={float(velopt_opts["gg_margin"])}',
            # step_size_opt no longer overridden; velopt node default used
            f'_w_T:={float(velopt_opts["w_T"])}',
            f'_w_jx:={float(velopt_opts["w_jx"])}',
            # asymmetric jerk + tanh-saturated curvature/transition weighting
            f'_w_jx_acc_ratio:={float(velopt_opts["w_jx_acc_ratio"])}',
            f'_w_jx_brk_ratio:={float(velopt_opts["w_jx_brk_ratio"])}',
            f'_w_jx_curv_alpha_acc:={float(velopt_opts["w_jx_curv_alpha_acc"])}',
            f'_w_jx_curv_alpha_brk:={float(velopt_opts["w_jx_curv_alpha_brk"])}',
            f'_w_jx_curv_k_alpha:={float(velopt_opts["w_jx_curv_k_alpha"])}',
            f'_w_jx_curv_beta_acc:={float(velopt_opts["w_jx_curv_beta_acc"])}',
            f'_w_jx_curv_beta_brk:={float(velopt_opts["w_jx_curv_beta_brk"])}',
            f'_w_jx_curv_k_beta:={float(velopt_opts["w_jx_curv_k_beta"])}',
            # legacy w_jx_curv_alpha launch arg removed; node default 0 → fallback inactive.
            # direct ax_pos^2 corner penalty
            f'_w_ax_corner_acc:={float(velopt_opts["w_ax_corner_acc"])}',
            f'_w_ax_corner_k:={float(velopt_opts["w_ax_corner_k"])}',
            f'_bridge_effect:={float(bridge_effect)}',
            # per-point friction scaling toggle
            f'_use_friction_scaling:={str(bool(use_friction_scaling)).lower()}',
            # extra per-direction mu scale
            # f'_mu_scale_x:={float(velopt_opts.get("mu_scale_x", 1.0))}',
            # f'_mu_scale_y:={float(velopt_opts.get("mu_scale_y", 1.0))}',
        ]
        rospy.loginfo(f"[GGTuner] [velopt] cold start: {' '.join(cmd)}")
        try:
            self.velopt_proc = subprocess.Popen(cmd)
            return True
        except OSError as e:
            rospy.logerr(f"[GGTuner] Failed to start velopt: {e}")
            self.status_pub.publish(f"FAILED_VELOPT: {vehicle_name}")
            return False

    def _kill_velopt_planner(self):
        rospy.loginfo("[GGTuner] [velopt] kill requested")
        try:
            subprocess.run(['rosnode', 'kill', '/vel_opt_3d'],
                           capture_output=True, timeout=5, check=False)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        if self.velopt_proc is not None and self.velopt_proc.poll() is None:
            try:
                self.velopt_proc.terminate()
                self.velopt_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.velopt_proc.kill()
        self.velopt_proc = None
        rospy.loginfo("[GGTuner] [velopt] killed")

    # publish GGV diamond results as JSON for rqt_gg_viewer
    def _publish_gg_results(self, vehicle_name, tuning=None):
        vf_dir = os.path.join(self.data_path, 'gg_diagrams', vehicle_name,
                              'velocity_frame')
        if not os.path.isdir(vf_dir):
            rospy.logwarn(f"[GGTuner] publish_gg_results: no velocity_frame dir: {vf_dir}")
            return
        try:
            v_list = np.load(os.path.join(vf_dir, 'v_list.npy')).tolist()
            g_list = np.load(os.path.join(vf_dir, 'g_list.npy')).tolist()
            ax_max = np.load(os.path.join(vf_dir, 'ax_max.npy')).tolist()
            ax_min = np.load(os.path.join(vf_dir, 'ax_min.npy')).tolist()
            ay_max = np.load(os.path.join(vf_dir, 'ay_max.npy')).tolist()
            gg_exp = np.load(os.path.join(vf_dir, 'gg_exponent.npy')).tolist()
        except (FileNotFoundError, OSError) as e:
            rospy.logwarn(f"[GGTuner] publish_gg_results: npy load failed: {e}")
            return
        payload = {
            'vehicle_name': vehicle_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'v_list': v_list,
            'g_list': g_list,
            'diamond': {
                'ax_max': ax_max,
                'ax_min': ax_min,
                'ay_max': ay_max,
                'gg_exponent': gg_exp,
            },
            'tuning_params': tuning or {},
        }
        self.results_pub.publish(json.dumps(payload))
        rospy.loginfo(f"[GGTuner] Published /gg_results "
                      f"(V={len(v_list)}, g={len(g_list)})")

    def _run_full_pipeline(self, tuning, run_opts):
        with self.pipeline_lock:
            try:
                rospy.loginfo(f"[GGTuner] ===== Pipeline start =====")
                rospy.loginfo(f"[GGTuner] tuning: {tuning}")
                rospy.loginfo(f"[GGTuner] options: {run_opts}")
                self.status_pub.publish(f"STARTED: {self.base_vehicle}")

                vehicle_name = f"{self.base_vehicle}_latest"

                if not run_opts['run_ggv']:
                    latest_gg = os.path.join(
                        self.data_path, 'gg_diagrams', vehicle_name)
                    if not os.path.exists(latest_gg):
                        rospy.logerr(
                            f"[GGTuner] latest gg_diagrams missing: {latest_gg} "
                            f"(run_ggv=False)")
                        self.status_pub.publish("FAILED: no latest")
                        return
                    rospy.loginfo(
                        f"[GGTuner] Stage 2 SKIP (run_ggv=False), reusing latest")
                    self.status_pub.publish(f"GGV_SKIP: {vehicle_name}")
                    # publish existing results for rqt_gg_viewer
                    self._publish_gg_results(vehicle_name, tuning)
                else:
                    rospy.loginfo(
                        f"[GGTuner] Stage 2: compute fresh into latest "
                        f"(friction={tuning.get('friction', '?')})")
                    merged = self._merge_all_params(tuning)
                    self._write_latest_params_yml(merged)
                    ok = self._run_fast_ggv(
                        vehicle_name,
                        full_resolution=run_opts['full_resolution'])
                    if not ok:
                        self.status_pub.publish(f"FAILED_GGV: {vehicle_name}")
                        return
                    if not self._copy_to_gg_diagrams(vehicle_name):
                        self.status_pub.publish(f"FAILED_GGV: {vehicle_name}")
                        return
                    self._save_meta(vehicle_name, tuning)
                    self.status_pub.publish(f"GGV_DONE: {vehicle_name}")
                    # publish fresh results for rqt_gg_viewer
                    self._publish_gg_results(vehicle_name, tuning)

                # ---- Stage 3: raceline (optional) ----
                if run_opts['regen_raceline']:
                    # overlay rqt RACELINE_KEYS onto latest yml (idempotent)
                    self._update_raceline_keys_in_yml(vehicle_name, tuning)
                    # safety sector master switch — surfaced as a
                    # rosparam so gen_global_racing_line._build_safety_table
                    # can short-circuit to legacy uniform safety_distance
                    # when False. Set right before the launch so
                    # gen_global_racing_line sees the current rqt value.
                    rospy.set_param('/safety_sector_params/use',
                                    bool(run_opts['safety_sector_use']))
                    # raceline keeps the old g_weight signature; map from bridge_effect
                    # also thread bridge constraints + tuning from rqt
                    # so gen_global_racing_line.py can read both yaml
                    # (positions) and rqt (global knobs). Defaults in
                    # _run_raceline reproduce legacy behaviour if any
                    # bridge_* keys are absent.
                    ok = self._run_raceline(
                        vehicle_name, run_opts['map'],
                        safety_distance=run_opts['safety_distance'],
                        g_weight=run_opts['bridge_effect'],
                        bridge_constraints_enable=run_opts[
                            'bridge_constraints_enable'],
                        bridge_pre_m=run_opts['bridge_pre_m'],
                        bridge_post_m=run_opts['bridge_post_m'],
                        bridge_force_center=run_opts['bridge_force_center'],
                        bridge_chi_max_rad=run_opts['bridge_chi_max_rad'])
                    if not ok:
                        rospy.logerr(f"[GGTuner] raceline failed")
                        return
                else:
                    rospy.loginfo(f"[GGTuner] raceline regen SKIP")

                # Stage 4 single-engine dispatch (vel_engine selector)
                engine = run_opts.get('vel_engine', 'none')
                force_restart = bool(run_opts['regen_raceline'])
                if engine == 'fbga':
                    self._kill_velopt_planner()       # ensure other engine off
                    ok = self._run_fbga_planner(
                        vehicle_name,
                        enable_mu=run_opts['enable_mu'],
                        force_restart=force_restart,
                        bridge_effect=run_opts['bridge_effect'],
                        g_tilde_soft_beta=run_opts['g_tilde_soft_beta'],
                        smooth_mu=run_opts['smooth_mu'],
                        mu_smooth_window=run_opts['mu_smooth_window'],
                        mu_smooth_polyorder=run_opts['mu_smooth_polyorder'],
                        smooth_kappa=run_opts['smooth_kappa'],
                        kappa_smooth_window=run_opts['kappa_smooth_window'],
                        kappa_smooth_polyorder=run_opts['kappa_smooth_polyorder'],
                        slope_brake_margin=run_opts['slope_brake_margin'],
                        slope_brake_vmax=run_opts['slope_brake_vmax'])
                    if not ok:
                        return
                    self.status_pub.publish(f"DONE_ALL: {vehicle_name}")
                elif engine == 'velopt':
                    self._kill_fbga_planner()         # ensure other engine off
                    # rqt vo_* keys → velopt_opts dict (internal keys
                    # match _run_velopt_planner's ROS-param-style names).
                    velopt_opts = {
                        'V_min':                run_opts['vo_Vmin'],
                        'gg_margin':            run_opts['vo_gg_marg'],
                        # step_size removed; velopt node uses fixed default
                        'w_T':                  run_opts['vo_w_T'],
                        'w_jx':                 run_opts['vo_w_jx'],
                        'w_jx_acc_ratio':       run_opts['vo_acc_r'],
                        'w_jx_brk_ratio':       run_opts['vo_brk_r'],
                        # tanh-saturated curvature/transition (alpha + beta)
                        'w_jx_curv_alpha_acc':  run_opts['vo_alpha_acc'],
                        'w_jx_curv_alpha_brk':  run_opts['vo_alpha_brk'],
                        'w_jx_curv_k_alpha':    run_opts['vo_k_alpha'],
                        'w_jx_curv_beta_acc':   run_opts['vo_beta_acc'],
                        'w_jx_curv_beta_brk':   run_opts['vo_beta_brk'],
                        'w_jx_curv_k_beta':     run_opts['vo_k_beta'],
                        # legacy w_jx_curv_alpha key removed.
                        # direct ax_pos^2 corner penalty
                        'w_ax_corner_acc':      run_opts['vo_w_axc'],
                        'w_ax_corner_k':        run_opts['vo_k_axc'],
                        # extra per-direction mu scale slots (currently
                        # disabled; vel_opt reads them as rosparam each reload).
                        # 'mu_scale_x':           run_opts['vo_mu_scale_x'],
                        # 'mu_scale_y':           run_opts['vo_mu_scale_y'],
                    }
                    ok = self._run_velopt_planner(
                        vehicle_name, run_opts['map'], velopt_opts,
                        bridge_effect=run_opts['vo_bridge'],
                        force_restart=force_restart,
                        # per-point friction scaling toggle
                        use_friction_scaling=run_opts[
                            'vo_use_friction_scaling'])
                    if not ok:
                        return
                    self.status_pub.publish(f"DONE_ALL: {vehicle_name}")
                else:  # 'none'
                    self._kill_fbga_planner()
                    self._kill_velopt_planner()
                    rospy.loginfo(f"[GGTuner] vel_engine=none → both engines off")
                    self.status_pub.publish(f"DONE_VEL_OFF: {vehicle_name}")

                rospy.loginfo(f"[GGTuner] ===== Pipeline done: {vehicle_name} =====")

            except Exception as e:
                rospy.logerr(f"[GGTuner] Pipeline exception: {e}")
                import traceback
                rospy.logerr(traceback.format_exc())
                self.status_pub.publish(f"EXCEPTION: {str(e)[:100]}")
                
    def reconfigure_cb(self, config, level):

        # OR-merge per-group Apply mirrors with the root `apply`. All
        # trigger the same pipeline; we clear them together at the end so
        # rqt's checkbox auto-resets regardless of which one was clicked.
        # 'apply_fbga' left in the tuple so re-enabling FBGA needs no
        # change here. dict.get() (not getattr) is the correct fallback for
        # dynamic_reconfigure.Config — getattr can't catch its KeyError.
        APPLY_KEYS = ('apply', 'apply_ggv', 'apply_raceline',
                      'apply_fbga', 'apply_velopt',
                      # Bridge group apply mirror
                      'apply_bridge',
                      # Safety group apply mirror
                      'apply_safety',
                      # Friction group apply mirror
                      'apply_friction')
        apply_any = any(bool(config.get(k, False)) for k in APPLY_KEYS)

        # save_ggv (renamed from save_now) — snapshots latest GGV to v<N+1>.
        if getattr(config, 'save_ggv', False):
            if self.pipeline_thread is not None and self.pipeline_thread.is_alive():
                rospy.logwarn(
                    "[GGTuner] SAVE_GGV ignored: pipeline running")
            else:
                try:
                    self._snapshot_latest_to_version()
                except Exception as e:
                    rospy.logerr(f"[GGTuner] SAVE_GGV exception: {e}")
                    self.status_pub.publish(
                        f"SAVE_GGV_EXCEPTION: {str(e)[:80]}")
            config.save_ggv = False
            if not apply_any:
                return config

        # VelOpt-only save/load triggers — scoped to vo_* sliders so
        # velopt presets are decoupled from ggv/raceline/fbga/bridge.
        # Same auto-clear + early-return-when-no-apply pattern.
        if getattr(config, 'save_velopt_params', False):
            try:
                self._save_velopt_params_yaml(config)
            except Exception as e:
                rospy.logerr(
                    f"[GGTuner] save_velopt_params exception: {e}")
                self.status_pub.publish(
                    f"SAVE_VELOPT_PARAMS_EXCEPTION: {str(e)[:80]}")
            config.save_velopt_params = False
            if not apply_any:
                return config

        if getattr(config, 'load_velopt_params', False):
            try:
                self._load_velopt_params_yaml_into(config)
            except Exception as e:
                rospy.logerr(
                    f"[GGTuner] load_velopt_params exception: {e}")
                self.status_pub.publish(
                    f"LOAD_VELOPT_PARAMS_EXCEPTION: {str(e)[:80]}")
            config.load_velopt_params = False
            if not apply_any:
                return config

        # sector_slicing / bridge_tuner are now *trigger* checkboxes
        # (mirroring apply/save_ggv): clicking True spawns the GUI and
        # the box auto-returns to False. The matplotlib GUIs own their
        # own lifecycle — user clicks Done or closes the window to
        # terminate, no separate "uncheck to kill" step. This avoids
        # the prior stale-toggle confusion where leaving the box True
        # across sessions silently re-spawned the GUI on next launch.
        # Already-running spawn requests are no-ops (logged).
        if bool(getattr(config, 'run_sector_slicing', False)):
            self._run_sector_slicing(str(config.map))
            config.run_sector_slicing = False
            self._last_sector_slicing_state = False

        # apply_new_sectors — auto-clear trigger (kill+respawn 5 nodes)
        if bool(getattr(config, 'apply_new_sectors', False)):
            self._apply_new_sectors(str(config.map))
            config.apply_new_sectors = False
            if not apply_any:
                return config
        if bool(getattr(config, 'run_bridge_tuner', False)):
            self._run_bridge_tuner(str(config.map))
            config.run_bridge_tuner = False
            self._last_bridge_tuner_state = False
            if not apply_any:
                return config
        # safety sector tuner — auto-clear trigger (bridge_tuner pattern).
        # Spawns slicing GUI only; the dyn cfg server lives on a
        # separate toggle (safety_sector_setting). Checkbox returns to
        # False right after spawn.
        if bool(getattr(config, 'safety_sector_tuner', False)):
            self._run_safety_sector_tuner(str(config.map))
            config.safety_sector_tuner = False
            if not apply_any:
                return config
        # safety sector setting — persistent toggle, edge-detected.
        # ON edge ⇒ safety_setting.launch (server only — slicing GUI
        # is on a separate trigger above). OFF edge ⇒ kill server and
        # wipe rosparam tree so use=ON without setting falls through
        # to yaml-on-disk.
        cur_safety_setting = bool(getattr(config, 'safety_sector_setting',
                                          False))
        if cur_safety_setting != self._last_safety_setting_state:
            if cur_safety_setting:
                self._start_safety_sector_setting(str(config.map))
            else:
                self._stop_safety_sector_setting()
            self._last_safety_setting_state = cur_safety_setting
            if not apply_any:
                return config

        # friction sector tuner / setting — clone of safety pattern.
        if bool(getattr(config, 'friction_sector_tuner', False)):
            self._run_friction_sector_tuner(str(config.map))
            config.friction_sector_tuner = False
            if not apply_any:
                return config
        cur_friction_setting = bool(getattr(config, 'friction_sector_setting',
                                            False))
        if cur_friction_setting != self._last_friction_setting_state:
            if cur_friction_setting:
                self._start_friction_sector_setting(str(config.map))
            else:
                self._stop_friction_sector_setting()
            self._last_friction_setting_state = cur_friction_setting
            if not apply_any:
                return config

        # if the user switched maps in rqt, re-sync bridge sliders from
        # the new map's yaml. Done here (in reconfigure_cb) because
        # this is the only callback that sees `config.map` changes.
        cur_map = str(getattr(config, 'map', '') or '')
        if cur_map and cur_map != self._last_bridge_yaml_map:
            # pass config= so the helper mutates it in place instead
            # of calling srv.update_configuration (which would
            # deadlock — we're already inside the server's lock).
            self._sync_bridge_cfg_from_yaml(cur_map, config=config)
            if not apply_any:
                return config
        # safety yaml sync is handled by safety_sector_server.py
        # (callback fires on every reconfigure). gg_tuner doesn't need
        # to mirror map changes — the server reads /map on its own.

        if not apply_any:
            return config

        # refuse concurrent runs
        if self.pipeline_thread is not None and self.pipeline_thread.is_alive():
            rospy.logwarn("[GGTuner] Pipeline already running — ignoring apply")
            for k in APPLY_KEYS:
                setattr(config, k, False)
            return config

        # collect ALL tuning parameters
        tuning = {k: config[k] for k in self.ALL_TUNING_KEYS}

        run_opts = {
            'run_ggv':          bool(config.run_ggv),
            'full_resolution':  bool(config.full_resolution),
            'regen_raceline':   bool(config.regen_raceline),
            'map':              str(config.map),
            # Stage 4 single selector (replaces run_fbga bool)
            'vel_engine':       str(config.vel_engine),
            'enable_mu':        bool(config.enable_mu),
            'safety_distance':  float(config.safety_distance),
            # FBGA knobs (used when vel_engine='fbga'; bridge_effect also shared by velopt)
            'bridge_effect':            float(config.bridge_effect),
            'g_tilde_soft_beta':        float(config.g_tilde_soft_beta),
            'smooth_mu':                bool(config.smooth_mu),
            'mu_smooth_window':         int(config.mu_smooth_window),
            'mu_smooth_polyorder':      int(config.mu_smooth_polyorder),
            'smooth_kappa':             bool(config.smooth_kappa),
            'kappa_smooth_window':      int(config.kappa_smooth_window),
            'kappa_smooth_polyorder':   int(config.kappa_smooth_polyorder),
            'slope_brake_margin':       float(config.slope_brake_margin),
            'slope_brake_vmax':         float(config.slope_brake_vmax),
            # VelOpt knobs (rqt names shortened velopt_* → vo_*).
            'vo_bridge':         float(config.vo_bridge),
            'vo_Vmin':           float(config.vo_Vmin),
            'vo_gg_marg':        float(config.vo_gg_marg),
            # velopt_step_size removed (fixed at velopt node default 0.2 m)
            'vo_w_T':            float(config.vo_w_T),
            'vo_w_jx':           float(config.vo_w_jx),
            'vo_acc_r':          float(config.vo_acc_r),
            'vo_brk_r':          float(config.vo_brk_r),
            # tanh-saturated curvature/transition weighting (alpha + beta)
            'vo_alpha_acc':      float(config.vo_alpha_acc),
            'vo_alpha_brk':      float(config.vo_alpha_brk),
            'vo_k_alpha':        float(config.vo_k_alpha),
            'vo_beta_acc':       float(config.vo_beta_acc),
            'vo_beta_brk':       float(config.vo_beta_brk),
            'vo_k_beta':         float(config.vo_k_beta),
            # legacy w_jx_curv_alpha slider removed (rqt-side).
            # direct ax_pos^2 corner penalty
            'vo_w_axc':          float(config.vo_w_axc),
            'vo_k_axc':          float(config.vo_k_axc),
            # per-point friction scaling toggle (velopt NLP layer).
            # When True the base GGV is scaled in-NLP by mu_scale_k[k]
            # from /friction_map_params/Sector{i}/friction; multi-GGV
            # directory lookup is bypassed.
            'vo_use_friction_scaling': bool(config.vo_use_friction_scaling),
            # extra per-direction scale on top of sector mu (live yaml).
            # Default 1.0 ≡ legacy isotropic scaling. Re-read from
            # rosparam by vel_opt every reload, so this updates each
            # apply with no node restart needed.
            # 'vo_mu_scale_x': float(config.vo_mu_scale_x),
            # 'vo_mu_scale_y': float(config.vo_mu_scale_y),
            # bridge centerline / heading-lock toggle (raceline NLP).
            # Bridge zones (start/end indices) live in
            # <map>/bridge_sectors.yaml; the *tuning* lives here in
            # rqt and is forwarded as roslaunch args so a single yaml
            # can be tuned without reopening the GUI. Missing yaml ⇒
            # no-op (silent).
            'bridge_constraints_enable': bool(config.bridge_constraints_enable),
            'bridge_pre_m':              float(config.bridge_pre_m),
            'bridge_post_m':             float(config.bridge_post_m),
            'bridge_force_center':       bool(config.bridge_force_center),
            'bridge_chi_max_rad':        float(config.bridge_chi_max_rad),
            # safety sector master switch (bridge_constraints_enable
            # analogue). Per-sector values live in /safety_sector_params/...
            # rosparam tree (owned by safety_sector_server.py); this flag
            # gates whether gen_global_racing_line reads them at all.
            'safety_sector_use': bool(config.safety_sector_use),
        }

        # validate map
        if run_opts['regen_raceline'] and run_opts['map'] not in self.available_maps:
            rospy.logerr(f"[GGTuner] Invalid map '{run_opts['map']}'. "
                         f"Available: {self.available_maps}")
            self.status_pub.publish(f"FAILED_RACELINE: invalid map")
            for k in APPLY_KEYS:
                setattr(config, k, False)
            return config

        # persist current bridge sliders to <map>/bridge_sectors.yaml
        # on every apply. yaml is the single source of truth — the
        # raceline NLP reads tuning back from it (positions + tuning
        # both come from yaml, not CLI args, for the per-bridge case).
        self._dump_bridge_cfg_to_yaml(run_opts)
        # safety sector dump is handled by safety_sector_server.py's
        # save_params trigger (in rqt /dyn_sector_tuner/safety). gg_tuner
        # doesn't write the yaml.

        # background thread
        self.pipeline_thread = threading.Thread(
            target=self._run_full_pipeline,
            args=(tuning, run_opts),
            daemon=True)
        self.pipeline_thread.start()
        rospy.loginfo("[GGTuner] Pipeline spawned in background thread")

        # clear all Apply mirrors so rqt resets every checkbox
        for k in APPLY_KEYS:
            setattr(config, k, False)
        return config

    def _shutdown_cleanup(self):
        if self.fbga_proc is not None and self.fbga_proc.poll() is None:
            rospy.loginfo("[GGTuner] Terminating FBGA subprocess...")
            try:
                self.fbga_proc.terminate()
                self.fbga_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.fbga_proc.kill()
        if self.velopt_proc is not None and self.velopt_proc.poll() is None:
            rospy.loginfo("[GGTuner] Terminating velopt subprocess...")
            try:
                self.velopt_proc.terminate()
                self.velopt_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.velopt_proc.kill()
        if self.sector_slicing_proc is not None and self.sector_slicing_proc.poll() is None:
            rospy.loginfo("[GGTuner] Terminating sector_slicing roslaunch...")
            try:
                self.sector_slicing_proc.terminate()
                self.sector_slicing_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.sector_slicing_proc.kill()
        # also clean up sector_respawn roslaunch
        if self.sector_respawn_proc is not None and \
                self.sector_respawn_proc.poll() is None:
            rospy.loginfo("[GGTuner] Terminating sector_respawn roslaunch...")
            try:
                self.sector_respawn_proc.terminate()
                self.sector_respawn_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.sector_respawn_proc.kill()
        # also clean up bridge_tuner roslaunch
        if self.bridge_tuner_proc is not None and \
                self.bridge_tuner_proc.poll() is None:
            rospy.loginfo("[GGTuner] Terminating bridge_tuner roslaunch...")
            try:
                self.bridge_tuner_proc.terminate()
                self.bridge_tuner_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.bridge_tuner_proc.kill()
        # safety slicing GUI cleanup (auto-clear trigger may still
        # have a live subprocess if user hasn't clicked Done yet).
        if self.safety_tuner_proc is not None and \
                self.safety_tuner_proc.poll() is None:
            rospy.loginfo("[GGTuner] Terminating safety_tuner roslaunch...")
            try:
                self.safety_tuner_proc.terminate()
                self.safety_tuner_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.safety_tuner_proc.kill()
        # safety sector setting (dyn cfg server) cleanup.
        if self.safety_setting_proc is not None and \
                self.safety_setting_proc.poll() is None:
            rospy.loginfo("[GGTuner] Terminating safety_setting roslaunch...")
            try:
                self.safety_setting_proc.terminate()
                self.safety_setting_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.safety_setting_proc.kill()


if __name__ == '__main__':
    rospy.init_node("gg_tuner")
    node = GGTunerNode()
    rospy.spin()
