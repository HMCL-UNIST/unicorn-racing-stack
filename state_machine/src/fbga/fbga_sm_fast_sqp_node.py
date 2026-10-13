#!/usr/bin/env python3
# FBGA SM with fast_sqp_planner-owned dynamic obstacle handling.
# - GB_TRACK / OVERTAKE / TRAILING transitions use fast_sqp_planner_transitions.
# - OVERTAKE local_wpnts source = fast_sqp's /planner/avoidance/otwpnts (raw, not
#   via cur_avoidance_wpnts pipeline that 3d_sm needs _check_availability for).
# - TRAILING target falls back to nearest obstacle when cur_*_wpnts.closest_target
#   is None (slim SM style).
# - RECOVERY / START / SMART_STATIC / FTGONLY unchanged (parent 3d_sm logic).
# - FBGA vx overlay still applies on OVERTAKE (parent __main__ calls _fbga_step).

import os
import sys
import importlib.util

import numpy as np
import rospy

from rospkg import RosPack
from f110_msgs.msg import Wpnt


_SM_SRC = os.path.join(RosPack().get_path('state_machine'), 'src')
if _SM_SRC not in sys.path:
    sys.path.insert(0, _SM_SRC)
_FAST_SQP_SRC = os.path.join(_SM_SRC, 'fast_sqp_planner')
if _FAST_SQP_SRC not in sys.path:
    sys.path.insert(0, _FAST_SQP_SRC)

_fbga_path = os.path.join(_SM_SRC, 'fbga', 'fbga_state_machine_node.py')
_spec = importlib.util.spec_from_file_location('fbga_sm', _fbga_path)
fbga_sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fbga_sm)

from states_types import StateType
import fast_sqp_planner_transitions as fast_sqp_t


class FBGASMFastSQP(fbga_sm.FBGAStateMachine):
    """FBGA SM + fast_sqp owning dynamic obstacle decisions in OVERTAKE/TRAILING."""

    def __init__(self, name):
        super().__init__(name)

        # params used by fast_sqp transitions
        self.ot_engage_ttc_s = rospy.get_param("state_machine/ot_engage_ttc_s", 1.0)
        self.ot_engage_min_m = rospy.get_param("state_machine/ot_engage_min_m", 5.0)
        self.ot_margin_safe_m = rospy.get_param("state_machine/ot_margin_safe_m", 0.15)
        self.ot_blocking_s_gap_m = rospy.get_param("state_machine/ot_blocking_s_gap_m", 0.5)
        self.ot_collision_ttc_s = rospy.get_param("state_machine/ot_collision_ttc_s", 0.5)
        self.overtaking_ttl_sec = rospy.get_param("state_machine/overtaking_ttl_sec", 0.5)
        self.overtaking_ttl_count = 0
        self.overtaking_ttl_count_threshold = int(self.overtaking_ttl_sec * self.rate_hz)

        # swap dynamic obstacle transitions to fast_sqp
        self.state_transitions[StateType.GB_TRACK] = fast_sqp_t.GlobalTrackingTransition
        self.state_transitions[StateType.OVERTAKE] = fast_sqp_t.OvertakingTransition
        self.state_transitions[StateType.TRAILING] = fast_sqp_t.TrailingTransition

        # OVERTAKE / RECOVERY handlers use raw subscriber data, bypassing
        # 3d_sm's cur_*_wpnts pipeline (which depends on _check_availability hook).
        self.states[StateType.OVERTAKE] = self._overtake_state_handler
        self.states[StateType.RECOVERY] = self._recovery_state_handler

    # adapter — fast_sqp transitions expect List[Wpnt] in _ot_fast_sqp_wpnts.
    @property
    def _ot_fast_sqp_wpnts(self):
        if self.avoidance_wpnts is None:
            return None
        wpnts = self.avoidance_wpnts.wpnts
        return list(wpnts) if len(wpnts) > 0 else None

    # OVERTAKE local_wpnts — slice fast_sqp path from nearest s, pad with GB.
    def _overtake_state_handler(self, _sm_unused):
        ot_list = self._ot_fast_sqp_wpnts
        if ot_list is None or len(ot_list) == 0:
            # Fallback: no OT path → GB.
            return self.states[StateType.GB_TRACK](self) if StateType.GB_TRACK in self.states else []

        s_arr = np.array([w.s_m for w in ot_list])
        dists = np.abs(
            (s_arr - self.cur_s + self.track_length / 2) % self.track_length
            - self.track_length / 2)
        start_idx = int(np.argmin(dists))
        ot_local = list(ot_list[start_idx:])
        last_s = ot_local[-1].s_m if ot_local else self.cur_s
        gb_start = int(last_s / self.waypoints_dist + 1) % self.num_glb_wpnts
        while len(ot_local) < self.n_loc_wpnts:
            ot_local.append(self.cur_gb_wpnts.list[gb_start % self.num_glb_wpnts])
            gb_start += 1
        out = ot_local[:self.n_loc_wpnts]
        # FBGA vx overlay (rqt v_scale / gg_scale for 'avoidance' mode).
        self._fbga_apply_latch(out, 'avoidance')
        return out

    # RECOVERY local_wpnts — slice raw recovery_wpnts from nearest s, pad with GB.
    def _recovery_state_handler(self, _sm_unused):
        rec = self.recovery_wpnts
        if rec is None or len(rec.wpnts) == 0:
            return self.states[StateType.GB_TRACK](self) if StateType.GB_TRACK in self.states else []
        rec_list = list(rec.wpnts)
        s_arr = np.array([w.s_m for w in rec_list])
        dists = np.abs(
            (s_arr - self.cur_s + self.track_length / 2) % self.track_length
            - self.track_length / 2)
        start_idx = int(np.argmin(dists))
        local = list(rec_list[start_idx:])
        last_s = local[-1].s_m if local else self.cur_s
        gb_start = int(last_s / self.waypoints_dist + 1) % self.num_glb_wpnts
        while len(local) < self.n_loc_wpnts:
            local.append(self.cur_gb_wpnts.list[gb_start % self.num_glb_wpnts])
            gb_start += 1
        out = local[:self.n_loc_wpnts]
        # FBGA vx overlay (rqt v_scale / gg_scale for 'recovery' mode).
        self._fbga_apply_latch(out, 'recovery')
        return out

    # TRAILING target — slim SM fallback: nearest forward obstacle when
    # parent's closest_target pipeline is empty (3d_sm needs _check_availability
    # to populate cur_*_wpnts.closest_target, which fast_sqp transitions skip).
    def get_farthest_target(self, local_wpnts_src):
        targets, src = super().get_farthest_target(local_wpnts_src)
        if len(targets) == 0 and len(self.cur_obstacles_in_interest) > 0:
            closest = min(self.cur_obstacles_in_interest,
                          key=lambda o: (o.s_center - self.cur_s) % self.track_length)
            return [closest], src
        return targets, src

    # fast_sqp transition helpers
    def _check_ot_path_collision(self) -> bool:
        wpnts = self._ot_fast_sqp_wpnts
        if wpnts is None or len(wpnts) == 0:
            return False
        ot_s = np.array([w.s_m for w in wpnts])
        ot_d = np.array([w.d_m for w in wpnts])
        half_width = 0.15
        ot_collision_horizon = np.clip(
            self.cur_vs * self.ot_collision_ttc_s, 2.0, self.gb_horizon_m)
        for obs in self.obstacles:
            if obs.is_static:
                continue
            gap = (obs.s_center - self.cur_s) % self.track_length
            if gap > ot_collision_horizon:
                continue
            s_dists = np.abs(
                (ot_s - obs.s_center + self.track_length / 2) % self.track_length
                - self.track_length / 2)
            idx = int(np.argmin(s_dists))
            d_var = getattr(obs, 'd_var', 0.0)
            sigma = np.sqrt(max(d_var, 0.0))
            obs_half = obs.size / 2.0 + 2.0 * sigma
            if abs(ot_d[idx] - obs.d_center) < obs_half + half_width:
                return True
        return False

    def _get_ot_path_min_margin(self) -> float:
        wpnts = self._ot_fast_sqp_wpnts
        if wpnts is None or len(wpnts) == 0:
            return float('inf')
        ot_s = np.array([w.s_m for w in wpnts])
        ot_d = np.array([w.d_m for w in wpnts])
        half_width = 0.15
        ot_collision_horizon = np.clip(
            self.cur_vs * self.ot_collision_ttc_s, 2.0, self.gb_horizon_m)
        min_margin = float('inf')
        for obs in self.obstacles:
            if obs.is_static:
                continue
            gap = (obs.s_center - self.cur_s) % self.track_length
            if gap > ot_collision_horizon:
                continue
            s_dists = np.abs(
                (ot_s - obs.s_center + self.track_length / 2) % self.track_length
                - self.track_length / 2)
            idx = int(np.argmin(s_dists))
            d_var = getattr(obs, 'd_var', 0.0)
            sigma = np.sqrt(max(d_var, 0.0))
            obs_half = obs.size / 2.0 + 2.0 * sigma
            margin = abs(ot_d[idx] - obs.d_center) - obs_half - half_width
            min_margin = min(min_margin, margin)
        return min_margin


if __name__ == "__main__":
    name = "state_machine"
    rospy.init_node(name, anonymous=False, log_level=rospy.WARN)

    sm = FBGASMFastSQP(name)

    rospy.on_shutdown(sm.on_shutdown)

    loop_rate = rospy.Rate(sm.rate_hz)
    while not rospy.is_shutdown():
        sm._fbga_step()
        sm.loop()
        loop_rate.sleep()
