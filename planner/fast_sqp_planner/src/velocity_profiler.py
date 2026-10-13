from __future__ import annotations

import os

import numpy as np
from vel_planner_25d.vel_planner import calc_vel_profile
import trajectory_planning_helpers as tph

class VelocityProfiler:

    def __init__(self,
                 stack_master_cfg_dir: str,
                 racecar_version: str,
                 m_veh: float = 3.0,
                 drag_coeff: float = 0.01,
                 dyn_model_exp: float = 1.0):
        self.m_veh = m_veh
        self.drag_coeff = drag_coeff
        self.dyn_model_exp = dyn_model_exp

        veh_dyn_dir = os.path.join(stack_master_cfg_dir, racecar_version,
                                   'veh_dyn_info')
        # keep paths so reload() can re-read CSVs at runtime
        self._ggv_path = os.path.join(veh_dyn_dir, 'ggv.csv')
        self._ax_path = os.path.join(veh_dyn_dir, 'ax_max_machines.csv')
        self._b_ax_path = os.path.join(veh_dyn_dir, 'b_ax_max_machines.csv')
        self.reload()

    # re-read CSVs from disk so rqt reload_ggv applies in 2_5d/3d modes
    def reload(self) -> None:
        self.ggv, self.ax_max_machines = tph.import_veh_dyn_info.\
            import_veh_dyn_info(ggv_import_path=self._ggv_path,
                                ax_max_machines_import_path=self._ax_path)
        _, self.b_ax_max_machines = tph.import_veh_dyn_info.\
            import_veh_dyn_info(ggv_import_path=self._ggv_path,
                                ax_max_machines_import_path=self._b_ax_path)
        self.v_max_default = float(min(self.ggv[-1, 0],
                                       self.ax_max_machines[-1, 0]))

    # add slope param for vel_planner_25d gravity correction
    def profile(self,
                kappa: np.ndarray,
                el_lengths: np.ndarray,
                v_start: float,
                v_end: float | None = None,
                v_max: float | None = None,
                mu: np.ndarray | None = None,
                loc_gg: np.ndarray | None = None,
                slope: np.ndarray | None = None,
                filt_window: int | None = None) -> np.ndarray:

        if v_max is None:
            v_max = self.v_max_default
        if v_start is None or v_start < 0.0:
            v_start = 0.0
        kwargs = dict(
            ax_max_machines=self.ax_max_machines,
            b_ax_max_machines=self.b_ax_max_machines,
            kappa=kappa,
            el_lengths=el_lengths,
            closed=False,
            drag_coeff=self.drag_coeff,
            m_veh=self.m_veh,
            v_max=v_max,
            v_start=v_start,
            v_end=v_end,
            dyn_model_exp=self.dyn_model_exp,
            filt_window=filt_window,
        )
        if loc_gg is not None:
            kwargs['loc_gg'] = loc_gg
        else:
            kwargs['ggv'] = self.ggv
            if mu is not None:
                kwargs['mu'] = mu
        if slope is not None:
            kwargs['slope'] = slope
        return calc_vel_profile(**kwargs)
