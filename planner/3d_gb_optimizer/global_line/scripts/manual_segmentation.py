import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import sys
import os
import json
import threading

class GroundWallSelectorApp:
    def __init__(self, pcd_path, seed_radius=0.3, z_tolerance=0.15, z_min=None, z_max=None):
        self.seed_radius = seed_radius
        self.save_dir = os.path.dirname(os.path.abspath(pcd_path))

        # Load & z-filter & downsample
        print(f"Loading {pcd_path}...")
        self.pcd_full = o3d.io.read_point_cloud(pcd_path)
        print(f"Loaded {len(self.pcd_full.points)} points")

        if z_min is not None or z_max is not None:
            pts = np.asarray(self.pcd_full.points)
            mask = np.ones(len(pts), dtype=bool)
            if z_min is not None:
                mask &= pts[:, 2] >= z_min
            if z_max is not None:
                mask &= pts[:, 2] <= z_max
            self.pcd_full = self.pcd_full.select_by_index(np.where(mask)[0])
            print(f"Z filter [{z_min}, {z_max}]: {mask.sum()} / {len(pts)} points kept")

        voxel_size = 0.05
        self.pcd = self.pcd_full.voxel_down_sample(voxel_size)
        self.pts = np.asarray(self.pcd.points)
        print(f"Downsampled to {len(self.pts)} points (voxel={voxel_size}m)")

        self.tree = o3d.geometry.KDTreeFlann(self.pcd)

        # Estimate normals (hybrid search is faster than pure KNN for large clouds)
        print("Estimating normals...")
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=20))
        # Orient normals upward (z+) — fast O(N) vs tangent_plane's O(N²)
        self.pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0.0, 0.0, 100.0]))
        self.normals = np.asarray(self.pcd.normals)
        print("Normals computed")

        # State
        self.ground_mask = np.zeros(len(self.pts), dtype=bool)
        self.wall_mask = np.zeros(len(self.pts), dtype=bool)
        self.history = []
        self.picking_mode = False
        self.growing = False
        self.mode = "ground"     # "ground", "wall", "erase", "delete"
        self.deleted_mask = np.zeros(len(self.pts), dtype=bool)  # permanently deleted

        # Per-mode presets
        self.presets = {
            "ground": {"normal_thresh": 0.8, "z_tol": 0.15, "z_band": 0.3},
            "wall":   {"normal_thresh": 0.5, "z_tol": 0.3,  "z_band": 0.5},
        }

        # Drag state
        self.drag_start = None
        self.dragging = False

        # Height colormap
        z = self.pts[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)
        self.base_colors = np.zeros((len(z), 3))
        self.base_colors[:, 0] = np.clip(2.0 * z_norm - 0.5, 0, 1)
        self.base_colors[:, 1] = np.clip(1.0 - np.abs(2.0 * z_norm - 1.0), 0, 1)
        self.base_colors[:, 2] = np.clip(1.0 - 2.0 * z_norm, 0, 1)
        self.base_colors = 0.3 + 0.7 * self.base_colors
        self._apply_colors()

        self._setup_gui()

    def _apply_colors(self):
        colors = self.base_colors.copy()
        colors[self.ground_mask & ~self.deleted_mask] = [0.0, 1.0, 0.0]
        colors[self.wall_mask & ~self.deleted_mask] = [1.0, 0.3, 0.0]
        colors[self.deleted_mask] = [0.15, 0.15, 0.15]  # near-invisible (match bg)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def _get_preset(self):
        return self.presets[self.mode]

    def _sync_ui_from_preset(self):
        p = self._get_preset()
        self.normal_slider.int_value = int(p["normal_thresh"] * 100)
        self.normal_label.text = f"Normal similarity: {p['normal_thresh']:.2f}"
        self.ztol_slider.int_value = int(p["z_tol"] * 100)
        self.ztol_label.text = f"Z tolerance: {p['z_tol']:.2f} m"
        self.zband_slider.int_value = int(p["z_band"] * 100)
        self.zband_label.text = f"Z band: {p['z_band']:.2f} m"

    def _sync_preset_from_ui(self):
        if self.mode not in self.presets:
            return
        p = self.presets[self.mode]
        p["normal_thresh"] = self.normal_slider.int_value / 100.0
        p["z_tol"] = self.ztol_slider.int_value / 100.0
        p["z_band"] = self.zband_slider.int_value / 100.0

    def _on_normal_changed(self, val):
        v = int(val) / 100.0
        self.normal_label.text = f"Normal similarity: {v:.2f}"
        self._sync_preset_from_ui()

    def _on_ztol_changed(self, val):
        v = int(val) / 100.0
        self.ztol_label.text = f"Z tolerance: {v:.2f} m"
        self._sync_preset_from_ui()

    def _on_zband_changed(self, val):
        v = int(val) / 100.0
        self.zband_label.text = f"Z band: {v:.2f} m"
        self._sync_preset_from_ui()

    def _on_radius_changed(self, val):
        self.seed_radius = int(val) / 100.0
        self.radius_label.text = f"Grow radius: {self.seed_radius:.2f} m"

    def _setup_gui(self):
        app = gui.Application.instance
        app.initialize()

        self.window = app.create_window("Ground & Wall Selector", 1400, 900)
        em = self.window.theme.font_size

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background_color([0.15, 0.15, 0.15, 1.0])

        self.mat = rendering.Material()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 3.0

        self.scene_widget.scene.add_geometry("pcd", self.pcd, self.mat)
        bounds = self.pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())

        panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))

        panel.add_child(gui.Label("=== Ground & Wall Selector ==="))
        panel.add_fixed(0.3 * em)
        panel.add_child(gui.Label("Click: grow from seed"))
        panel.add_child(gui.Label("Drag: select rectangle"))
        panel.add_fixed(em)

        self.status_label = gui.Label("Ground: 0 | Wall: 0")
        panel.add_child(self.status_label)
        panel.add_fixed(em)

        self.mode_btn = gui.Button("Mode: GROUND")
        self.mode_btn.set_on_clicked(self._toggle_mode)
        panel.add_child(self.mode_btn)
        panel.add_fixed(0.3 * em)

        self.pick_btn = gui.Button("Pick Mode: OFF")
        self.pick_btn.set_on_clicked(self._toggle_pick)
        panel.add_child(self.pick_btn)
        panel.add_fixed(0.5 * em)

        self.info_label = gui.Label("Camera mode")
        panel.add_child(self.info_label)
        panel.add_fixed(em)

        undo_btn = gui.Button("Undo")
        undo_btn.set_on_clicked(self._on_undo)
        panel.add_child(undo_btn)
        panel.add_fixed(0.3 * em)

        reset_btn = gui.Button("Reset All")
        reset_btn.set_on_clicked(self._on_reset)
        panel.add_child(reset_btn)
        panel.add_fixed(0.3 * em)

        save_btn = gui.Button("Save")
        save_btn.set_on_clicked(self._on_save)
        panel.add_child(save_btn)
        panel.add_fixed(em)

        self.settings_title = gui.Label("--- GROUND Settings ---")
        panel.add_child(self.settings_title)
        panel.add_fixed(0.3 * em)

        self.normal_label = gui.Label(f"Normal similarity: {self.presets['ground']['normal_thresh']:.2f}")
        panel.add_child(self.normal_label)
        self.normal_slider = gui.Slider(gui.Slider.INT)
        self.normal_slider.set_limits(0, 100)  # 0.00 ~ 1.00
        self.normal_slider.int_value = int(self.presets["ground"]["normal_thresh"] * 100)
        self.normal_slider.set_on_value_changed(self._on_normal_changed)
        panel.add_child(self.normal_slider)
        panel.add_fixed(0.3 * em)

        self.ztol_label = gui.Label(f"Z tolerance: {self.presets['ground']['z_tol']:.2f} m")
        panel.add_child(self.ztol_label)
        self.ztol_slider = gui.Slider(gui.Slider.INT)
        self.ztol_slider.set_limits(1, 100)  # 0.01 ~ 1.00
        self.ztol_slider.int_value = int(self.presets["ground"]["z_tol"] * 100)
        self.ztol_slider.set_on_value_changed(self._on_ztol_changed)
        panel.add_child(self.ztol_slider)
        panel.add_fixed(0.3 * em)

        self.zband_label = gui.Label(f"Z band: {self.presets['ground']['z_band']:.2f} m")
        panel.add_child(self.zband_label)
        self.zband_slider = gui.Slider(gui.Slider.INT)
        self.zband_slider.set_limits(1, 200)  # 0.01 ~ 2.00
        self.zband_slider.int_value = int(self.presets["ground"]["z_band"] * 100)
        self.zband_slider.set_on_value_changed(self._on_zband_changed)
        panel.add_child(self.zband_slider)
        panel.add_fixed(em)

        self.radius_label = gui.Label(f"Grow radius: {self.seed_radius:.2f} m")
        panel.add_child(self.radius_label)
        self.radius_slider = gui.Slider(gui.Slider.INT)
        self.radius_slider.set_limits(5, 200)  # 0.05 ~ 2.00
        self.radius_slider.int_value = int(self.seed_radius * 100)
        self.radius_slider.set_on_value_changed(self._on_radius_changed)
        panel.add_child(self.radius_slider)
        panel.add_fixed(0.3 * em)

        panel.add_child(gui.Label("Point size:"))
        self.psize_slider = gui.Slider(gui.Slider.INT)
        self.psize_slider.set_limits(1, 10)
        self.psize_slider.int_value = 3
        self.psize_slider.set_on_value_changed(self._on_psize_changed)
        panel.add_child(self.psize_slider)

        self.scene_widget.set_on_mouse(self._on_mouse)

        self.panel = panel
        self.window.add_child(self.scene_widget)
        self.window.add_child(panel)
        self.window.set_on_layout(self._on_layout)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        pw = 280
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = gui.Rect(r.x + r.width - pw, r.y, pw, r.height)

    def _toggle_mode(self):
        # Save current UI values to current mode's preset
        if self.mode in self.presets:
            self._sync_preset_from_ui()
        # Cycle: ground -> wall -> erase -> delete -> ground
        modes = ["ground", "wall", "erase", "delete"]
        labels = {
            "ground": ("Mode: GROUND", "--- GROUND Settings ---"),
            "wall":   ("Mode: WALL",   "--- WALL Settings ---"),
            "erase":  ("Mode: ERASE (unassign)", "--- ERASE (no grow) ---"),
            "delete": ("Mode: DELETE (remove)", "--- DELETE (no grow) ---"),
        }
        idx = modes.index(self.mode) if self.mode in modes else 0
        self.mode = modes[(idx + 1) % len(modes)]
        btn_text, title_text = labels[self.mode]
        self.mode_btn.text = btn_text
        self.settings_title.text = title_text
        # Load preset if applicable
        if self.mode in self.presets:
            self._sync_ui_from_preset()
        if self.picking_mode:
            self.info_label.text = f"Click/drag: {self.mode}!"

    def _toggle_pick(self):
        self.picking_mode = not self.picking_mode
        if self.picking_mode:
            self.pick_btn.text = "Pick Mode: ON"
            self.info_label.text = f"Click/drag on {self.mode}!"
        else:
            self.pick_btn.text = "Pick Mode: OFF"
            self.info_label.text = "Camera mode"
            self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _screen_to_3d(self, sx, sy, depth_image):
        """Unproject screen pixel to 3D world coordinate using depth buffer."""
        depth = np.asarray(depth_image)
        lx = int(sx - self.scene_widget.frame.x)
        ly = int(sy - self.scene_widget.frame.y)
        if lx < 0 or ly < 0 or lx >= depth.shape[1] or ly >= depth.shape[0]:
            return None
        d = depth[ly, lx]
        if d >= 1.0:
            return None

        cam = self.scene_widget.scene.camera
        view_mat = np.array(cam.get_view_matrix())
        proj_mat = np.array(cam.get_projection_matrix())
        w = self.scene_widget.frame.width
        h = self.scene_widget.frame.height

        ndc_x = 2.0 * lx / w - 1.0
        ndc_y = 1.0 - 2.0 * ly / h
        ndc_z = 2.0 * d - 1.0

        clip = np.array([ndc_x, ndc_y, ndc_z, 1.0])
        proj_inv = np.linalg.inv(proj_mat)
        view_inv = np.linalg.inv(view_mat)

        eye = proj_inv @ clip
        eye /= eye[3]
        world = view_inv @ eye
        return world[:3]

    def _project_to_screen(self):
        """Project all points to screen coordinates. Returns (N, 2) array of (sx, sy)."""
        cam = self.scene_widget.scene.camera
        view_mat = np.array(cam.get_view_matrix())
        proj_mat = np.array(cam.get_projection_matrix())
        w = self.scene_widget.frame.width
        h = self.scene_widget.frame.height
        fx = self.scene_widget.frame.x
        fy = self.scene_widget.frame.y

        # Homogeneous coordinates
        pts_h = np.hstack([self.pts, np.ones((len(self.pts), 1))])  # (N, 4)
        # View transform
        view_pts = (view_mat @ pts_h.T).T  # (N, 4)
        # Projection
        clip_pts = (proj_mat @ view_pts.T).T  # (N, 4)
        # Perspective divide
        w_clip = clip_pts[:, 3:4]
        w_clip[w_clip == 0] = 1e-10
        ndc = clip_pts[:, :3] / w_clip

        # NDC to screen
        screen = np.zeros((len(self.pts), 2))
        screen[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * w + fx
        screen[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * h + fy

        # Also get depth for front-face check
        depths = ndc[:, 2]
        return screen, depths

    def _on_mouse(self, event):
        if not self.picking_mode:
            return gui.SceneWidget.EventCallbackResult.IGNORED
        if self.growing:
            return gui.SceneWidget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
            self.drag_start = (event.x, event.y)
            self.dragging = False
            return gui.SceneWidget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.DRAG and self.drag_start is not None:
            dx = abs(event.x - self.drag_start[0])
            dy = abs(event.y - self.drag_start[1])
            if dx > 5 or dy > 5:
                self.dragging = True
                self.info_label.text = f"Dragging... ({int(dx)}x{int(dy)})"
            return gui.SceneWidget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.BUTTON_UP and self.drag_start is not None:
            if self.dragging:
                # Drag selection
                x0, y0 = self.drag_start
                x1, y1 = event.x, event.y
                self.drag_start = None
                self.dragging = False
                self._drag_select(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            else:
                # Single click -> grow from seed
                x, y = self.drag_start
                self.drag_start = None
                self._click_pick(x, y)
            return gui.SceneWidget.EventCallbackResult.CONSUMED

        return gui.SceneWidget.EventCallbackResult.IGNORED

    def _click_pick(self, x, y):
        """Single click: grow (ground/wall) or erase/delete nearby points."""
        def do_pick():
            def on_depth(depth_image):
                world_pt = self._screen_to_3d(x, y, depth_image)
                if world_pt is None:
                    return
                if self.mode in ("erase", "delete"):
                    # Find points near click within seed_radius
                    [k, indices, _] = self.tree.search_radius_vector_3d(world_pt, self.seed_radius)
                    if k > 0:
                        self._erase_or_delete(list(indices))
                else:
                    [k, idx, dist] = self.tree.search_knn_vector_3d(world_pt, 1)
                    if k > 0 and dist[0] < 1.0:
                        seed_idx = idx[0]
                        print(f"[{self.mode.upper()}] Click seed {seed_idx}: "
                              f"({self.pts[seed_idx][0]:.2f}, {self.pts[seed_idx][1]:.2f}, {self.pts[seed_idx][2]:.2f})")
                        self._grow_from_seeds([seed_idx])
            self.scene_widget.scene.scene.render_to_depth_image(on_depth)
        gui.Application.instance.post_to_main_thread(self.window, do_pick)

    def _drag_select(self, sx0, sy0, sx1, sy1):
        """Drag rectangle: select all visible points within the rectangle."""
        def do_drag():
            def on_depth(depth_image):
                screen, depths = self._project_to_screen()

                # Points within rectangle and in front of camera
                in_rect = ((screen[:, 0] >= sx0) & (screen[:, 0] <= sx1) &
                           (screen[:, 1] >= sy0) & (screen[:, 1] <= sy1) &
                           (depths > -1.0) & (depths < 1.0))

                if self.mode == "erase":
                    # Erase: only select assigned points in rectangle
                    assigned = self.ground_mask | self.wall_mask
                    candidates = in_rect & assigned
                elif self.mode == "delete":
                    # Delete: select all points in rectangle (except already deleted)
                    candidates = in_rect & ~self.deleted_mask
                else:
                    # Ground/wall grow: exclude already labeled
                    already = self.ground_mask | self.wall_mask
                    candidates = in_rect & ~already

                selected_indices = np.where(candidates)[0]
                if len(selected_indices) == 0:
                    print("No points in drag rectangle")
                    self.info_label.text = f"Click/drag on {self.mode}!"
                    return

                print(f"[{self.mode.upper()}] Drag selected {len(selected_indices)} points")

                if self.mode in ("erase", "delete"):
                    self._erase_or_delete(selected_indices.tolist())
                    return

                self._grow_from_seeds(selected_indices.tolist())

            self.scene_widget.scene.scene.render_to_depth_image(on_depth)
        gui.Application.instance.post_to_main_thread(self.window, do_drag)

    def _erase_or_delete(self, indices):
        """Erase (unassign from ground/wall) or delete (permanently hide) points."""
        mask = np.zeros(len(self.pts), dtype=bool)
        for i in indices:
            mask[i] = True

        if self.mode == "erase":
            # Remove ground/wall assignment
            affected = (self.ground_mask & mask) | (self.wall_mask & mask)
            count = affected.sum()
            if count > 0:
                self.history.append(("erase", {
                    "ground": self.ground_mask[mask].copy(),
                    "wall": self.wall_mask[mask].copy(),
                    "indices": mask.copy()
                }))
                self.ground_mask[mask] = False
                self.wall_mask[mask] = False
                print(f"[ERASE] Unassigned {count} pts")
            else:
                print("[ERASE] No assigned points in selection")
        else:  # delete
            count = mask.sum()
            self.history.append(("delete", {
                "ground": self.ground_mask[mask].copy(),
                "wall": self.wall_mask[mask].copy(),
                "deleted": self.deleted_mask[mask].copy(),
                "indices": mask.copy()
            }))
            self.ground_mask[mask] = False
            self.wall_mask[mask] = False
            self.deleted_mask[mask] = True
            print(f"[DELETE] Removed {count} pts")

        self._apply_colors()
        self.scene_widget.scene.remove_geometry("pcd")
        self.scene_widget.scene.add_geometry("pcd", self.pcd, self.mat)
        self.status_label.text = (f"Ground: {self.ground_mask.sum()} | "
                                  f"Wall: {self.wall_mask.sum()} | "
                                  f"Del: {self.deleted_mask.sum()}")
        self.info_label.text = f"Click/drag: {self.mode}!"

    def _grow_from_seeds(self, seed_indices):
        """Region grow from multiple seed points."""
        if self.growing:
            return
        self.growing = True

        mode = self.mode
        p = self.presets[mode]
        normal_thresh = p["normal_thresh"]
        z_tol = p["z_tol"]
        z_band = p["z_band"]

        # Compute average seed normal and z
        seed_normals = []
        for si in seed_indices:
            n = self.normals[si].copy()
            if mode == "ground":
                if n[2] < 0:
                    n = -n
            seed_normals.append(n)
        avg_normal = np.mean(seed_normals, axis=0)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)

        seed_zs = self.pts[seed_indices, 2]
        seed_z_center = np.mean(seed_zs)

        def grow():
            new_mask = np.zeros(len(self.pts), dtype=bool)
            visited = (self.ground_mask | self.wall_mask | self.deleted_mask).copy()

            queue = []
            for si in seed_indices:
                if not visited[si]:
                    visited[si] = True
                    new_mask[si] = True
                    queue.append(si)

            head = 0
            while head < len(queue):
                idx = queue[head]
                head += 1
                [k, nn_indices, _] = self.tree.search_radius_vector_3d(
                    self.pts[idx], self.seed_radius)
                for ni in nn_indices:
                    if visited[ni]:
                        continue
                    visited[ni] = True

                    dz_local = abs(self.pts[ni][2] - self.pts[idx][2])
                    dz_seed = abs(self.pts[ni][2] - seed_z_center)

                    n = self.normals[ni].copy()
                    if mode == "ground":
                        if n[2] < 0:
                            n = -n
                        dot = np.dot(avg_normal, n)
                    else:
                        dot = abs(np.dot(avg_normal, n / (np.linalg.norm(n) + 1e-10)))

                    if dz_local < z_tol and dz_seed < z_band and dot > normal_thresh:
                        new_mask[ni] = True
                        queue.append(ni)

            added = new_mask.sum()
            if added > 0:
                self.history.append((mode, new_mask.copy()))
                if mode == "ground":
                    self.ground_mask |= new_mask
                else:
                    self.wall_mask |= new_mask

            def update():
                self._apply_colors()
                self.scene_widget.scene.remove_geometry("pcd")
                self.scene_widget.scene.add_geometry("pcd", self.pcd, self.mat)
                self.status_label.text = (f"Ground: {self.ground_mask.sum()} | "
                                          f"Wall: {self.wall_mask.sum()} | "
                                          f"Del: {self.deleted_mask.sum()}")
                self.info_label.text = f"Click/drag: {self.mode}!"
                self.growing = False
            gui.Application.instance.post_to_main_thread(self.window, update)
            print(f"[{mode.upper()}] Added {added} pts "
                  f"(ground={self.ground_mask.sum()}, wall={self.wall_mask.sum()})")

        self.info_label.text = f"Growing {mode}..."
        threading.Thread(target=grow, daemon=True).start()

    def _on_undo(self):
        if not self.history:
            return
        mode, last = self.history.pop()
        if mode in ("ground", "wall"):
            if mode == "ground":
                self.ground_mask &= ~last
            else:
                self.wall_mask &= ~last
            print(f"Undo [{mode}]: -{last.sum()} pts")
        elif mode in ("erase", "delete"):
            # Restore previous state
            indices = last["indices"]
            self.ground_mask[indices] = last["ground"]
            self.wall_mask[indices] = last["wall"]
            if "deleted" in last:
                self.deleted_mask[indices] = last["deleted"]
            print(f"Undo [{mode}]")
        self._apply_colors()
        self.scene_widget.scene.remove_geometry("pcd")
        self.scene_widget.scene.add_geometry("pcd", self.pcd, self.mat)
        self.status_label.text = (f"Ground: {self.ground_mask.sum()} | "
                                  f"Wall: {self.wall_mask.sum()} | "
                                  f"Del: {self.deleted_mask.sum()}")

    def _on_reset(self):
        self.ground_mask[:] = False
        self.wall_mask[:] = False
        self.deleted_mask[:] = False
        self.history.clear()
        self._apply_colors()
        self.scene_widget.scene.remove_geometry("pcd")
        self.scene_widget.scene.add_geometry("pcd", self.pcd, self.mat)
        self.status_label.text = "Ground: 0 | Wall: 0 | Del: 0"

    def _on_save(self):
        print("Propagating to full resolution...")
        self.info_label.text = "Saving..."

        full_pts = np.asarray(self.pcd_full.points)
        full_tree = o3d.geometry.KDTreeFlann(self.pcd_full)

        full_ground_mask = np.zeros(len(full_pts), dtype=bool)
        for gp in self.pts[self.ground_mask]:
            [k, idx, _] = full_tree.search_radius_vector_3d(gp, 0.06)
            for i in idx:
                full_ground_mask[i] = True

        full_wall_mask = np.zeros(len(full_pts), dtype=bool)
        for wp in self.pts[self.wall_mask]:
            [k, idx, _] = full_tree.search_radius_vector_3d(wp, 0.06)
            for i in idx:
                if not full_ground_mask[i]:
                    full_wall_mask[i] = True

        # Propagate deleted mask to full resolution
        full_deleted_mask = np.zeros(len(full_pts), dtype=bool)
        for dp in self.pts[self.deleted_mask]:
            [k, idx, _] = full_tree.search_radius_vector_3d(dp, 0.06)
            for i in idx:
                full_deleted_mask[i] = True

        full_other_mask = ~(full_ground_mask | full_wall_mask | full_deleted_mask)

        ground_pcd = self.pcd_full.select_by_index(np.where(full_ground_mask)[0])
        wall_pcd = self.pcd_full.select_by_index(np.where(full_wall_mask)[0])
        other_pcd = self.pcd_full.select_by_index(np.where(full_other_mask)[0])

        o3d.io.write_point_cloud(os.path.join(self.save_dir, "ground.pcd"), ground_pcd)
        # Write wall.pcd — if empty, write a minimal valid PCD file
        wall_path = os.path.join(self.save_dir, "wall.pcd")
        if len(wall_pcd.points) > 0:
            o3d.io.write_point_cloud(wall_path, wall_pcd)
        else:
            with open(wall_path, 'w') as f:
                f.write("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\n"
                        "SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                        "WIDTH 0\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                        "POINTS 0\nDATA ascii\n")
            print("Wall: no points selected — wrote empty PCD")
        o3d.io.write_point_cloud(os.path.join(self.save_dir, "other.pcd"), other_pcd)

        with open(os.path.join(self.save_dir, "segmentation_info.json"), "w") as f:
            json.dump({
                "seed_radius": self.seed_radius,
                "presets": self.presets,
                "ground_count": int(full_ground_mask.sum()),
                "wall_count": int(full_wall_mask.sum()),
                "other_count": int(full_other_mask.sum()),
                "deleted_count": int(full_deleted_mask.sum()),
            }, f, indent=2)

        self.status_label.text = (f"Saved! G={full_ground_mask.sum()} "
                                  f"W={full_wall_mask.sum()} "
                                  f"D={full_deleted_mask.sum()}")
        self.info_label.text = "Done!"
        print(f"Saved: ground={full_ground_mask.sum()}, wall={full_wall_mask.sum()}, "
              f"other={full_other_mask.sum()}, deleted={full_deleted_mask.sum()}")

    def _on_psize_changed(self, val):
        self.mat.point_size = int(val)
        self.scene_widget.scene.modify_geometry_material("pcd", self.mat)

    def run(self):
        gui.Application.instance.run()


def extract_pcd_from_bag(bag_path, output_dir, pcl_topic="/livox/lidar",
                         target_frame="map", voxel_size=0.02):
    """Extract accumulated point cloud from rosbag, transform to target frame, save as PCD."""
    import struct
    import rosbag
    import tf.transformations as tft

    print(f"Opening bag: {bag_path}")
    bag = rosbag.Bag(bag_path, 'r')

    # -- 1. Collect TF data -----------------------------------------------------
    # Separate static transforms (published once at arbitrary time) from dynamic ones.
    print("Building TF data...")
    static_transforms = {}   # (parent, child) → (trans, rot) as 4x4
    dynamic_tf_data = {}     # (parent, child) → list of (stamp_sec, 4x4)

    for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
        for tr in msg.transforms:
            parent = tr.header.frame_id
            child = tr.child_frame_id
            p = tr.transform.translation
            q = tr.transform.rotation
            mat = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
            mat[:3, 3] = [p.x, p.y, p.z]

            if topic == '/tf_static':
                static_transforms[(parent, child)] = mat
            else:
                key = (parent, child)
                if key not in dynamic_tf_data:
                    dynamic_tf_data[key] = []
                dynamic_tf_data[key].append((tr.header.stamp.to_sec(), mat))

    # Sort dynamic TFs by time for interpolation
    for key in dynamic_tf_data:
        dynamic_tf_data[key].sort(key=lambda x: x[0])

    def lookup_dynamic(parent, child, stamp_sec):
        """Lookup dynamic TF at given time (nearest neighbor)."""
        key = (parent, child)
        if key not in dynamic_tf_data:
            return None
        entries = dynamic_tf_data[key]
        # Binary search for nearest
        import bisect
        times = [e[0] for e in entries]
        idx = bisect.bisect_left(times, stamp_sec)
        if idx == 0:
            return entries[0][1]
        if idx >= len(entries):
            return entries[-1][1]
        # Pick closer one
        if abs(times[idx] - stamp_sec) < abs(times[idx - 1] - stamp_sec):
            return entries[idx][1]
        return entries[idx - 1][1]

    def lookup_static(parent, child):
        """Lookup static TF."""
        return static_transforms.get((parent, child))

    # Determine PCL frame from first message
    pcl_frame = None
    for _, msg, _ in bag.read_messages(topics=[pcl_topic]):
        pcl_frame = msg.header.frame_id
        break
    if pcl_frame is None:
        print(f"ERROR: No messages on topic '{pcl_topic}'")
        bag.close()
        return None
    print(f"  PCL frame: {pcl_frame}")

    # Build static chain from pcl_frame up to a frame that has dynamic TF to target_frame.
    # Typical chain: map →(dynamic)→ base_link →(static)→ livox_mid360
    # We need: T_map_livox = T_map_base(dynamic) @ T_base_livox(static)
    # Find static chain: which static transforms connect pcl_frame to a dynamically-linked frame?
    static_chain_mat = np.eye(4)
    current = pcl_frame
    # Walk up static parents until we find one that has a dynamic link from target_frame
    for _ in range(10):  # max depth
        if (target_frame, current) in dynamic_tf_data:
            break
        # Find static parent of current
        found = False
        for (parent, child), mat in static_transforms.items():
            if child == current:
                static_chain_mat = mat @ static_chain_mat
                current = parent
                found = True
                break
        if not found:
            break

    dynamic_key_child = current
    print(f"  TF chain: {target_frame} →(dynamic)→ {dynamic_key_child} →(static)→ {pcl_frame}")
    print(f"  Dynamic TF entries: {len(dynamic_tf_data.get((target_frame, dynamic_key_child), []))}")

    # -- 2. Extract and transform point clouds ----------------------------------
    print(f"Extracting '{pcl_topic}' → '{target_frame}' frame...")
    all_points = []
    count = 0
    skipped = 0
    for _, msg, t in bag.read_messages(topics=[pcl_topic]):
        stamp_sec = msg.header.stamp.to_sec()

        # Dynamic part: target_frame → dynamic_key_child
        dyn_mat = lookup_dynamic(target_frame, dynamic_key_child, stamp_sec)
        if dyn_mat is None:
            skipped += 1
            continue

        # Full transform: target_frame → pcl_frame
        full_mat = dyn_mat @ static_chain_mat

        # Parse points from PointCloud2 (vectorized)
        n_pts = msg.width * msg.height
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_pts, msg.point_step)
        xyz = np.frombuffer(raw[:, :12].tobytes(), dtype=np.float32).reshape(n_pts, 3)
        valid = ~np.isnan(xyz).any(axis=1)
        xyz = xyz[valid]

        if len(xyz) == 0:
            continue

        # Transform to target frame
        ones = np.ones((len(xyz), 1), dtype=np.float32)
        pts_h = np.hstack([xyz, ones])
        pts_tf = (full_mat @ pts_h.T).T[:, :3]
        all_points.append(pts_tf)
        count += 1

    bag.close()
    print(f"  Processed {count} messages (skipped {skipped})")

    if not all_points:
        print("ERROR: No points extracted!")
        return None

    all_points = np.vstack(all_points)
    print(f"  Total points: {len(all_points)}")

    # Voxel downsample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"  After voxel downsample ({voxel_size}m): {len(pcd.points)} points")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "accumulated.pcd")
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"  Saved: {out_path}")
    return out_path


def select_bag_interactive(bag_dir="bag"):
    """List .bag files (and dirs containing .bag) in bag_dir, let user pick one."""
    bag_dir = os.path.abspath(bag_dir)
    if not os.path.isdir(bag_dir):
        print(f"Bag directory not found: {bag_dir}")
        return None

    candidates = []
    for entry in sorted(os.listdir(bag_dir)):
        full = os.path.join(bag_dir, entry)
        if entry.endswith('.bag') and os.path.isfile(full):
            candidates.append((entry, full))
        elif os.path.isdir(full):
            # Check for .bag files inside subdirectory
            for sub in sorted(os.listdir(full)):
                if sub.endswith('.bag'):
                    candidates.append((f"{entry}/{sub}", os.path.join(full, sub)))

    if not candidates:
        print(f"No .bag files found in {bag_dir}")
        return None

    print(f"\n=== Bag files in {bag_dir} ===")
    for i, (name, _) in enumerate(candidates):
        print(f"  [{i + 1}] {name}")
    print()

    while True:
        try:
            choice = input(f"Select bag [1-{len(candidates)}] (q to quit): ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                print(f"  → {candidates[idx][0]}")
                return candidates[idx][1]
        except (ValueError, EOFError):
            pass
        print("  Invalid choice, try again.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manual ground/wall segmentation tool")
    parser.add_argument("pcd", nargs="?", default=None, help="Input PCD file")
    parser.add_argument("--bag", type=str, default=None,
                        help="Input bag file (extracts PCD first). "
                             "If omitted and no pcd given, shows interactive bag selector.")
    parser.add_argument("--topic", type=str, default="/livox/lidar",
                        help="PointCloud2 topic in bag (default: /livox/lidar)")
    parser.add_argument("--frame", type=str, default="map",
                        help="Target TF frame (default: map)")
    parser.add_argument("--radius", type=float, default=0.3, help="Grow radius [m]")
    parser.add_argument("--z-tol", type=float, default=0.15, help="Z tolerance [m]")
    parser.add_argument("--z-min", type=float, default=None, help="Z min filter [m]")
    parser.add_argument("--z-max", type=float, default=None, help="Z max filter [m]")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for ground/wall/other.pcd (default: pcd/<bag_name>/)")
    args, _ = parser.parse_known_args()

    pcd_path = args.pcd

    # If --bag given, or no pcd given → extract from bag
    if args.bag or pcd_path is None:
        bag_path = args.bag
        if bag_path is None:
            # Interactive selection
            script_dir = os.path.dirname(os.path.abspath(__file__))
            bag_path = select_bag_interactive(os.path.join(script_dir, "../bag"))
            if bag_path is None:
                print("No bag selected. Exiting.")
                return

        # Determine output dir: pcd/<bag_name>/
        bag_name = os.path.splitext(os.path.basename(bag_path))[0]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = args.output_dir or os.path.join(script_dir, "../pcd", bag_name)

        pcd_path = extract_pcd_from_bag(bag_path, out_dir,
                                        pcl_topic=args.topic,
                                        target_frame=args.frame)
        if pcd_path is None:
            return
        args.output_dir = out_dir

    app = GroundWallSelectorApp(pcd_path, args.radius, args.z_tol, args.z_min, args.z_max)
    if args.output_dir:
        app.save_dir = os.path.abspath(args.output_dir)
        os.makedirs(app.save_dir, exist_ok=True)
        print(f"Output dir: {app.save_dir}")
    app.run()

if __name__ == "__main__":
    main()
