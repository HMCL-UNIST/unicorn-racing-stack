# HSL Setup for IPOPT (ma27)
The IPOPT problems of the stack (GGV generation, track smoothing, racing line, `3d_optimized_vel_planner.py`, `fast_sqp_planner`) use the HSL linear solver `ma27` when it is available and MUMPS otherwise. Nothing has to be changed in the code; HSL only makes them faster (the fast GGV generation about 2-5x).

## License
HSL is **not included** in this repository. The Coin-HSL Archive is free for academic and commercial use, but **requires a license request** and **may not be redistributed**, so every user has to request and download it:
- https://licences.stfc.ac.uk/product/coin-hsl-archive

## Installation
1. After the license is approved, download the **source tarball** `coinhsl-<date>.tar.gz` (the mac/windows binaries do not work in the Linux container) and put it into this folder. It is ignored by git.
2. Inside the container, run the setup script (as root if the user has no `sudo`):
   ```bash
   cd ~/catkin_ws/src/race_stack/planner/3d_gb_optimizer/fast_ggv_gen/solver
   bash setup_hsl.sh
   ```
   It installs the build dependencies, builds the library into `install/`, links it as CasADi's `libhsl.so` and checks `ma27` with a small problem.
3. Check: the scripts print `linear_solver: ma27 (HSL)` at start, e.g.
   ```bash
   cd ~/catkin_ws/src/race_stack/planner/3d_gb_optimizer/fast_ggv_gen
   python3 fast_gen_gg_diagrams.py --vehicle_name rc_car_10th --fast
   ```

## After recreating the container
The built library stays in `install/` on the host, but the link inside the container is lost. Run `bash setup_hsl.sh` again, or recreate the link:
```bash
ln -sf ~/catkin_ws/src/race_stack/planner/3d_gb_optimizer/fast_ggv_gen/solver/install/lib/x86_64-linux-gnu/libcoinhsl.so \
       /usr/local/lib/python3.8/dist-packages/casadi/libhsl.so
```

## Troubleshooting
- `add_install_script args must be strings`: Meson is too old, `pip3 install --upgrade meson`.
- `metis.h: No such file or directory`: `apt-get install libmetis-dev`.
- Scripts still report MUMPS: the `libhsl.so` link is missing or wrong (see above).
