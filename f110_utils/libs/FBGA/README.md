FBGA
=========

FBGA - Forward Backward Generic Acceleration constraints is a library specifically developed to solve optimal velocity profile given a complex g-g-v constraint envelope the described implementation is under review in the Robotics and Automation Letters (RAL) journal.
This library will later evolve into a generic tool including multiple dimensions constraints and a more general approach to the problem.
The library is able to compute the maximum performance longitudinal maneuver given a generic g-g-v envelope and a curvature profile (i.e., the trajectory path).

For usage details, please refer to the [FBGA Usage Guide](FBGA_Usage_Guide.md).

Cite as:

```{bibtex}
@article{piazza2025real,
  title={Real-Time Velocity Profile Optimization for Time-Optimal Maneuvering With Generic Acceleration Constraints},
  author={Piazza, Mattia and Piccinini, Mattia and Taddei, Sebastiano and Biral, Francesco and Bertolazzi, Enrico},
  journal={IEEE Robotics and Automation Letters},
  volume={11},
  number={2},
  pages={1674--1681},
  year={2025},
  publisher={IEEE}
}
```
<!-- download the bib -->
[__Download the bib__](bibtex.bib)



## Requirements

### General

- git
- cmake
- make

### Third party libraries

Some third party libraries are needed to compile the project.

You will need to run the following command to install the third party libraries:

```{shell}
./third_party.sh
```

The library is self-contained and standalone, so you don't need to install any other library to use it as is. Third party libraries are used for testing and development purposes.

<!-- Note -->

**Note**: The third party libraries are not included in the repository, so you need to run the command above to download them. Please make sure you have an internet connection when running the command and that you have the necessary permissions to install the libraries.

## Building

To install the project you need to run the following commands:

```{shell}
./build.sh
```

Default will build the project in release mode, if you want to build in debug mode you can run:

```{shell}
./build.sh -debug
```

For further information you can run:

```{shell}
./build.sh -h
```

### Alternative build

If you want to install the project in a different directory you can run:

```{shell}
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
make -j
```

### Running the examples 

To run the tests you can use the following command:

```{shell}
./bin/GIGI_test_moto.exe
```

## Usage in this stack

FBGA is a standalone CMake project, not a catkin package. It is built by `.devcontainer/.install_utils/post_installation.sh` (`build.sh -a -b -release`), which produces:

- `bin/GIGI_test_unicorn.exe`: CLI driver (`src_tests/GIGI_test_unicorn.cc`) that solves one speed profile from a CSV of `s`, curvature and apparent gravity, using a packed GGV (`gg.bin`) and a vehicle `params.txt`.
- `bin/fbga_native*.so`: pybind11 module (`python/fbga_native.cpp`) exposing the same solver in-process. It is built only if `pybind11` is found; otherwise the CLI driver is used. `python/test_native_vs_exe.py` checks that both give the same result.

Used by:

- `run_fwbw.sh`: offline speed profile for a 3D racing line with the CLI driver (wraps `planner/3d_gb_optimizer/global_line/global_racing_line/run_fwbw.py`, builds FBGA first if the binary is missing). The options are listed in the script header.
  ```shell
  ./run_fwbw.sh --map-dir <path/to/map>   # reads <map>_3d_smoothed.csv and <map>_3d_<vehicle>_timeoptimal.csv, writes <map>_fwbw_<model>.json
  ./run_fwbw.sh --track <smoothed.csv> --raceline <raceline.csv>   # files under global_line/data/
  ```
- [`fbga_runner`](../fbga_runner/README.md): Python wrapper (native module or CLI driver) used online by the FBGA state machine and `fast_sqp_planner`.
