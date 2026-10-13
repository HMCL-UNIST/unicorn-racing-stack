# Unicorn Race Stack Installation

The stack runs on `ROS Noetic` inside a ***Docker*** container. Most of it is written in `Python`, so we recommend an `Intel NUC` as the on-board computer. Recent NUC models do not support `Ubuntu 20.04`; install an Ubuntu version supported by **your NUC** on the host and use the container for ROS.

## Prerequisites

A 1/10 scale RoboRacer (formerly F1TENTH) vehicle is used as the hardware platform. If you do not have one prepared, please refer to the build instructions on the [official RoboRacer website](https://roboracer.ai/).

The 3D stack additionally uses:

- a 3D LiDAR with IMU (3D mapping and GLIL localization)
- an IMU (or the VESC built-in IMU)
- an ELRS receiver for manual control (optional, a USB joystick also works)
- [blink(1)](https://blink1.thingm.com/) for the state indicator LED (optional)

To deploy or develop using Docker, make sure you have both `Docker` and `Docker Compose` installed, and that Docker is accessible **without** sudo (see the official [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)).

  - [Docker / Docker Compose](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)

## Clone the Repo

Create a workspace folder with the build cache directories used by the container:
```bash
mkdir ~/unicorn_ws && cd ~/unicorn_ws

mkdir -p cache/noetic/build cache/noetic/devel cache/noetic/logs
```

Clone the `ros1_3d` branch together with its submodules. Some submodules are fetched over SSH, so add an SSH key to your GitHub account first.

> **Note:** the submodules `state_estimation/GLIL_unicorn_racing` and `perception/2.5d_detection` are private repositories. Cloning them requires access granted by their owners; without it, the clone of these submodules fails.

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub

git clone --recurse-submodules -b ros1_3d https://github.com/HMCL-UNIST/unicorn-racing-stack.git
cd unicorn-racing-stack
```

## udev Rules Setup
To ensure that the USB-connected devices are recognized consistently and accessible without requiring manual reconfiguration, a set of custom udev rules must be installed.

For installation of custom udev rules, run the script with root privileges:

```bash
# for Blink1
sudo ./.devcontainer/.install_utils/udev_rules_blink1.sh

# for IMU
sudo ./.devcontainer/.install_utils/udev_rules_imu.sh

# for VESC
sudo ./.devcontainer/.install_utils/udev_rules_vesc.sh

# for ELRS receiver (CP2102 USB-TTL, VID:10c4 PID:ea60)
sudo ./.devcontainer/.install_utils/udev_rules_elrs.sh
```

After installing the udev rules, **unplug and re-plug the affected USB devices** (or reboot) so the new rules are applied to the existing device nodes.

Then check if the devices are correctly linked:

```bash
# for IMU
ls -l /dev/IMU

# for VESC
ls -l /dev/VESC

# for ELRS
ls -l /dev/ELRS
```

## Build with docker

First, build the base image with `docker compose`:
```bash
docker compose build base_x86_3d
```

Then export the needed environment variables and build the stack image:
```bash
export UID=$(id -u)
export GID=$(id -g)
docker compose build nuc_3d
```

Then open the repository in VS Code:
```bash
code ~/unicorn_ws/unicorn-racing-stack
```

In the VS Code window, press `Ctrl + Shift + P` to open the command palette and select `Dev Containers: Rebuild and Reopen in Container`. The container (named `icra2026`) mounts the repository at `~/catkin_ws/src/race_stack` and builds the workspace on first start.
By installing the `Remote Development` extension from the `Extensions` tab in VS Code, you will be able to easily reconnect to previously created containers in the future.

## Optional components

- **acados** — needed only for the local racing line (`use_local_racing_line:=true`). Run [`thirdparty/setup_acados.sh`](./thirdparty/setup_acados.sh) inside the container.
- **HSL linear solvers** — speed up the IPOPT-based GGV, raceline and planner optimizations. HSL is not included: the Coin-HSL archive requires a license (free, [request from STFC](https://licences.stfc.ac.uk/product/coin-hsl-archive)) and has to be downloaded and built by each user; see [`fast_ggv_gen/solver`](./planner/3d_gb_optimizer/fast_ggv_gen/solver/README.md). Without HSL, MUMPS is used automatically.

## How to use GUI applications with the container
To have more information on how to use GUI applications with the container, please refer to the [GUI applications documentation](./.docker_utils/README_GUI.md).
