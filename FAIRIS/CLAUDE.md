# FAIRIS Project

**Framework for AI Robotic Intelligence Simulation** — a Python framework for developing and testing robotic navigation and reinforcement learning algorithms in the [Webots](https://cyberbotics.com/) simulator.

## Project Structure

```
FAIRIS/
├── fairis_lib/              # Core robot control and simulation libraries
│   ├── robot_lib/           # Robot abstraction layer (HamBot, MyRobot)
│   └── simulation_lib/      # Webots environment management
├── fairis_tools/            # Experiment utilities (logging, place cells, paths)
├── reinforcement_lib/       # RL algorithm implementations
│   ├── SAC/                 # Soft Actor-Critic
│   ├── PPO/                 # Proximal Policy Optimization
│   ├── OC/                  # Option Critic (hierarchical RL)
│   └── BrendonSAC/          # Brendon's SAC variant (Webots-native, no place cells)
├── simulation/
│   ├── worlds/              # Webots .wbt world files and maze XMLs
│   └── controllers/         # Webots controller scripts (Train*/Test*)
├── data/                    # Cached simulation data
├── docs/                    # Documentation and figures
├── create_venv.py           # Virtual environment setup
└── requirements.txt         # Python dependencies
```

## Key Files

| File | Purpose |
|------|---------|
| `fairis_lib/robot_lib/hambot.py` | Core robot interface — motors, sensors, environment loading |
| `fairis_lib/robot_lib/my_robot.py` | User-facing robot class extending HamBot |
| `fairis_lib/simulation_lib/environment.py` | Maze loading, robot teleportation, goal checking |
| `reinforcement_lib/SAC/Agent.py` | SAC agent with experience replay buffer |
| `reinforcement_lib/SAC/Networks.py` | Actor, Critic, Value networks for SAC |
| `reinforcement_lib/BrendonSAC/FAIRISEnvNoPC.py` | Gym-compatible env wrapper (no place cells) |
| `simulation/controllers/TrainSAC/TrainSAC.py` | Webots controller for SAC training |

## Setup

```bash
python create_venv.py       # Creates venv and generates Webots runtime.ini files
python add_runtime_ini.py   # Reconfigure Webots runtime if needed
```

**Requirements:** Python 3.10–3.11 (3.12 not compatible), Webots R2025a

## Running Experiments

1. Launch Webots and open a world file (e.g., `simulation/worlds/empty_room.wbt`)
2. Select the appropriate controller (TrainSAC, TrainPPO, etc.)
3. Training/testing controllers import `MyRobot`, load a maze XML, and run a loop

Brendon's SAC variant (`reinforcement_lib/BrendonSAC/`) runs directly via `TestSAC.py` without going through Webots controllers.

## RL Algorithms

- **SAC** (Soft Actor-Critic) — main algorithm, continuous action space
- **PPO** (Proximal Policy Optimization) — policy gradient alternative
- **Option Critic** — hierarchical RL with options
- Place cell variants (with/without visual place cells from CNN features)

## Dependencies

Key libraries: `torch`, `tensorflow`, `gym`, `numpy`, `opencv-python`, `matplotlib`, `scipy`, `scikit-learn`, `shapely`

## Current Branch: `brendon_oc`

Active development on the BrendonSAC module — a Webots-native SAC implementation without place cells (`FAIRISEnvNoPC.py`, `TestSAC.py`).