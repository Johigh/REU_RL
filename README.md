# REU_RL

Research repository for the REU research group – Soft Actor-Critic (SAC) reinforcement learning algorithm.

## Repository Structure

```
REU_RL/
├── sac/
│   ├── __init__.py        # Package entry point
│   ├── agent.py           # SAC agent (main algorithm)
│   ├── networks.py        # Actor and Critic neural networks
│   └── replay_buffer.py   # Experience replay buffer
├── tests/
│   └── test_sac.py        # Unit tests
├── train.py               # Training script
└── requirements.txt       # Python dependencies
```

## Algorithm Overview

SAC is an off-policy maximum-entropy deep RL algorithm for continuous action spaces. Key features:

- **Twin Q-networks** to reduce overestimation bias in value estimates.
- **Reparameterized Gaussian policy** with tanh squashing for bounded actions.
- **Automatic entropy tuning** – the temperature parameter α is learned to target a desired entropy level.
- **Soft target updates** for the critic via Polyak averaging.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --env HalfCheetah-v4 --max_steps 1000000
```

Common arguments:

| Argument | Default | Description |
|---|---|---|
| `--env` | `HalfCheetah-v4` | Gymnasium environment ID |
| `--seed` | `0` | Random seed |
| `--max_steps` | `1000000` | Total environment steps |
| `--start_steps` | `10000` | Random exploration steps before training |
| `--eval_freq` | `5000` | Evaluation frequency (steps) |
| `--batch_size` | `256` | Mini-batch size |
| `--hidden_dim` | `256` | Hidden layer size |
| `--discount` | `0.99` | Discount factor γ |
| `--tau` | `0.005` | Soft update coefficient τ |
| `--auto_tune_alpha` | `True` | Automatic entropy tuning |

### Using the Agent in Your Own Code

```python
from sac import SAC

agent = SAC(state_dim=17, action_dim=6)

# Collect experience
agent.store_transition(state, action, next_state, reward, done)

# Update (call after every environment step)
losses = agent.update()

# Select action (stochastic during training, deterministic for evaluation)
action = agent.select_action(state)
action = agent.select_action(state, evaluate=True)

# Save / load
agent.save("models/my_model")
agent.load("models/my_model")
```

## Running Tests

```bash
pytest tests/
```

## References

- Haarnoja et al., [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), ICML 2018.
- Haarnoja et al., [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905), 2019.
