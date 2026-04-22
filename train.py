"""Training script for Soft Actor-Critic (SAC).

Usage
-----
    python train.py --env HalfCheetah-v4 --max_steps 1000000

The script will:
  1. Create the Gymnasium environment.
  2. Initialise the SAC agent.
  3. Collect experience and periodically update the agent.
  4. Evaluate the policy every `eval_freq` environment steps.
  5. Save the best policy to the `models/` directory.
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch

from sac import SAC


def evaluate(agent, env_name, seed, eval_episodes=10):
    """Run several deterministic episodes and return the mean return."""
    eval_env = gym.make(env_name)
    eval_env.reset(seed=seed + 100)
    total_reward = 0.0
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
    eval_env.close()
    return total_reward / eval_episodes


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env)
    env.reset(seed=args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        discount=args.discount,
        tau=args.tau,
        alpha=args.alpha,
        auto_tune_alpha=args.auto_tune_alpha,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_buffer_size,
    )

    os.makedirs("models", exist_ok=True)

    state, _ = env.reset()
    episode_reward = 0.0
    episode_num = 0
    best_eval_reward = float("-inf")

    for t in range(1, args.max_steps + 1):
        # Collect experience
        if t < args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Store transition (mask terminal due to time-limit as non-terminal)
        agent.store_transition(state, action, next_state, reward, float(terminated))

        state = next_state
        episode_reward += reward

        # Update agent
        if t >= args.start_steps:
            agent.update()

        if done:
            print(
                f"Step: {t:>8d} | Episode: {episode_num:>5d} | "
                f"Reward: {episode_reward:>10.2f}"
            )
            state, _ = env.reset()
            episode_reward = 0.0
            episode_num += 1

        # Evaluate
        if t % args.eval_freq == 0:
            eval_reward = evaluate(agent, args.env, args.seed)
            print(
                f"--- Evaluation at step {t:>8d}: mean return = {eval_reward:.2f} ---"
            )
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(f"models/{args.env}_best")
                print(f"    New best model saved (return={best_eval_reward:.2f})")

    env.close()
    print("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="SAC Training")
    parser.add_argument("--env", default="HalfCheetah-v4", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_steps", default=1_000_000, type=int)
    parser.add_argument("--start_steps", default=10_000, type=int,
                        help="Random exploration steps before training starts")
    parser.add_argument("--eval_freq", default=5_000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--replay_buffer_size", default=1_000_000, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--auto_tune_alpha", action=argparse.BooleanOptionalAction,
                        default=True, help="Automatically tune entropy temperature alpha")
    parser.add_argument("--actor_lr", default=3e-4, type=float)
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    parser.add_argument("--alpha_lr", default=3e-4, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
