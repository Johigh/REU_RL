import copy

import numpy as np
import torch
import torch.nn.functional as F

from sac.networks import Actor, Critic
from sac.replay_buffer import ReplayBuffer


class SAC:
    """Soft Actor-Critic (SAC) agent.

    Based on:
        Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
        Reinforcement Learning with a Stochastic Actor", ICML 2018.
        Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2019.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        discount=0.99,
        tau=0.005,
        alpha=0.2,
        auto_tune_alpha=True,
        target_entropy=None,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        batch_size=256,
        replay_buffer_size=int(1e6),
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        self.auto_tune_alpha = auto_tune_alpha

        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic (twin Q-networks) + target
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Entropy temperature
        if auto_tune_alpha:
            self.target_entropy = (
                -action_dim if target_entropy is None else target_entropy
            )
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha

        # Replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, replay_buffer_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, state, evaluate=False):
        """Select an action given a state.

        Args:
            state: numpy array of shape (state_dim,).
            evaluate: if True, use the deterministic mean action.

        Returns:
            numpy array of shape (action_dim,) in [-1, 1].
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        return action.cpu().numpy().flatten()

    def store_transition(self, state, action, next_state, reward, done):
        """Store a single transition in the replay buffer."""
        self.replay_buffer.add(state, action, next_state, reward, done)

    def update(self):
        """Sample a batch and perform one gradient update step.

        Returns:
            dict with scalar losses, or None if the buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, next_states, rewards, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # ---- Critic update ----
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_q = rewards + (1.0 - dones) * self.discount * min_q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Actor update ----
        pi, log_pi, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Alpha update ----
        alpha_loss = None
        if self.auto_tune_alpha:
            alpha_loss = -(
                self.log_alpha * (log_pi.detach() + self.target_entropy)
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ---- Soft update target critic ----
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if alpha_loss is not None else 0.0,
            "alpha": self.alpha,
        }

    def save(self, path):
        """Save model weights to *path* (without extension)."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha if self.auto_tune_alpha else None,
            },
            f"{path}.pt",
        )

    def load(self, path):
        """Load model weights from *path* (without extension)."""
        checkpoint = torch.load(f"{path}.pt", map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        if self.auto_tune_alpha and checkpoint["log_alpha"] is not None:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            self.alpha = self.log_alpha.exp().item()
