"""Unit tests for the SAC implementation."""

import numpy as np
import pytest
import torch

from sac.replay_buffer import ReplayBuffer
from sac.networks import Actor, Critic
from sac.agent import SAC


# -------------------------------------------------------------------------
# ReplayBuffer tests
# -------------------------------------------------------------------------

class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(state_dim=4, action_dim=2, max_size=100)
        assert len(buf) == 0
        buf.add(
            np.zeros(4), np.zeros(2), np.zeros(4), 0.0, 0.0
        )
        assert len(buf) == 1

    def test_overflow_wraps_around(self):
        buf = ReplayBuffer(state_dim=4, action_dim=2, max_size=5)
        for i in range(10):
            buf.add(np.full(4, i), np.zeros(2), np.zeros(4), 0.0, 0.0)
        assert len(buf) == 5  # capped at max_size

    def test_sample_shape(self):
        buf = ReplayBuffer(state_dim=4, action_dim=2, max_size=100)
        for _ in range(50):
            buf.add(np.random.randn(4), np.random.randn(2), np.random.randn(4), 0.0, 0.0)
        states, actions, next_states, rewards, dones = buf.sample(batch_size=16)
        assert states.shape == (16, 4)
        assert actions.shape == (16, 2)
        assert next_states.shape == (16, 4)
        assert rewards.shape == (16, 1)
        assert dones.shape == (16, 1)


# -------------------------------------------------------------------------
# Network tests
# -------------------------------------------------------------------------

class TestActor:
    def test_output_shapes(self):
        actor = Actor(state_dim=8, action_dim=3)
        state = torch.randn(4, 8)
        mean, log_std = actor(state)
        assert mean.shape == (4, 3)
        assert log_std.shape == (4, 3)

    def test_sample_shapes(self):
        actor = Actor(state_dim=8, action_dim=3)
        state = torch.randn(4, 8)
        action, log_prob, mean_action = actor.sample(state)
        assert action.shape == (4, 3)
        assert log_prob.shape == (4, 1)
        assert mean_action.shape == (4, 3)

    def test_action_bounded(self):
        actor = Actor(state_dim=8, action_dim=3)
        state = torch.randn(100, 8)
        action, _, _ = actor.sample(state)
        assert action.abs().max().item() <= 1.0 + 1e-5


class TestCritic:
    def test_output_shapes(self):
        critic = Critic(state_dim=8, action_dim=3)
        state = torch.randn(4, 8)
        action = torch.randn(4, 3)
        q1, q2 = critic(state, action)
        assert q1.shape == (4, 1)
        assert q2.shape == (4, 1)

    def test_q1_value_shape(self):
        critic = Critic(state_dim=8, action_dim=3)
        state = torch.randn(4, 8)
        action = torch.randn(4, 3)
        q1 = critic.q1_value(state, action)
        assert q1.shape == (4, 1)


# -------------------------------------------------------------------------
# SAC agent tests
# -------------------------------------------------------------------------

class TestSAC:
    def _make_agent(self, state_dim=8, action_dim=3):
        return SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            batch_size=32,
            replay_buffer_size=1000,
            device="cpu",
        )

    def test_select_action_shape(self):
        agent = self._make_agent()
        state = np.random.randn(8)
        action = agent.select_action(state)
        assert action.shape == (3,)

    def test_select_action_evaluate(self):
        agent = self._make_agent()
        state = np.random.randn(8)
        action = agent.select_action(state, evaluate=True)
        assert action.shape == (3,)

    def test_update_returns_none_when_buffer_small(self):
        agent = self._make_agent()
        result = agent.update()
        assert result is None

    def test_update_returns_losses_after_filling_buffer(self):
        agent = self._make_agent()
        for _ in range(50):
            s = np.random.randn(8)
            a = np.random.uniform(-1, 1, 3)
            ns = np.random.randn(8)
            agent.store_transition(s, a, ns, 0.0, 0.0)
        result = agent.update()
        assert result is not None
        assert "critic_loss" in result
        assert "actor_loss" in result
        assert "alpha" in result

    def test_save_and_load(self, tmp_path):
        agent = self._make_agent()
        path = str(tmp_path / "model")
        agent.save(path)
        agent.load(path)
