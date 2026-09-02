import gymnasium as gym
import numpy as np
import pytest
import torch

from gae_actor_critic_cartpole import *


@pytest.fixture
def env():
    environment = gym.make("CartPole-v1")
    yield environment
    environment.close()


@pytest.fixture
def policy(env):
    torch.manual_seed(SEED)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    return PolicyNetwork(obs_dim, n_actions)


@pytest.fixture
def episode(env, policy):
    return collect_episode(env, policy, seed=SEED)

# λ=0 must equal one-step TD errors
def test_lambda_zero():
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    next_values = torch.tensor([1.0, 123.0])
    terminated = torch.tensor([False, True])

    advantages, targets, td_errors = compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        gamma=0.9,
        gae_lambda=0.0,
    )

    expected_td_errors = torch.tensor([1.4, 1.0])
    expected_targets = torch.tensor([1.9, 2.0])

    assert torch.allclose(td_errors, expected_td_errors)
    assert torch.allclose(advantages, expected_td_errors)
    assert torch.allclose(targets, expected_targets)

# λ=1 propagates the later TD error backward
def test_lambda_one():
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    next_values = torch.tensor([1.0, 123.0])
    terminated = torch.tensor([False, True])

    advantages, targets, _ = compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        gamma=0.9,
        gae_lambda=1.0,
    )

    expected_advantages = torch.tensor([2.3, 1.0])
    expected_targets = torch.tensor([2.8, 2.0])

    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(targets, expected_targets)


# truncation must still bootstrap
def test_truncation_bootstraps():
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.5])
    next_values = torch.tensor([2.0])

    truncated_advantage, _, _ = compute_gae(
        rewards,
        values,
        next_values,
        terminated=torch.tensor([False]),
        gamma=0.9,
        gae_lambda=0.95,
    )

    terminal_advantage, _, _ = compute_gae(
        rewards,
        values,
        next_values,
        terminated=torch.tensor([True]),
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert torch.allclose(
        truncated_advantage,
        torch.tensor([2.3]),
    )

    assert torch.allclose(
        terminal_advantage,
        torch.tensor([0.5]),
    )


