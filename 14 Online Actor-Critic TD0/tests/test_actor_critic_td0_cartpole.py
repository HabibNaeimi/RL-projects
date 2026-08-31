import gymnasium as gym
import numpy as np
import pytest
import torch

from actor_critic_td0_cartpole import (
    PolicyNetwork,
    SEED,
    collect_episode,
    ValueNetwork,
    train_one_actor_critic_episode,
    compute_actor_critic_losses,
    compute_td_target,    
    ENTROPY_COEF,
)


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


def test_td_target_bootstrapping_and_detachment():
    next_value = torch.tensor(
        4.0,
        requires_grad=True,
    )

    continuing_target = compute_td_target(
        reward=1.0,
        next_value=next_value,
        terminated=False,
        gamma=0.9,
    )

    terminal_target = compute_td_target(
        reward=1.0,
        next_value=next_value,
        terminated=True,
        gamma=0.9,
    )

    assert continuing_target.shape == torch.Size([])
    assert torch.isclose(
        continuing_target,
        torch.tensor(4.6),
    )

    assert torch.isclose(
        terminal_target,
        torch.tensor(1.0),
    )

    assert not continuing_target.requires_grad
    assert not terminal_target.requires_grad



def test_actor_critic_gradient_separation():
    log_prob = torch.tensor(
        -0.4,
        requires_grad=True,
    )

    entropy = torch.tensor(
        0.5,
        requires_grad=True,
    )

    value = torch.tensor(
        2.0,
        requires_grad=True,
    )

    td_target = torch.tensor(3.0)

    actor_loss, critic_loss, td_error = (
        compute_actor_critic_losses(
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            td_target=td_target,
            entropy_coef=0.1,
        )
    )

    assert torch.isclose(
        td_error,
        torch.tensor(1.0),
    )

    # 0.4 - 0.1 × 0.5 = 0.35
    assert torch.isclose(
        actor_loss,
        torch.tensor(0.35),
    )

    assert torch.isclose(
        critic_loss,
        torch.tensor(1.0),
    )

    actor_loss.backward()

    assert value.grad is None
    assert torch.isclose(
        log_prob.grad,
        torch.tensor(-1.0),
    )
    assert torch.isclose(
        entropy.grad,
        torch.tensor(-0.1),
    )

    critic_loss.backward()

    assert value.grad is not None
    assert torch.isclose(
        value.grad,
        torch.tensor(-2.0),
    )


def test_one_episode_updates_both_networks():
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")

    policy = PolicyNetwork(
        obs_dim=4,
        n_actions=2,
    )
    value_network = ValueNetwork(obs_dim=4)

    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=1e-3,
    )
    value_optimizer = torch.optim.Adam(
        value_network.parameters(),
        lr=1e-3,
    )

    policy_before = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]
    value_before = [
        parameter.detach().clone()
        for parameter in value_network.parameters()
    ]

    metrics = train_one_actor_critic_episode(
        env=env,
        policy=policy,
        value_network=value_network,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        gamma=0.99,
        entropy_coef=0.01,
        seed=SEED,
    )

    assert metrics["episode_length"] > 0
    assert metrics["episode_length"] <= 500
    assert np.isclose(
        metrics["episode_return"],
        metrics["episode_length"],
    )

    assert any(
        not torch.allclose(before, after)
        for before, after in zip(
            policy_before,
            policy.parameters(),
        )
    )

    assert any(
        not torch.allclose(before, after)
        for before, after in zip(
            value_before,
            value_network.parameters(),
        )
    )

    for value in metrics.values():
        assert np.isfinite(value)

    env.close()