import gymnasium as gym
import numpy as np
import pytest
import torch

from reinforce_value_baseline_cartpole import (
    PolicyNetwork,
    SEED,
    collect_episode,
    ValueNetwork,
    compute_loss_with_baseline,
    update_policy_and_value, 
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




def test_value_network_output():
    value_network = ValueNetwork(
        obs_dim=4,
        hidden_dim=32,
    )

    observations = torch.zeros(7, 4)
    predicted_values = value_network(observations)

    assert predicted_values.shape == (7,)
    assert predicted_values.requires_grad
    assert torch.all(torch.isfinite(predicted_values))


def test_losses_with_baseline():
    log_probs = [
        torch.tensor(-0.2, requires_grad=True),
        torch.tensor(-0.7, requires_grad=True),
    ]

    returns = torch.tensor([3.0, 1.0])

    predicted_values = torch.tensor(
        [1.0, 2.0],
        requires_grad=True,
    )

    policy_loss, value_loss, advantages = (
        compute_loss_with_baseline(
            log_probs,
            returns,
            predicted_values,
        )
    )

    expected_advantages = torch.tensor([2.0, -1.0])

    assert torch.allclose(
        advantages,
        expected_advantages,
    )

    # -((-0.2 × 2) + (-0.7 × -1)) = -0.3
    assert torch.isclose(
        policy_loss,
        torch.tensor(-0.3),
        atol=1e-6,
    )

    # ((1 - 3)² + (2 - 1)²) / 2 = 2.5
    assert torch.isclose(
        value_loss,
        torch.tensor(2.5),
    )

    policy_loss.backward()

    # Policy loss must not train the value predictions.
    assert predicted_values.grad is None

    assert torch.isclose(
        log_probs[0].grad,
        torch.tensor(-2.0),
    )
    assert torch.isclose(
        log_probs[1].grad,
        torch.tensor(1.0),
    )

    value_loss.backward()

    assert predicted_values.grad is not None
    assert torch.allclose(
        predicted_values.grad,
        torch.tensor([-2.0, 1.0]),
    )


def test_update_changes_both_networks(env, policy):
    value_network = ValueNetwork(
        obs_dim=env.observation_space.shape[0],
    )

    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=1e-2,
    )

    value_optimizer = torch.optim.Adam(
        value_network.parameters(),
        lr=1e-3,
    )

    episode = collect_episode(
        env,
        policy,
        seed=SEED,
    )

    policy_before = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]

    value_before = [
        parameter.detach().clone()
        for parameter in value_network.parameters()
    ]

    metrics = update_policy_and_value(
        policy_optimizer,
        value_optimizer,
        value_network,
        episode,
        gamma=0.99,
    )

    policy_changed = any(
        not torch.allclose(before, after)
        for before, after in zip(
            policy_before,
            policy.parameters(),
        )
    )

    value_changed = any(
        not torch.allclose(before, after)
        for before, after in zip(
            value_before,
            value_network.parameters(),
        )
    )

    assert policy_changed
    assert value_changed
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])







    