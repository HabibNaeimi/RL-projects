import gymnasium as gym
import numpy as np
import pytest
import torch

from reinforce_cartpole import (
    SEED,
    PolicyNetwork,
    choose_action,
    collect_episode,
    compute_discounted_returns,
    policy_update,
    compute_reinforce_loss
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


def test_policy_output(env, policy):



    test_observation, _ = env.reset(seed=SEED)
    action, log_prob, probs = choose_action(policy, test_observation)

    assert action in (0, 1)
    assert probs.shape == (2,)
    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)

    assert log_prob.shape == torch.Size([])
    assert torch.isfinite(log_prob)
    assert log_prob.requires_grad

    print("Action:", action)
    print("Probabilities:", probs)
    print("Log-probabilities:", log_prob)


def test_sampling_behavior(env, policy):
    """
    Calls the policy repeatedly for the same observation
    """
    test_observation, _ = env.reset(seed=SEED)

    sampled_actions = [
        choose_action(policy, test_observation)[0]
        for _ in range(200)
    ]

    print(np.bincount(sampled_actions, minlength=2))


def test_episode_collection(episode):

    length = episode["episode_length"]

    assert length > 0
    assert length <= 500
    assert length == len(episode["actions"])
    assert length == len(episode["rewards"])
    assert length == len(episode["log_probs"])
    assert all(action in (0, 1) for action in episode["actions"])
    assert all(log_prob.requires_grad for log_prob in episode["log_probs"])

    # Standard CartPole gives +1 at every step.
    assert np.isclose(episode["episode_return"], length)


def test_gradient_connectivity(policy, episode):
    """
    This checks the graph without updating the policy
    """
    policy.zero_grad(set_to_none=True)

    test_loss = -episode["log_probs"][0]
    test_loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in policy.parameters()
    )


def test_discounted_returns():
    rewards = [1.0, 1.0, 1.0]

    returns = compute_discounted_returns(
        rewards,
        gamma=0.5,
    )

    expected = torch.tensor([
        1.75,  # 1 + 0.5(1) + 0.5²(1)
        1.50,  # 1 + 0.5(1)
        1.00,
    ])

    assert returns.shape == (3,)
    assert returns.dtype == torch.float32
    assert torch.allclose(returns, expected)
    assert not returns.requires_grad


def test_reinforce_loss():
    log_probs = [
        torch.tensor(-0.2, requires_grad=True),
        torch.tensor(-0.7, requires_grad=True),
    ]

    returns = torch.tensor([2.0, 1.0])

    loss = compute_reinforce_loss(log_probs, returns)

    # -((-0.2 × 2) + (-0.7 × 1)) = 1.1
    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, torch.tensor(1.1))
    assert loss.requires_grad

    loss.backward()

    assert torch.isclose(
        log_probs[0].grad,
        torch.tensor(-2.0),
    )
    assert torch.isclose(
        log_probs[1].grad,
        torch.tensor(-1.0),
    )


def test_policy_update_changes_parameters(env, policy):
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=1e-2,
    )

    episode = collect_episode(env, policy, seed=SEED)

    parameters_before = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]

    loss_value = policy_update(
        optimizer,
        episode,
        gamma=0.99,
    )

    parameters_changed = any(
        not torch.allclose(before, after)
        for before, after in zip(
            parameters_before,
            policy.parameters(),
        )
    )

    assert np.isfinite(loss_value)
    assert parameters_changed