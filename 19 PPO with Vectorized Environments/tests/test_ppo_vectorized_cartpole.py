
import math
import re

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical

import ppo_vectorized_cartpole as ppo


class CountingEnv(gym.Env):
    """Deterministic episodes with visibly different final/reset observations."""

    def __init__(self, env_id, episode_length, ending_type, reward):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self.env_id = env_id
        self.episode_length = episode_length
        self.ending_type = ending_type
        self.reward = reward
        self.episode_number = -1
        self.step_number = 0
        self.reset_seeds = []

    def _observation(self):
        return np.asarray(
            [self.env_id, self.episode_number, self.step_number, 0.0],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_seeds.append(seed)
        self.episode_number += 1
        self.step_number = 0
        return self._observation(), {}

    def step(self, action):
        self.step_number += 1
        ended = self.step_number == self.episode_length
        terminated = ended and self.ending_type == "terminated"
        truncated = ended and self.ending_type == "truncated"
        return self._observation(), self.reward, terminated, truncated, {}


class RecordingPolicy(nn.Module):
    """State-dependent logits; records which observations the collector uses."""

    def __init__(self):
        super().__init__()
        self.obs_dim = 4
        self.inputs = []

    def forward(self, observations):
        self.inputs.append(observations.detach().clone())
        score = 0.1 * observations[:, 0] + 0.25 * observations[:, 2]
        return torch.stack((score, -score), dim=-1)


@pytest.fixture
def vector_setup():
    env = gym.vector.SyncVectorEnv(
        [
            lambda: CountingEnv(0, 2, "terminated", 1.0),
            lambda: CountingEnv(1, 3, "truncated", 2.0),
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    observations, _ = env.reset(seed=101)
    returns = np.zeros(env.num_envs, dtype=np.float64)
    lengths = np.zeros(env.num_envs, dtype=np.int64)
    policy = RecordingPolicy()
    try:
        yield env, policy, observations, returns, lengths
    finally:
        env.close()


def collect(setup, steps):
    env, policy, observations, returns, lengths = setup
    # The collector accepts lengths before returns, but returns them in the
    # opposite order. Positional arguments also work with singular/plural names.
    return ppo.collect_vector_rollout(
        env, policy, observations, lengths, returns, steps
    )


def test_make_vector_env_creates_independent_same_step_cartpoles():
    # Deliberately different from the module's NUM_ENVS constant.
    env = ppo.make_vector_env(3)
    try:
        assert isinstance(env, gym.vector.SyncVectorEnv)
        assert env.num_envs == 3
        assert env.autoreset_mode == gym.vector.AutoresetMode.SAME_STEP
        assert len({id(subenv.unwrapped) for subenv in env.envs}) == 3
        assert all(subenv.spec.id == "CartPole-v1" for subenv in env.envs)

        observations, _ = env.reset(seed=123)
        assert observations.shape == (3, 4)
        assert env.single_action_space.n == 2
        assert env.single_observation_space.shape == (4,)
    finally:
        env.close()


def test_rollout_to_tensors_preserves_vector_shapes_dtypes_and_values():
    T, N, D = 3, 2, 4
    state_data = np.arange(T * N * D, dtype=np.float32).reshape(T, N, D)
    rollout = {
        "states": state_data,
        "actions": [[0, 1], [1, 0], [0, 1]],
        "rewards": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "next_states": state_data + 1.0,
        "terminated": [[False, False], [True, False], [False, False]],
        "episode_ends": [[False, False], [True, True], [False, False]],
        "old_log_probs": [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]],
    }

    states, actions, rewards, next_states, terminated, ends, old_logs = (
        ppo.rollout_to_tensors(rollout)
    )

    assert states.shape == next_states.shape == (T, N, D)
    assert all(x.shape == (T, N) for x in (actions, rewards, terminated, ends, old_logs))
    assert states.dtype == next_states.dtype == rewards.dtype == old_logs.dtype == torch.float32
    assert actions.dtype == torch.int64
    assert terminated.dtype == ends.dtype == torch.bool
    assert not old_logs.requires_grad
    torch.testing.assert_close(states, torch.from_numpy(state_data))
    torch.testing.assert_close(next_states, torch.from_numpy(state_data + 1.0))
    torch.testing.assert_close(actions, torch.tensor(rollout["actions"]))
    torch.testing.assert_close(rewards, torch.tensor(rollout["rewards"]))
    torch.testing.assert_close(old_logs, torch.tensor(rollout["old_log_probs"]))
    assert torch.equal(terminated, torch.tensor(rollout["terminated"]))
    assert torch.equal(ends, torch.tensor(rollout["episode_ends"]))


def test_vector_collector_handles_asynchronous_boundaries_and_final_observations(vector_setup):
    env, _, _, _, _ = vector_setup
    rollout, observations, returns, lengths, completed_returns, completed_lengths = (
        collect(vector_setup, steps=4)
    )

    states = np.asarray(rollout["states"])
    next_states = np.asarray(rollout["next_states"])
    assert states.shape == next_states.shape == (4, 2, 4)
    for name in ("actions", "rewards", "terminated", "episode_ends", "old_log_probs"):
        assert np.asarray(rollout[name]).shape == (4, 2)

    np.testing.assert_array_equal(
        rollout["terminated"],
        [[False, False], [True, False], [False, False], [True, False]],
    )
    np.testing.assert_array_equal(
        rollout["episode_ends"],
        [[False, False], [True, False], [False, True], [True, False]],
    )
    np.testing.assert_array_equal(rollout["rewards"], [[1.0, 2.0]] * 4)
    assert completed_returns == [2.0, 6.0, 2.0]
    assert completed_lengths == [2, 3, 2]
    assert all(isinstance(x, float) for x in completed_returns)
    assert all(isinstance(x, int) for x in completed_lengths)
    np.testing.assert_array_equal(returns, [0.0, 2.0])
    np.testing.assert_array_equal(lengths, [0, 1])

    # Stored next_states belong to the OLD episode, including on truncation.
    np.testing.assert_array_equal(next_states[1, 0], [0.0, 0.0, 2.0, 0.0])
    np.testing.assert_array_equal(next_states[2, 1], [1.0, 0.0, 3.0, 0.0])
    # The next acting states instead belong to the automatically reset episodes.
    np.testing.assert_array_equal(states[2, 0], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(states[3, 1], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(observations[0], [0.0, 2.0, 0.0, 0.0])
    np.testing.assert_array_equal(observations[1], [1.0, 1.0, 1.0, 0.0])

    # No manual whole-vector resets or repeated seeding by the collector.
    assert env.envs[0].reset_seeds == [101, None, None]
    assert env.envs[1].reset_seeds == [102, None]


def test_vector_collector_refreshes_policy_inputs_and_stores_aligned_old_logs(vector_setup):
    _, policy, _, _, _ = vector_setup
    rollout, *_ = collect(vector_setup, steps=4)
    states, actions, _, _, _, _, old_logs = ppo.rollout_to_tensors(rollout)

    assert len(policy.inputs) == 4
    for t, policy_input in enumerate(policy.inputs):
        torch.testing.assert_close(policy_input, states[t])
    assert not torch.equal(policy.inputs[0], policy.inputs[1])

    with torch.no_grad():
        expected_logs = Categorical(
            logits=policy(states.reshape(8, 4))
        ).log_prob(actions.reshape(8)).reshape(4, 2)
    torch.testing.assert_close(old_logs, expected_logs)
    assert not old_logs.requires_grad
    assert torch.isfinite(old_logs).all()


def test_vector_collector_carries_partial_episodes_across_rollouts(vector_setup):
    env, policy, _, _, _ = vector_setup
    first, observations, returns, lengths, *_ = collect(vector_setup, steps=4)
    observation_before = observations.copy()
    completed_before = list(first["rewards"])

    second, observations, returns, lengths, completed_returns, completed_lengths = (
        ppo.collect_vector_rollout(env, policy, observations, lengths, returns, 2)
    )

    np.testing.assert_array_equal(second["states"][0], observation_before)
    assert np.asarray(second["states"]).shape == (2, 2, 4)
    # Environment 1's earlier one-step partial episode reaches total length 3.
    assert completed_returns == [2.0, 6.0]
    assert completed_lengths == [2, 3]
    np.testing.assert_array_equal(returns, [0.0, 0.0])
    np.testing.assert_array_equal(lengths, [0, 0])
    np.testing.assert_array_equal(observations, [[0, 3, 0, 0], [1, 2, 0, 0]])
    np.testing.assert_array_equal(first["rewards"], completed_before)


def test_vector_collector_allows_a_rollout_without_completed_episodes(vector_setup):
    _, _, initial_observations, _, _ = vector_setup
    initial_copy = initial_observations.copy()
    rollout, observations, returns, lengths, completed_returns, completed_lengths = (
        collect(vector_setup, steps=1)
    )

    assert completed_returns == []
    assert completed_lengths == []
    np.testing.assert_array_equal(returns, [1.0, 2.0])
    np.testing.assert_array_equal(lengths, [1, 1])
    np.testing.assert_array_equal(rollout["states"][0], initial_copy)
    np.testing.assert_array_equal(observations[:, 2], [1.0, 1.0])


def test_vector_gae_matches_hand_calculated_reverse_recurrence():
    rewards = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float64)
    values = torch.zeros_like(rewards)
    next_values = torch.zeros_like(rewards)
    flags = torch.zeros_like(rewards, dtype=torch.bool)

    advantages, targets, deltas = ppo.compute_gae(
        rewards, values, next_values, flags, flags, 0.9, 0.8
    )
    # gamma * lambda = 0.72; each column follows its own reverse recurrence.
    expected = torch.tensor(
        [[3.9952, 39.952], [4.16, 41.6], [3.0, 30.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(deltas, rewards)
    torch.testing.assert_close(advantages, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(targets, expected, rtol=1e-12, atol=1e-12)


def test_vector_gae_bootstraps_truncations_but_cuts_both_boundary_traces():
    rewards = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    values = rewards.clone()
    next_values = torch.tensor([[2.0, 20.0], [999.0, 30.0], [4.0, 999.0], [5.0, 50.0]])
    terminated = torch.tensor([[False, False], [True, False], [False, True], [False, False]])
    episode_ends = torch.tensor([[False, False], [True, True], [True, True], [False, False]])

    advantages, targets, deltas = ppo.compute_gae(
        rewards, values, next_values, terminated, episode_ends, 1.0, 1.0
    )
    # The 999 values must be ignored at true terminations. At truncations,
    # use the actual next value, but never leak the NEW episode's advantages.
    torch.testing.assert_close(deltas, torch.tensor([[2.0, 20.0], [0.0, 30.0], [4.0, 0.0], [5.0, 50.0]]))
    torch.testing.assert_close(advantages, torch.tensor([[2.0, 50.0], [0.0, 30.0], [4.0, 0.0], [5.0, 50.0]]))
    torch.testing.assert_close(targets, advantages + values)


def test_vector_gae_does_not_mix_environment_columns():
    rewards = torch.ones((3, 2))
    zeros = torch.zeros_like(rewards)
    terminated = torch.zeros_like(rewards, dtype=torch.bool)
    episode_ends = torch.tensor([[False, True], [True, False], [False, False]])
    baseline, _, _ = ppo.compute_gae(
        rewards, zeros, zeros, terminated, episode_ends, 1.0, 1.0
    )
    torch.testing.assert_close(baseline, torch.tensor([[2.0, 1.0], [1.0, 2.0], [1.0, 1.0]]))

    changed_rewards = rewards.clone()
    changed_rewards[:, 1] *= 1000.0
    changed, _, _ = ppo.compute_gae(
        changed_rewards, zeros, zeros, terminated, episode_ends, 1.0, 1.0
    )
    torch.testing.assert_close(changed[:, 0], baseline[:, 0])
    torch.testing.assert_close(changed[:, 1], baseline[:, 1] * 1000.0)


@pytest.mark.parametrize("T,N", [(1, 1), (1, 3), (4, 1)])
def test_vector_gae_preserves_singleton_dimensions(T, N):
    rewards = torch.ones((T, N))
    zeros = torch.zeros_like(rewards)
    flags = torch.zeros_like(rewards, dtype=torch.bool)
    outputs = ppo.compute_gae(rewards, zeros, zeros, flags, flags, 0.9, 0.0)
    for output in outputs:
        assert output.shape == (T, N)
        torch.testing.assert_close(output, rewards)


def test_vector_gae_returns_fixed_detached_targets():
    rewards = torch.ones((3, 2), requires_grad=True)
    values = torch.full((3, 2), 0.5, requires_grad=True)
    next_values = torch.full((3, 2), 0.8, requires_grad=True)
    flags = torch.zeros((3, 2), dtype=torch.bool)
    advantages, targets, deltas = ppo.compute_gae(
        rewards, values, next_values, flags, flags, 0.99, 0.95
    )
    for tensor in (advantages, targets, deltas):
        assert tensor.shape == (3, 2)
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
    torch.testing.assert_close(targets, advantages + values.detach())


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_flatten_vector_batch_preserves_every_field_alignment(noncontiguous):
    T, N, D = 3, 2, 2
    ids = torch.arange(T * N).reshape(T, N)
    states = torch.stack((ids.float(), ids.float() + 100.0), dim=-1)
    actions = ids.clone()
    old_logs = -(ids.float() + 1.0) / 10.0
    advantages = ids.float() + 0.5
    targets = ids.float() + 10.0
    originals = (states, actions, old_logs, advantages, targets)

    if noncontiguous:
        originals = tuple(
            x.transpose(0, 1).contiguous().transpose(0, 1) for x in originals
        )
        assert not originals[0].is_contiguous()

    flattened = ppo.flatten_vector_batch(*originals)
    assert len(flattened) == 5
    assert flattened[0].shape == (T * N, D)
    assert all(x.shape == (T * N,) for x in flattened[1:])

    # Check the mapping directly, not by another reshape implementation.
    for t in range(T):
        for n in range(N):
            flat_index = t * N + n
            for original, flat in zip(originals, flattened):
                torch.testing.assert_close(flat[flat_index], original[t, n])
    assert originals[0].shape == (T, N, D)


def test_minibatches_cover_flattened_sample_indices_exactly_once():
    batches = ppo.make_minibatch_indices(10, 4, device="cpu")
    assert [batch.numel() for batch in batches] == [4, 4, 2]
    assert all(batch.ndim == 1 for batch in batches)
    assert all(batch.dtype == torch.int64 for batch in batches)
    torch.testing.assert_close(torch.sort(torch.cat(batches)).values, torch.arange(10))


def test_short_vector_ppo_update_reuses_gae_and_updates_both_networks(vector_setup, monkeypatch):
    torch.manual_seed(7)
    env, _, observations, returns, lengths = vector_setup
    policy = ppo.PolicyNetwork(4, 2, hidden_dim=8)
    value_network = ppo.ValueNetwork(4, hidden_dim=8)
    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    critic_optimizer = torch.optim.Adam(value_network.parameters(), lr=3e-3)
    rollout, *_ = ppo.collect_vector_rollout(env, policy, observations, lengths, returns, 5)
    old_logs_before = np.asarray(rollout["old_log_probs"]).copy()
    actor_before = [p.detach().clone() for p in policy.parameters()]
    critic_before = [p.detach().clone() for p in value_network.parameters()]

    original_gae = ppo.compute_gae
    original_flatten = ppo.flatten_vector_batch
    gae_calls = []
    flatten_calls = []

    def tracked_gae(*args, **kwargs):
        gae_calls.append(args[0].shape)
        result = original_gae(*args, **kwargs)
        assert all(not x.requires_grad for x in result)
        return result

    def tracked_flatten(states, actions, logs, advantages, targets):
        flatten_calls.append(states.shape)
        assert advantages.shape == targets.shape == (5, 2)
        assert not advantages.requires_grad
        assert not targets.requires_grad
        # Normalize across the whole rollout, not separately per minibatch/env.
        assert advantages.mean().item() == pytest.approx(0.0, abs=2e-6)
        assert advantages.std(unbiased=False).item() == pytest.approx(1.0, abs=2e-6)
        return original_flatten(states, actions, logs, advantages, targets)

    monkeypatch.setattr(ppo, "compute_gae", tracked_gae)
    monkeypatch.setattr(ppo, "flatten_vector_batch", tracked_flatten)
    metrics = ppo.update_ppo_from_rollout(
        policy=policy,
        value_network=value_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        rollout=rollout,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        clip_eps=0.2,
        update_epochs=3,
        minibatch_size=4,
    )

    assert gae_calls == [torch.Size([5, 2])]
    assert flatten_calls == [torch.Size([5, 2, 4])]
    assert metrics["update_epoch"] == 3
    # 10 samples => batches of 4, 4, 2; three epochs => nine actor/critic pairs.
    assert metrics["optimizer_steps"] == 9
    for key in ("actor_loss", "critic_loss", "mean_entropy", "mean_ratio", "min_ratio", "max_ratio", "clip_fraction"):
        assert math.isfinite(metrics[key])
    assert 0.0 <= metrics["mean_entropy"] <= math.log(2.0) + 1e-6
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert metrics["min_ratio"] <= metrics["mean_ratio"] <= metrics["max_ratio"]
    np.testing.assert_array_equal(rollout["old_log_probs"], old_logs_before)
    assert any(not torch.equal(old, new.detach()) for old, new in zip(actor_before, policy.parameters()))
    assert any(not torch.equal(old, new.detach()) for old, new in zip(critic_before, value_network.parameters()))


def test_final_epoch_metrics_are_sample_weighted_for_unequal_minibatches(monkeypatch):
    T, N, D = 5, 2, 4
    policy = ppo.PolicyNetwork(D, 2, hidden_dim=8)
    value_network = ppo.ValueNetwork(D, hidden_dim=8)
    with torch.no_grad():
        for parameter in (*policy.parameters(), *value_network.parameters()):
            parameter.zero_()

    states = torch.zeros((T, N, D))
    actions = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(T, N)
    rewards = torch.arange(1, T * N + 1, dtype=torch.float32).reshape(T, N)
    with torch.no_grad():
        old_logs = Categorical(logits=policy(states)).log_prob(actions)

    rollout = {
        "states": states.numpy(),
        "actions": actions.numpy(),
        "rewards": rewards.numpy(),
        "next_states": states.numpy().copy(),
        "terminated": np.zeros((T, N), dtype=bool),
        "episode_ends": np.zeros((T, N), dtype=bool),
        "old_log_probs": old_logs.numpy().copy(),
    }
    entropy_coef, clip_eps = 0.01, 0.1
    with torch.no_grad():
        # gamma=0 and V=0 => raw advantages and value targets equal rewards.
        normalized_advantages = (rewards - rewards.mean()) / (
            rewards.std(unbiased=False) + torch.finfo(rewards.dtype).eps
        )
        policy.network[-1].bias[0] = 0.4
        distribution = Categorical(logits=policy(states))
        ratios = torch.exp(distribution.log_prob(actions) - old_logs)
        surrogate = torch.minimum(
            ratios * normalized_advantages,
            ratios.clamp(1.0 - clip_eps, 1.0 + clip_eps) * normalized_advantages,
        )
        expected = {
            "actor_loss": (-surrogate.mean() - entropy_coef * distribution.entropy().mean()).item(),
            "critic_loss": rewards.square().mean().item(),
            "mean_entropy": distribution.entropy().mean().item(),
            "mean_ratio": ratios.mean().item(),
            "min_ratio": ratios.min().item(),
            "max_ratio": ratios.max().item(),
            "clip_fraction": ((ratios - 1.0).abs() > clip_eps).float().mean().item(),
        }
    assert expected["critic_loss"] == pytest.approx(38.5)

    index_call_count = 0

    def ordered_minibatches(num_samples, minibatch_size, device=None):
        nonlocal index_call_count
        assert num_samples == 10
        assert minibatch_size == 4
        # Freeze each epoch's policy, but make the FIRST and FINAL epoch differ.
        # This catches accidentally averaging epochs as well as minibatches.
        with torch.no_grad():
            policy.network[-1].bias[0] = 0.0 if index_call_count == 0 else 0.4
        index_call_count += 1
        return [
            torch.arange(0, 4, device=device),
            torch.arange(4, 8, device=device),
            torch.arange(8, 10, device=device),
        ]

    monkeypatch.setattr(ppo, "make_minibatch_indices", ordered_minibatches)
    metrics = ppo.update_ppo_from_rollout(
        policy=policy,
        value_network=value_network,
        actor_optimizer=torch.optim.Adam(policy.parameters(), lr=0.0),
        critic_optimizer=torch.optim.Adam(value_network.parameters(), lr=0.0),
        rollout=rollout,
        gamma=0.0,
        gae_lambda=0.95,
        entropy_coef=entropy_coef,
        clip_eps=clip_eps,
        update_epochs=2,
        minibatch_size=4,
    )

    assert index_call_count == 2
    assert metrics["optimizer_steps"] == 6
    assert metrics["update_epoch"] == 2
    for key, expected_value in expected.items():
        assert metrics[key] == pytest.approx(expected_value, rel=2e-6, abs=2e-6)


def test_vector_training_carries_state_and_counts_individual_environment_steps(monkeypatch, capsys):
    class ResetOnlyVectorEnv:
        num_envs = 2

        def __init__(self):
            self.reset_seeds = []

        def reset(self, seed=None):
            self.reset_seeds.append(seed)
            return np.zeros((2, 4), dtype=np.float32), {}

    env = ResetOnlyVectorEnv()
    collector_inputs = []
    update_inputs = []
    outputs = [
        (
            {"number": 1},
            np.ones((2, 4), dtype=np.float32),
            np.asarray([0.5, 2.0]),
            np.asarray([1, 1]),
            [2.0, 6.0],
            [2, 3],
        ),
        (
            {"number": 2},
            np.full((2, 4), 2.0, dtype=np.float32),
            np.zeros(2),
            np.zeros(2, dtype=np.int64),
            [4.0],
            [4],
        ),
    ]

    def fake_collect(**kwargs):
        collector_inputs.append({
            "observations": kwargs["observations"].copy(),
            "returns": kwargs["running_episode_returns"].copy(),
            "lengths": kwargs["running_episode_lengths"].copy(),
            "steps": kwargs["steps_per_env"],
        })
        return outputs[len(collector_inputs) - 1]

    def fake_update(**kwargs):
        update_inputs.append(kwargs["rollout"])
        return {
            "update_epoch": 1,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "mean_entropy": 0.0,
            "mean_ratio": 1.0,
            "min_ratio": 1.0,
            "max_ratio": 1.0,
            "clip_fraction": 0.0,
            "optimizer_steps": 1,
        }

    monkeypatch.setattr(ppo, "collect_vector_rollout", fake_collect)
    monkeypatch.setattr(ppo, "update_ppo_from_rollout", fake_update)
    monkeypatch.setattr(ppo, "REPORT_EVERY_UPDATES", 1)
    returns, lengths = ppo.train(
        vector_env=env,
        policy=object(),
        value_network=object(),
        actor_optimizer=object(),
        critic_optimizer=object(),
        seed=11,
        num_updates=2,
        steps_per_env=3,
    )

    assert env.reset_seeds == [11]
    assert len(collector_inputs) == len(update_inputs) == 2
    assert all(call["steps"] == 3 for call in collector_inputs)
    np.testing.assert_array_equal(collector_inputs[0]["returns"], [0.0, 0.0])
    np.testing.assert_array_equal(collector_inputs[0]["lengths"], [0, 0])
    np.testing.assert_array_equal(collector_inputs[1]["observations"], np.ones((2, 4)))
    np.testing.assert_array_equal(collector_inputs[1]["returns"], [0.5, 2.0])
    np.testing.assert_array_equal(collector_inputs[1]["lengths"], [1, 1])
    assert returns == [2.0, 6.0, 4.0]
    assert lengths == [2, 3, 4]

    output = capsys.readouterr().out
    # Two updates * three vector steps * two environments = 12 transitions.
    assert re.findall(r"Environment steps:\s*(\d+)", output) == ["6", "12"]
