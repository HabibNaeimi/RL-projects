import math

import numpy as np
import pytest
import torch
from torch import nn

import ppo_minibatch_cartpole as ppo


class AlwaysActionZeroPolicy(nn.Module):
    """A deterministic categorical policy for collector tests."""

    def __init__(self, obs_dim=2):
        super().__init__()
        self.obs_dim = obs_dim

    def forward(self, observations):
        logits = torch.full(
            (observations.shape[0], 2),
            -torch.inf,
            dtype=observations.dtype,
            device=observations.device,
        )
        logits[:, 0] = 0.0
        return logits


class ScriptedEnv:
    """Small environment with known episode lengths and ending types."""

    def __init__(
        self,
        episode_lengths=(2, 3, 4, 2),
        ending_types=("terminated", "truncated", "terminated", "truncated"),
    ):
        assert len(episode_lengths) == len(ending_types)
        self.episode_lengths = tuple(episode_lengths)
        self.ending_types = tuple(ending_types)
        self.episode_index = -1
        self.step_in_episode = 0
        self.reset_seeds = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.episode_index += 1
        self.step_in_episode = 0
        return self._observation(), {}

    def step(self, action):
        self.step_in_episode += 1

        table_index = self.episode_index % len(self.episode_lengths)
        episode_length = self.episode_lengths[table_index]
        ending_type = self.ending_types[table_index]
        episode_end = self.step_in_episode == episode_length

        terminated = episode_end and ending_type == "terminated"
        truncated = episode_end and ending_type == "truncated"

        return self._observation(), 1.0, terminated, truncated, {}

    def _observation(self):
        return np.asarray(
            [self.episode_index, self.step_in_episode],
            dtype=np.float32,
        )


def test_rollout_to_tensors_preserves_shapes_dtypes_and_alignment():
    rollout = {
        "states": [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]],
        "actions": [0, 1, 0],
        "rewards": [1.0, -1.0, 2.0],
        "next_states": [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
        "terminated": [False, True, False],
        "episode_ends": [False, True, True],
        "old_log_probs": [-0.2, -0.7, -0.4],
    }

    (
        states,
        actions,
        rewards,
        next_states,
        terminated,
        episode_ends,
        old_log_probs,
    ) = ppo.rollout_to_tensors(rollout)

    assert states.shape == (3, 2)
    assert next_states.shape == (3, 2)
    assert actions.shape == rewards.shape == (3,)
    assert terminated.shape == episode_ends.shape == old_log_probs.shape == (3,)

    assert states.dtype == torch.float32
    assert next_states.dtype == torch.float32
    assert rewards.dtype == torch.float32
    assert old_log_probs.dtype == torch.float32
    assert actions.dtype == torch.int64
    assert terminated.dtype == torch.bool
    assert episode_ends.dtype == torch.bool
    assert not old_log_probs.requires_grad


def test_compute_gae_matches_exact_reverse_time_recurrence():
    rewards = torch.ones(3)
    values = torch.zeros(3)
    next_values = torch.zeros(3)
    terminated = torch.zeros(3, dtype=torch.bool)
    episode_ends = torch.zeros(3, dtype=torch.bool)

    advantages, value_targets, td_errors = ppo.compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        episode_ends,
        gamma=0.9,
        gae_lambda=0.8,
    )

    # gamma * lambda = 0.72:
    # A_2 = 1
    # A_1 = 1 + 0.72 * 1       = 1.72
    # A_0 = 1 + 0.72 * 1.72    = 2.2384
    expected_advantages = torch.tensor([2.2384, 1.72, 1.0])

    assert torch.allclose(td_errors, torch.ones(3), atol=1e-6)
    assert torch.allclose(advantages, expected_advantages, atol=1e-6)
    assert torch.allclose(value_targets, expected_advantages, atol=1e-6)
    assert not advantages.requires_grad
    assert not value_targets.requires_grad


def test_compute_gae_uses_different_bootstrap_and_trace_masks():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    next_values = torch.tensor([2.0, 999.0, 4.0, 5.0])

    # t=1 is a true termination: no bootstrapping and no trace continuation.
    # t=2 is a truncation: bootstrap from next_values[2], but stop the trace.
    terminated = torch.tensor([False, True, False, False])
    episode_ends = torch.tensor([False, True, True, False])

    advantages, value_targets, td_errors = ppo.compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        episode_ends,
        gamma=1.0,
        gae_lambda=1.0,
    )

    expected_td_errors = torch.tensor([2.0, 0.0, 4.0, 5.0])
    expected_advantages = torch.tensor([2.0, 0.0, 4.0, 5.0])
    expected_value_targets = torch.tensor([3.0, 2.0, 7.0, 9.0])

    assert torch.equal(td_errors, expected_td_errors)
    assert torch.equal(advantages, expected_advantages)
    assert torch.equal(value_targets, expected_value_targets)


def test_calculate_ppo_terms_applies_sign_aware_clipped_surrogate():
    expected_ratios = torch.tensor([1.5, 0.5, 1.5, 0.5])
    old_log_probs = torch.full((4,), -2.0)
    new_log_probs = (
        old_log_probs + torch.log(expected_ratios)
    ).detach().requires_grad_(True)
    advantages = torch.tensor([1.0, -1.0, -1.0, 1.0])

    terms = ppo.calculate_ppo_terms(
        new_actions_log_probs=new_log_probs,
        old_actions_log_probs=old_log_probs,
        advantages=advantages,
        clip_eps=0.2,
    )

    assert torch.allclose(terms["ratios"], expected_ratios)
    assert torch.allclose(
        terms["clipped_ratios"],
        torch.tensor([1.2, 0.8, 1.2, 0.8]),
    )
    assert torch.allclose(
        terms["unclipped_surrogate"],
        torch.tensor([1.5, -0.5, -1.5, 0.5]),
    )
    assert torch.allclose(
        terms["clipped_surrogate"],
        torch.tensor([1.2, -0.8, -1.2, 0.8]),
    )
    assert torch.allclose(
        terms["conservative_surrogate"],
        torch.tensor([1.2, -0.8, -1.5, 0.5]),
    )

    loss = -terms["conservative_surrogate"].mean()
    loss.backward()

    assert new_log_probs.grad is not None
    assert torch.isfinite(new_log_probs.grad).all()


def test_collect_rollout_has_fixed_length_and_continues_partial_episode():
    env = ScriptedEnv()
    policy = AlwaysActionZeroPolicy()
    observation, _ = env.reset(seed=123)

    (
        first_rollout,
        observation,
        running_return,
        running_length,
        completed_returns,
        completed_lengths,
    ) = ppo.collect_rollout(
        env=env,
        policy=policy,
        observation=observation,
        running_episode_return=0.0,
        running_episode_length=0,
        rollout_steps=7,
    )

    assert len(first_rollout["states"]) == 7
    assert len(first_rollout["actions"]) == 7
    assert len(first_rollout["old_log_probs"]) == 7
    assert first_rollout["terminated"] == [False, True, False, False, False, False, False]
    assert first_rollout["episode_ends"] == [False, True, False, False, True, False, False]
    assert completed_returns == [2.0, 3.0]
    assert completed_lengths == [2, 3]

    # Two steps of the four-step third episode remain in progress.
    assert running_return == pytest.approx(2.0)
    assert running_length == 2
    assert np.array_equal(observation, np.asarray([2.0, 2.0], dtype=np.float32))

    # A normal transition points directly to the following stored state.
    assert np.array_equal(first_rollout["next_states"][0], first_rollout["states"][1])
    # Across a boundary, next_state is the final state, not the reset state.
    assert not np.array_equal(first_rollout["next_states"][1], first_rollout["states"][2])

    tensors = ppo.rollout_to_tensors(first_rollout)
    states, actions, *_, stored_old_log_probs = tensors
    with torch.no_grad():
        expected_old_log_probs = torch.distributions.Categorical(
            logits=policy(states)
        ).log_prob(actions)
    assert torch.allclose(stored_old_log_probs, expected_old_log_probs)

    (
        second_rollout,
        observation,
        running_return,
        running_length,
        completed_returns,
        completed_lengths,
    ) = ppo.collect_rollout(
        env=env,
        policy=policy,
        observation=observation,
        running_episode_return=running_return,
        running_episode_length=running_length,
        rollout_steps=3,
    )

    assert np.array_equal(
        second_rollout["states"][0],
        np.asarray([2.0, 2.0], dtype=np.float32),
    )
    assert completed_returns == [4.0]
    assert completed_lengths == [4]
    assert running_return == pytest.approx(1.0)
    assert running_length == 1

    # Only the initial reset is explicitly seeded. Training resets are unseeded.
    assert env.reset_seeds == [123, None, None, None]


def test_update_ppo_from_rollout_is_finite_and_updates_both_networks(monkeypatch):
    torch.manual_seed(7)
    env = ScriptedEnv(
        episode_lengths=(3, 4, 5),
        ending_types=("terminated", "truncated", "terminated"),
    )
    observation, _ = env.reset(seed=7)

    policy = ppo.PolicyNetwork(obs_dim=2, n_actions=2, hidden_dim=8)
    value_network = ppo.ValueNetwork(obs_dim=2, hidden_dim=8)
    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    critic_optimizer = torch.optim.Adam(value_network.parameters(), lr=3e-3)

    rollout, *_ = ppo.collect_rollout(
        env=env,
        policy=policy,
        observation=observation,
        running_episode_return=0.0,
        running_episode_length=0,
        rollout_steps=24,
    )

    old_log_probs_before = list(rollout["old_log_probs"])
    actor_parameters_before = [parameter.detach().clone() for parameter in policy.parameters()]
    critic_parameters_before = [
        parameter.detach().clone() for parameter in value_network.parameters()
    ]

    original_compute_gae = ppo.compute_gae
    gae_call_count = 0

    def counted_compute_gae(*args, **kwargs):
        nonlocal gae_call_count
        gae_call_count += 1
        return original_compute_gae(*args, **kwargs)

    monkeypatch.setattr(ppo, "compute_gae", counted_compute_gae)

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
        minibatch_size=7,
    )

    # GAE targets and normalized actor advantages must be computed only once,
    # then reused across all three PPO epochs.
    assert gae_call_count == 1
    assert metrics["update_epoch"] == 3
    assert metrics["optimizer_steps"] == 12
    assert 0.0 <= metrics["mean_entropy"] <= math.log(2) + 1e-6

    for key in (
        "actor_loss",
        "critic_loss",
        "mean_entropy",
        "mean_ratio",
        "min_ratio",
        "max_ratio",
        "clip_fraction",
    ):
        assert math.isfinite(metrics[key])

    assert metrics["min_ratio"] <= metrics["mean_ratio"] <= metrics["max_ratio"]
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert rollout["old_log_probs"] == old_log_probs_before

    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(actor_parameters_before, policy.parameters())
    )
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(critic_parameters_before, value_network.parameters())
    )


def test_train_counts_updates_and_flattens_completed_episode_results(monkeypatch):
    class ResetOnlyEnv:
        def reset(self, seed=None):
            return np.asarray([0.0, 0.0], dtype=np.float32), {}

    collector_inputs = []
    update_rollouts = []

    collector_outputs = [
        (
            {"rollout_number": 1},
            np.asarray([1.0, 1.0], dtype=np.float32),
            0.5,
            1,
            [10.0, 20.0],
            [2, 3],
        ),
        (
            {"rollout_number": 2},
            np.asarray([2.0, 2.0], dtype=np.float32),
            0.0,
            0,
            [30.0],
            [4],
        ),
    ]

    def fake_collect_rollout(*args, **kwargs):
        collector_inputs.append(kwargs)
        return collector_outputs[len(collector_inputs) - 1]

    def fake_update_ppo_from_rollout(*args, **kwargs):
        update_rollouts.append(kwargs["rollout"])
        return {
            "update_epoch": 1,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "mean_entropy": 0.0,
            "mean_ratio": 1.0,
            "min_ratio": 1.0,
            "max_ratio": 1.0,
            "clip_fraction": 0.0,
        }

    monkeypatch.setattr(ppo, "collect_rollout", fake_collect_rollout)
    monkeypatch.setattr(ppo, "update_ppo_from_rollout", fake_update_ppo_from_rollout)
    monkeypatch.setattr(ppo, "REPORT_EVERY_UPDATES", 10_000)

    episode_returns, episode_lengths = ppo.train(
        env=ResetOnlyEnv(),
        policy=object(),
        value_network=object(),
        actor_optimizer=object(),
        critic_optimizer=object(),
        seed=5,
        num_updates=2,
        rollout_steps=4,
    )

    assert len(collector_inputs) == 2
    assert len(update_rollouts) == 2
    assert all(call["rollout_steps"] == 4 for call in collector_inputs)

    # The unfinished episode state returned by rollout 1 reaches rollout 2.
    assert collector_inputs[1]["running_episode_return"] == pytest.approx(0.5)
    assert collector_inputs[1]["running_episode_length"] == 1
    assert np.array_equal(
        collector_inputs[1]["observation"],
        np.asarray([1.0, 1.0], dtype=np.float32),
    )

    # Results are flat episode histories, not one nested list per update.
    assert episode_returns == [10.0, 20.0, 30.0]
    assert episode_lengths == [2, 3, 4]


def test_evaluate_policy_collects_complete_episodes_with_distinct_seeds():
    env = ScriptedEnv(
        episode_lengths=(2, 3, 4),
        ending_types=("terminated", "truncated", "terminated"),
    )
    policy = AlwaysActionZeroPolicy()

    returns = ppo.evaluate_policy(
        env=env,
        policy=policy,
        num_episodes=3,
        base_seed=50,
    )

    assert returns == [2.0, 3.0, 4.0]
    assert env.reset_seeds == [50, 51, 52]


def test_minibatches_cover_every_sample_once():
    torch.manual_seed(123)

    batches = ppo.make_minibatch_indices(
        num_samples=10,
        minibatch_size=4,
        device="cpu",
    )

    assert [batch.numel() for batch in batches] == [4, 4, 2]

    combined = torch.cat(batches)

    assert combined.dtype == torch.int64
    assert torch.equal(
        torch.sort(combined).values,
        torch.arange(10),
    )

def test_invalid_minibatch_arguments():
    with pytest.raises(ValueError):
        ppo.make_minibatch_indices(10, 0)

    with pytest.raises(ValueError):
        ppo.make_minibatch_indices(0, 4)


