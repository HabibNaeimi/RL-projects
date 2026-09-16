"""Level 20: KL diagnostics and complete-epoch PPO early stopping.

The signed estimator is allowed to be negative. The nonnegative estimator is
mean(expm1(new_log_probs - old_log_probs) - (new_log_probs - old_log_probs)).

These tests use tiny deterministic batches. There is no training-score target.
AssertionError or ValueError is accepted for input-validation failures.
"""

import math
import re
import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Categorical

import ppo_kl_cartpole as ppo


VALIDATION_ERRORS = (AssertionError, ValueError)
LOSS_METRICS = (
    "actor_loss", "critic_loss", "mean_entropy", "mean_ratio",
    "min_ratio", "max_ratio", "clip_fraction",
)


def assert_diagnostic_dict(result):
    assert isinstance(result, dict), "Return a dictionary of named diagnostics."
    for key in ("approx_kl", "signed_approx_kl"):
        assert key in result
        assert type(result[key]) is float, f"{key} must be a Python float."
        assert math.isfinite(result[key])


def reference_diagnostics(new_log_probs, old_log_probs):
    """A test oracle: no call to the student's diagnostic function."""
    with torch.no_grad():
        delta = new_log_probs - old_log_probs
        return {
            "approx_kl": (torch.expm1(delta) - delta).mean().item(),
            "signed_approx_kl": (-delta.mean()).item(),
        }


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, lr=0.0):
        super().__init__(parameters, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        result = super().step(closure=closure)
        self.step_calls += 1
        return result


def tiny_training_setup(actor_lr=0.0, critic_lr=0.0):
    """Five timesteps, two environments => ten aligned training samples."""
    torch.manual_seed(11)
    policy = ppo.PolicyNetwork(4, 2, hidden_dim=8)
    value_network = ppo.ValueNetwork(4, hidden_dim=8)
    with torch.no_grad():
        for parameter in (*policy.parameters(), *value_network.parameters()):
            parameter.zero_()

    states = torch.zeros((5, 2, 4))
    actions = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(5, 2)
    rewards = torch.arange(1, 11, dtype=torch.float32).reshape(5, 2)
    with torch.no_grad():
        old_logs = Categorical(logits=policy(states)).log_prob(actions)

    rollout = {
        "states": states.numpy().copy(),
        "actions": actions.numpy().copy(),
        "rewards": rewards.numpy().copy(),
        "next_states": states.numpy().copy(),
        "terminated": np.zeros((5, 2), dtype=bool),
        "episode_ends": np.zeros((5, 2), dtype=bool),
        "old_log_probs": old_logs.numpy().copy(),
    }
    actor_optimizer = CountingSGD(policy.parameters(), lr=actor_lr)
    critic_optimizer = CountingSGD(value_network.parameters(), lr=critic_lr)
    return policy, value_network, actor_optimizer, critic_optimizer, rollout


def run_update(setup, *, epochs, target_kl, stop_multiplier=1.5):
    policy, value_network, actor_optimizer, critic_optimizer, rollout = setup
    return ppo.update_ppo_from_rollout(
        policy=policy,
        value_network=value_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        rollout=rollout,
        gamma=0.0,
        gae_lambda=0.95,
        entropy_coef=0.01,
        clip_eps=0.1,
        update_epochs=epochs,
        minibatch_size=4,
        target_kl=target_kl,
        stop_multiplier=stop_multiplier,
    )


def assert_update_metrics(metrics, *, epochs, early_stopped, approx_kl):
    assert_diagnostic_dict(metrics)
    assert metrics["update_epoch"] == epochs
    assert metrics["optimizer_steps"] == 3 * epochs  # batches of 4, 4, 2
    assert type(metrics["early_stopped"]) is bool
    assert metrics["early_stopped"] is early_stopped
    assert type(metrics["approx_kl"]) is float
    assert metrics["approx_kl"] == pytest.approx(approx_kl, rel=2e-6, abs=1e-8)
    assert all(math.isfinite(metrics[key]) for key in LOSS_METRICS)
    assert 0.0 <= metrics["mean_entropy"] <= math.log(2.0) + 1e-6
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert metrics["min_ratio"] <= metrics["mean_ratio"] <= metrics["max_ratio"]


# Retained vector-batch behavior from Level 19.

def test_vector_environment_still_uses_independent_same_step_cartpoles():
    env = ppo.make_vector_env(3)
    try:
        assert isinstance(env, gym.vector.SyncVectorEnv)
        assert env.num_envs == 3
        assert env.autoreset_mode == gym.vector.AutoresetMode.SAME_STEP
        assert len({id(subenv.unwrapped) for subenv in env.envs}) == 3
        assert all(subenv.spec.id == "CartPole-v1" for subenv in env.envs)
        observations, _ = env.reset(seed=123)
        assert observations.shape == (3, 4)
    finally:
        env.close()


def test_vector_gae_preserves_independent_boundary_masks():
    rewards = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    values = rewards.clone()
    next_values = torch.tensor([[2.0, 20.0], [999.0, 30.0], [4.0, 999.0], [5.0, 50.0]])
    terminated = torch.tensor([[False, False], [True, False], [False, True], [False, False]])
    ends = torch.tensor([[False, False], [True, True], [True, True], [False, False]])
    advantages, targets, deltas = ppo.compute_gae(
        rewards, values, next_values, terminated, ends, 1.0, 1.0
    )
    expected_deltas = torch.tensor([[2.0, 20.0], [0.0, 30.0], [4.0, 0.0], [5.0, 50.0]])
    expected_advantages = torch.tensor([[2.0, 50.0], [0.0, 30.0], [4.0, 0.0], [5.0, 50.0]])
    torch.testing.assert_close(deltas, expected_deltas)
    torch.testing.assert_close(advantages, expected_advantages)
    torch.testing.assert_close(targets, expected_advantages + values)
    assert all(x.shape == (4, 2) and not x.requires_grad for x in (advantages, targets, deltas))


def test_flatten_keeps_time_environment_and_field_alignment():
    ids = torch.arange(6).reshape(3, 2)
    fields = (
        torch.stack((ids.float(), ids.float() + 100.0), dim=-1),
        ids.clone(), -(ids.float() + 1.0), ids.float() + 0.5, ids.float() + 10.0,
    )
    # Deliberately noncontiguous inputs; reshape is safer than an unchecked view.
    fields = tuple(x.transpose(0, 1).contiguous().transpose(0, 1) for x in fields)
    assert not fields[0].is_contiguous()
    flattened = ppo.flatten_vector_batch(*fields)
    assert flattened[0].shape == (6, 2)
    assert all(x.shape == (6,) for x in flattened[1:])
    for t in range(3):
        for n in range(2):
            for original, flat in zip(fields, flattened):
                torch.testing.assert_close(flat[t * 2 + n], original[t, n])


@pytest.mark.parametrize("num_samples,size,lengths", [(10, 4, [4, 4, 2]), (2, 4, [2]), (1, 1, [1])])
def test_minibatches_cover_every_flattened_sample_once(num_samples, size, lengths):
    batches = ppo.make_minibatch_indices(num_samples, size, device="cpu")
    assert [x.numel() for x in batches] == lengths
    assert all(x.ndim == 1 and x.dtype == torch.int64 for x in batches)
    torch.testing.assert_close(torch.sort(torch.cat(batches)).values, torch.arange(num_samples))


# KL diagnostics.

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_kl_is_zero_for_unchanged_policy_and_accepts_nonpositive_logs(dtype):
    old_logs = torch.tensor([0.0, -0.1, -0.7, -2.0], dtype=dtype)
    result = ppo.compute_kl_diagnostics(old_logs.clone(), old_logs)
    assert_diagnostic_dict(result)
    assert result["approx_kl"] == pytest.approx(0.0, abs=1e-12)
    assert result["signed_approx_kl"] == pytest.approx(0.0, abs=1e-12)


def test_kl_matches_a_hand_calculated_example():
    # Ratios 1/2 and 2 have opposite log ratios. Their sample KL mean is 1/4.
    old_logs = torch.log(torch.tensor([0.5, 0.25], dtype=torch.float64))
    new_logs = torch.log(torch.tensor([0.25, 0.5], dtype=torch.float64))
    result = ppo.compute_kl_diagnostics(new_logs, old_logs)
    assert_diagnostic_dict(result)
    assert result["approx_kl"] == pytest.approx(0.25, rel=1e-12, abs=1e-12)
    assert result["signed_approx_kl"] == pytest.approx(0.0, abs=1e-12)


def test_balanced_two_action_sample_matches_exact_old_to_new_kl():
    # One sample of each action under a uniform old policy gives its exact
    # action weights: 1/2 and 1/2. Mean ratio is one but KL is positive.
    old_logs = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    new_logs = torch.log(torch.tensor([0.75, 0.25], dtype=torch.float64))
    expected = -0.5 * (math.log(1.5) + math.log(0.5))
    result = ppo.compute_kl_diagnostics(new_logs, old_logs)
    assert_diagnostic_dict(result)
    assert torch.exp(new_logs - old_logs).mean().item() == pytest.approx(1.0)
    assert result["approx_kl"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert result["signed_approx_kl"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_signed_kl_can_be_negative_and_is_not_the_nonnegative_estimator():
    new_logs = torch.tensor([-0.2, -0.3], dtype=torch.float64)
    old_logs = torch.tensor([-0.8, -1.0], dtype=torch.float64)
    result = ppo.compute_kl_diagnostics(new_logs, old_logs)
    expected = ((math.expm1(0.6) - 0.6) + (math.expm1(0.7) - 0.7)) / 2.0
    assert_diagnostic_dict(result)
    assert result["signed_approx_kl"] == pytest.approx(-0.65, abs=1e-12)
    assert result["approx_kl"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert result["approx_kl"] > 0.0


@pytest.mark.parametrize("delta", [-1e-8, -1e-6])
def test_tiny_near_zero_kl_is_accepted_and_uses_stable_expm1(delta):
    old_logs = torch.zeros(1, dtype=torch.float64)
    new_logs = torch.tensor([delta], dtype=torch.float64)
    result = ppo.compute_kl_diagnostics(new_logs, old_logs)
    expected = math.expm1(delta) - delta
    assert_diagnostic_dict(result)
    # These are valid KL values below 1e-8, not invalid results to reject.
    assert 0.0 < result["approx_kl"] < 1e-8
    assert result["approx_kl"] == pytest.approx(expected, rel=2e-7, abs=1e-24)
    assert result["signed_approx_kl"] == pytest.approx(-delta, rel=1e-12)


def test_diagnostics_detach_current_policy_logs_without_changing_inputs():
    new_logs = torch.tensor([-0.4, -1.1], requires_grad=True)
    old_logs = torch.tensor([-0.7, -0.8])
    new_before, old_before = new_logs.detach().clone(), old_logs.clone()
    result = ppo.compute_kl_diagnostics(new_logs, old_logs)
    assert_diagnostic_dict(result)
    assert new_logs.requires_grad
    assert new_logs.grad is None
    torch.testing.assert_close(new_logs.detach(), new_before)
    torch.testing.assert_close(old_logs, old_before)


@pytest.mark.parametrize("case", ["matrix", "empty", "different_lengths", "old_matrix"])
def test_diagnostics_reject_invalid_shapes(case):
    new_logs = torch.full((2,), -0.5)
    old_logs = torch.full((2,), -0.7)
    if case == "matrix":
        new_logs, old_logs = new_logs.reshape(1, 2), old_logs.reshape(1, 2)
    elif case == "empty":
        new_logs, old_logs = new_logs[:0], old_logs[:0]
    elif case == "different_lengths":
        old_logs = old_logs[:1]
    else:
        old_logs = old_logs.reshape(2, 1)
    with pytest.raises(VALIDATION_ERRORS):
        ppo.compute_kl_diagnostics(new_logs, old_logs)


def test_diagnostics_reject_old_logs_that_require_grad():
    new_logs = torch.tensor([-0.4, -1.0])
    old_logs = torch.tensor([-0.7, -0.7], requires_grad=True)
    with pytest.raises(VALIDATION_ERRORS):
        ppo.compute_kl_diagnostics(new_logs, old_logs)


@pytest.mark.parametrize("which", ["new", "old"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_diagnostics_reject_nonfinite_inputs(which, bad):
    new_logs = torch.tensor([-0.5, -0.8])
    old_logs = torch.tensor([-0.7, -0.7])
    (new_logs if which == "new" else old_logs)[0] = bad
    with pytest.raises(VALIDATION_ERRORS):
        ppo.compute_kl_diagnostics(new_logs, old_logs)


def test_diagnostics_reject_nonfinite_results_from_finite_inputs():
    # Both log probabilities are finite, but the ratio exp(1000) overflows.
    new_logs = torch.tensor([-1.0], dtype=torch.float64)
    old_logs = torch.tensor([-1001.0], dtype=torch.float64)
    with pytest.raises(VALIDATION_ERRORS):
        ppo.compute_kl_diagnostics(new_logs, old_logs)


def test_diagnostics_permit_tiny_negative_roundoff_in_nonnegative_estimator(monkeypatch):
    logs = torch.full((2,), -0.7, dtype=torch.float64)
    # Inject numerical roundoff to exercise the guard deterministically.
    # Returning the tiny negative or clamping it to zero is accepted here.
    monkeypatch.setattr(torch, "expm1", lambda delta: delta - 1e-12)
    result = ppo.compute_kl_diagnostics(logs.clone(), logs)
    assert_diagnostic_dict(result)
    assert -1e-8 <= result["approx_kl"] <= 0.0
    assert result["signed_approx_kl"] == pytest.approx(0.0, abs=1e-12)


def test_diagnostics_reject_materially_negative_nonnegative_estimator(monkeypatch):
    logs = torch.full((2,), -0.7, dtype=torch.float64)
    monkeypatch.setattr(torch, "expm1", lambda delta: delta - 1e-3)
    with pytest.raises(VALIDATION_ERRORS):
        ppo.compute_kl_diagnostics(logs.clone(), logs)


# Stop decision: strict greater-than, scalar floats, and disabled stopping.

@pytest.mark.parametrize("factor", [0.0, 0.5, 1.0, 1.01])
def test_stop_decision_uses_python_floats_and_a_strict_threshold(factor):
    target, multiplier = 0.01, 1.5
    threshold = target * multiplier
    result = ppo.should_ppo_stop_for_kl(threshold * factor, target, multiplier)
    assert type(result) is bool, "Every branch must return True or False, not None."
    assert result is (factor > 1.0)


@pytest.mark.parametrize("approx_kl", [0.0, 0.4])
def test_target_none_disables_kl_stopping(approx_kl):
    assert ppo.should_ppo_stop_for_kl(approx_kl, None, 1.5) is False


def test_stop_decision_uses_the_default_multiplier():
    target = 0.01
    threshold = target * ppo.KL_STOP_MULTIPLIER
    assert ppo.should_ppo_stop_for_kl(threshold, target) is False
    assert ppo.should_ppo_stop_for_kl(math.nextafter(threshold, math.inf), target) is True


@pytest.mark.parametrize("target", [0.01, None])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_stop_decision_rejects_nonfinite_kl_even_when_stopping_is_disabled(target, bad):
    with pytest.raises(VALIDATION_ERRORS):
        ppo.should_ppo_stop_for_kl(bad, target, 1.5)


@pytest.mark.parametrize("bad_target", [0.0, -0.01, float("nan"), float("inf")])
def test_stop_decision_rejects_invalid_enabled_target(bad_target):
    with pytest.raises(VALIDATION_ERRORS):
        ppo.should_ppo_stop_for_kl(0.02, bad_target, 1.5)


@pytest.mark.parametrize("bad_multiplier", [0.0, -1.0, float("nan"), float("inf")])
def test_stop_decision_rejects_invalid_multiplier(bad_multiplier):
    with pytest.raises(VALIDATION_ERRORS):
        ppo.should_ppo_stop_for_kl(0.02, None, bad_multiplier)


# Controlled integration: isolate the epoch loop from the two scalar helpers.

@pytest.mark.parametrize("epochs", [0, -1])
def test_update_rejects_nonpositive_epoch_counts_without_optimizer_steps(epochs):
    setup = tiny_training_setup(actor_lr=0.01, critic_lr=0.02)
    with pytest.raises(VALIDATION_ERRORS):
        run_update(setup, epochs=epochs, target_kl=0.01)
    assert setup[2].step_calls == setup[3].step_calls == 0


def test_collected_rollout_logs_match_unchanged_policy_before_any_update():
    torch.manual_seed(91)
    policy = ppo.PolicyNetwork(4, 2, hidden_dim=8)
    env = ppo.make_vector_env(2)
    try:
        observations, _ = env.reset(seed=123)
        rollout, *_ = ppo.collect_vector_rollout(
            vector_env=env,
            policy=policy,
            observations=observations,
            running_episode_returns=np.zeros(2),
            running_episode_lengths=np.zeros(2, dtype=np.int64),
            steps_per_env=5,
        )
        states, actions, rewards, _, _, _, old_logs = ppo.rollout_to_tensors(rollout)
        assert states.shape == (5, 2, 4)
        assert actions.shape == rewards.shape == old_logs.shape == (5, 2)
        assert actions.dtype == torch.int64
        assert not old_logs.requires_grad
        with torch.no_grad():
            new_logs = Categorical(logits=policy(states.reshape(10, 4))).log_prob(
                actions.reshape(10)
            )
        torch.testing.assert_close(new_logs, old_logs.reshape(10), rtol=1e-6, atol=1e-6)
        result = ppo.compute_kl_diagnostics(new_logs, old_logs.reshape(10))
        assert_diagnostic_dict(result)
        assert result["approx_kl"] == pytest.approx(0.0, abs=1e-7)
        assert result["signed_approx_kl"] == pytest.approx(0.0, abs=1e-7)
    finally:
        env.close()


@pytest.mark.parametrize(
    "schedule,target,expected_epochs,expected_stopped",
    [
        ([0.03, 0.04, 0.05, 0.06], 0.01, 1, True),
        ([0.002, 0.015, 0.025, 0.001], 0.01, 3, True),
        ([0.0, 0.005, 0.01, 0.014], 0.01, 4, False),
        ([0.03, 0.04, 0.05, 0.06], None, 4, False),
        ([0.0, 0.0, 0.0, 0.03], 0.01, 4, False),
        ([0.03], 0.01, 1, False),
    ],
    ids=[
        "stop_epoch_1", "stop_epoch_3", "never_above_limit", "disabled",
        "above_limit_only_on_final_epoch", "single_epoch_high_kl",
    ],
)
def test_stopping_happens_after_complete_epochs_with_full_post_update_diagnostics(
    monkeypatch, schedule, target, expected_epochs, expected_stopped
):
    setup = tiny_training_setup(actor_lr=0.01, critic_lr=0.02)
    policy, _, actor_optimizer, critic_optimizer, rollout = setup
    full_states = torch.from_numpy(rollout["states"]).reshape(10, 4)
    full_actions = torch.from_numpy(rollout["actions"]).reshape(10)
    old_logs_before = torch.from_numpy(rollout["old_log_probs"].copy()).reshape(10)
    diagnostic_calls, stop_calls, index_calls = [], [], []

    def ordered_minibatches(num_samples, minibatch_size, device=None):
        assert num_samples == 10 and minibatch_size == 4
        index_calls.append(num_samples)
        return [torch.arange(0, 4, device=device), torch.arange(4, 8, device=device), torch.arange(8, 10, device=device)]

    def scripted_diagnostics(new_log_probs, old_log_probs):
        epoch_number = len(diagnostic_calls) + 1
        assert epoch_number <= len(schedule), "Do not evaluate diagnostics once per minibatch."
        assert new_log_probs.shape == old_log_probs.shape == (10,)
        assert not new_log_probs.requires_grad and not old_log_probs.requires_grad
        # Both optimizers must have finished ALL three minibatches this epoch.
        assert actor_optimizer.step_calls == critic_optimizer.step_calls == 3 * epoch_number
        with torch.no_grad():
            expected_logs = Categorical(logits=policy(full_states)).log_prob(full_actions)
        torch.testing.assert_close(new_log_probs, expected_logs)
        torch.testing.assert_close(old_log_probs, old_logs_before)
        diagnostic_calls.append(epoch_number)
        return {"approx_kl": schedule[epoch_number - 1], "signed_approx_kl": -0.123}

    def scripted_stop(approx_kl, target_kl, stop_multiplier):
        assert type(approx_kl) is float, "Read the diagnostic value; do not unpack dictionary keys."
        assert target_kl == target
        assert stop_multiplier == 1.5
        stop_calls.append(approx_kl)
        return target_kl is not None and approx_kl > target_kl * stop_multiplier

    monkeypatch.setattr(ppo, "make_minibatch_indices", ordered_minibatches)
    monkeypatch.setattr(ppo, "compute_kl_diagnostics", scripted_diagnostics)
    monkeypatch.setattr(ppo, "should_ppo_stop_for_kl", scripted_stop)
    metrics = run_update(setup, epochs=len(schedule), target_kl=target)

    assert len(index_calls) == len(diagnostic_calls) == len(stop_calls) == expected_epochs
    assert stop_calls == schedule[:expected_epochs]
    assert actor_optimizer.step_calls == critic_optimizer.step_calls == 3 * expected_epochs
    assert_update_metrics(
        metrics, epochs=expected_epochs, early_stopped=expected_stopped,
        approx_kl=schedule[expected_epochs - 1],
    )
    assert metrics["signed_approx_kl"] == pytest.approx(-0.123)
    np.testing.assert_array_equal(rollout["old_log_probs"].reshape(10), old_logs_before.numpy())


def test_real_short_update_reuses_fixed_targets_and_returns_current_full_rollout_kl(monkeypatch):
    # Real diagnostic and stop functions: this also catches incompatible returns.
    setup = tiny_training_setup(actor_lr=0.01, critic_lr=0.02)
    policy, value_network, actor_optimizer, critic_optimizer, rollout = setup
    actor_before = [p.detach().clone() for p in policy.parameters()]
    critic_before = [p.detach().clone() for p in value_network.parameters()]
    old_logs_before = rollout["old_log_probs"].copy()
    gae_calls, flatten_calls, critic_targets = [], [], []
    original_gae, original_flatten, original_mse = ppo.compute_gae, ppo.flatten_vector_batch, ppo.F.mse_loss

    def tracked_gae(*args, **kwargs):
        rewards = args[0] if args else kwargs["rewards"]
        gae_calls.append(rewards.shape)
        return original_gae(*args, **kwargs)

    def tracked_flatten(states, actions, logs, advantages, targets):
        flatten_calls.append(states.shape)
        assert advantages.shape == targets.shape == (5, 2)
        assert not advantages.requires_grad and not targets.requires_grad
        assert advantages.mean().item() == pytest.approx(0.0, abs=2e-6)
        assert advantages.std(unbiased=False).item() == pytest.approx(1.0, abs=2e-6)
        torch.testing.assert_close(targets, torch.from_numpy(rollout["rewards"]))
        return original_flatten(states, actions, logs, advantages, targets)

    def tracked_mse(predicted_values, targets, *args, **kwargs):
        assert predicted_values.shape == targets.shape
        assert targets.ndim == 1 and not targets.requires_grad
        critic_targets.append(targets.detach().clone())
        return original_mse(predicted_values, targets, *args, **kwargs)

    monkeypatch.setattr(ppo, "compute_gae", tracked_gae)
    monkeypatch.setattr(ppo, "flatten_vector_batch", tracked_flatten)
    monkeypatch.setattr(ppo.F, "mse_loss", tracked_mse)
    metrics = run_update(setup, epochs=3, target_kl=None)

    assert gae_calls == [torch.Size([5, 2])]
    assert flatten_calls == [torch.Size([5, 2, 4])]
    assert len(critic_targets) == 9
    for epoch in range(3):
        targets = torch.cat(critic_targets[epoch * 3 : (epoch + 1) * 3])
        torch.testing.assert_close(torch.sort(targets).values, torch.arange(1, 11, dtype=torch.float32))
    assert actor_optimizer.step_calls == critic_optimizer.step_calls == 9
    assert any(not torch.equal(old, new.detach()) for old, new in zip(actor_before, policy.parameters()))
    assert any(not torch.equal(old, new.detach()) for old, new in zip(critic_before, value_network.parameters()))
    np.testing.assert_array_equal(rollout["old_log_probs"], old_logs_before)
    with torch.no_grad():
        states = torch.from_numpy(rollout["states"]).reshape(10, 4)
        actions = torch.from_numpy(rollout["actions"]).reshape(10)
        new_logs = Categorical(logits=policy(states)).log_prob(actions)
        expected = reference_diagnostics(new_logs, torch.from_numpy(old_logs_before).reshape(10))
    assert_update_metrics(metrics, epochs=3, early_stopped=False, approx_kl=expected["approx_kl"])
    assert metrics["signed_approx_kl"] == pytest.approx(
        expected["signed_approx_kl"], rel=2e-6, abs=1e-8
    )


@pytest.mark.parametrize("target,expected_epochs,expected_stopped", [(None, 3, False), (0.01, 2, True)])
def test_last_completed_epoch_metrics_are_sample_weighted_even_after_early_stop(
    monkeypatch, target, expected_epochs, expected_stopped
):
    setup = tiny_training_setup()
    policy, _, actor_optimizer, critic_optimizer, rollout = setup
    states = torch.from_numpy(rollout["states"]).reshape(10, 4)
    actions = torch.from_numpy(rollout["actions"]).reshape(10)
    old_logs = torch.from_numpy(rollout["old_log_probs"].copy()).reshape(10)
    rewards = torch.from_numpy(rollout["rewards"]).reshape(10)
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + torch.finfo(rewards.dtype).eps)
    index_calls, diagnostic_calls = [], []

    # Epoch policies: bias 0.0, 0.4, 0.8. At bias 0.4 full-batch KL is about
    # 0.01987, exceeding 1.5 * 0.01, so enabled stopping completes two epochs.
    with torch.no_grad():
        policy.network[-1].bias[0] = 0.4 * (expected_epochs - 1)
        distribution = Categorical(logits=policy(states))
        new_logs = distribution.log_prob(actions)
        ratios = torch.exp(new_logs - old_logs)
        surrogate = torch.minimum(ratios * advantages, ratios.clamp(0.9, 1.1) * advantages)
        expected = {
            "actor_loss": (-surrogate.mean() - 0.01 * distribution.entropy().mean()).item(),
            "critic_loss": rewards.square().mean().item(),
            "mean_entropy": distribution.entropy().mean().item(),
            "mean_ratio": ratios.mean().item(),
            "min_ratio": ratios.min().item(),
            "max_ratio": ratios.max().item(),
            "clip_fraction": ((ratios - 1.0).abs() > 0.1).float().mean().item(),
        }
        expected_kl = reference_diagnostics(new_logs, old_logs)["approx_kl"]
    assert expected["critic_loss"] == pytest.approx(38.5)

    def ordered_minibatches(num_samples, minibatch_size, device=None):
        assert num_samples == 10 and minibatch_size == 4
        with torch.no_grad():
            policy.network[-1].bias[0] = 0.4 * len(index_calls)
        index_calls.append(num_samples)
        return [torch.arange(0, 4, device=device), torch.arange(4, 8, device=device), torch.arange(8, 10, device=device)]

    def full_diagnostics(new_log_probs, old_log_probs):
        assert new_log_probs.shape == old_log_probs.shape == (10,)
        assert actor_optimizer.step_calls == critic_optimizer.step_calls == 3 * (len(diagnostic_calls) + 1)
        result = reference_diagnostics(new_log_probs, old_log_probs)
        diagnostic_calls.append(result)
        return result

    def stop_decision(approx_kl, target_kl, stop_multiplier):
        assert type(approx_kl) is float, "Extract diagnostics['approx_kl'], not a dictionary key."
        assert target_kl == target and stop_multiplier == 1.5
        return target_kl is not None and approx_kl > target_kl * stop_multiplier

    monkeypatch.setattr(ppo, "make_minibatch_indices", ordered_minibatches)
    monkeypatch.setattr(ppo, "compute_kl_diagnostics", full_diagnostics)
    monkeypatch.setattr(ppo, "should_ppo_stop_for_kl", stop_decision)
    metrics = run_update(setup, epochs=3, target_kl=target)

    assert len(index_calls) == len(diagnostic_calls) == expected_epochs
    assert_update_metrics(metrics, epochs=expected_epochs, early_stopped=expected_stopped, approx_kl=expected_kl)
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value, rel=2e-6, abs=2e-6), f"Incorrect final-epoch metric: {key}"


def test_train_forwards_kl_settings_and_carries_vector_episode_state(monkeypatch, capsys):
    class ResetOnlyVectorEnv:
        num_envs = 2

        def __init__(self):
            self.reset_seeds = []

        def reset(self, seed=None):
            self.reset_seeds.append(seed)
            return np.zeros((2, 4), dtype=np.float32), {}

    env = ResetOnlyVectorEnv()
    collector_inputs, update_inputs = [], []
    outputs = [
        ({"number": 1}, np.ones((2, 4), dtype=np.float32), np.asarray([0.5, 2.0]), np.asarray([1, 1]), [2.0, 6.0], [2, 3]),
        ({"number": 2}, np.full((2, 4), 2.0, dtype=np.float32), np.zeros(2), np.zeros(2, dtype=np.int64), [4.0], [4]),
    ]

    def fake_collect(**kwargs):
        assert kwargs["vector_env"] is env
        collector_inputs.append({
            "observations": kwargs["observations"].copy(),
            "returns": kwargs["running_episode_returns"].copy(),
            "lengths": kwargs["running_episode_lengths"].copy(),
            "steps": kwargs["steps_per_env"],
        })
        return outputs[len(collector_inputs) - 1]

    def fake_update(**kwargs):
        assert kwargs["target_kl"] == 0.007
        assert kwargs["stop_multiplier"] == 2.3
        assert kwargs["update_epochs"] == 3
        assert kwargs["minibatch_size"] == 4
        update_inputs.append(kwargs["rollout"])
        return {
            "update_epoch": 1, "actor_loss": 0.0, "critic_loss": 0.0,
            "mean_entropy": 0.0, "mean_ratio": 1.0, "min_ratio": 1.0,
            "max_ratio": 1.0, "clip_fraction": 0.0, "optimizer_steps": 1,
            "approx_kl": 0.001, "signed_approx_kl": 0.0,
            # Ending one update early must not stop outer training.
            "early_stopped": len(update_inputs) == 1,
        }

    monkeypatch.setattr(ppo, "collect_vector_rollout", fake_collect)
    monkeypatch.setattr(ppo, "update_ppo_from_rollout", fake_update)
    monkeypatch.setattr(ppo, "REPORT_EVERY_UPDATES", 1)
    returns, lengths = ppo.train(
        vector_env=env, policy=object(), value_network=object(),
        actor_optimizer=object(), critic_optimizer=object(),
        seed=11, num_updates=2, steps_per_env=3,
        target_kl=0.007, stop_multiplier=2.3,
        update_epochs=3, minibatch_size=4,
    )

    assert env.reset_seeds == [11]
    assert len(collector_inputs) == len(update_inputs) == 2
    assert all(x["steps"] == 3 for x in collector_inputs)
    np.testing.assert_array_equal(collector_inputs[0]["observations"], np.zeros((2, 4)))
    np.testing.assert_array_equal(collector_inputs[0]["returns"], [0.0, 0.0])
    np.testing.assert_array_equal(collector_inputs[0]["lengths"], [0, 0])
    np.testing.assert_array_equal(collector_inputs[1]["observations"], outputs[0][1])
    np.testing.assert_array_equal(collector_inputs[1]["returns"], [0.5, 2.0])
    np.testing.assert_array_equal(collector_inputs[1]["lengths"], [1, 1])
    assert update_inputs == [{"number": 1}, {"number": 2}]
    assert returns == [2.0, 6.0, 4.0]
    assert lengths == [2, 3, 4]
    output = capsys.readouterr().out
    assert re.findall(r"Environment steps:\s*(\d+)", output) == ["6", "12"]
    assert re.findall(r"signed_approx_kl:\s*(\S+)", output) == ["0.0", "0.0"]
    assert re.findall(r"early_stopped:\s*(True|False)", output) == ["True", "False"]
