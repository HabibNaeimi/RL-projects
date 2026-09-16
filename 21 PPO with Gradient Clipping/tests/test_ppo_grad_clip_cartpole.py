
import math

import pytest
import torch
from torch.distributions import Categorical

import ppo_grad_clip_cartpole as ppo


def gradient_norm(parameters):
    """Independent test oracle for the global L2 gradient norm."""
    squared_sum = sum(
        parameter.grad.detach().square().sum().item()
        for parameter in parameters
        if parameter.grad is not None
    )
    return math.sqrt(squared_sum)


def test_get_gradient_norm_uses_one_global_l2_norm_and_ignores_none():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))
    unused = torch.nn.Parameter(torch.zeros(3))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0])
    assert unused.grad is None

    # Must also work when parameters is a one-use generator.
    result = ppo.get_gradient_norm(x for x in (first, second, unused))
    assert float(result) == pytest.approx(13.0)


def test_helpers_handle_parameters_with_no_gradients():
    parameters = list(torch.nn.Linear(3, 2).parameters())
    before = [parameter.detach().clone() for parameter in parameters]

    assert float(ppo.get_gradient_norm(iter(parameters))) == pytest.approx(0.0)
    assert float(ppo.clip_gradients(iter(parameters), max_grad_norm=1.0)) == pytest.approx(0.0)
    assert all(parameter.grad is None for parameter in parameters)
    for original, parameter in zip(before, parameters):
        torch.testing.assert_close(parameter, original)


def test_clip_gradients_returns_pre_clip_norm_and_scales_globally():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0])
    parameters = [first, second]

    pre_clip_norm = ppo.clip_gradients(iter(parameters), max_grad_norm=5.0)

    assert float(pre_clip_norm) == pytest.approx(13.0)
    assert gradient_norm(parameters) <= 5.0 + 1e-5
    torch.testing.assert_close(first.grad, torch.tensor([15 / 13, 20 / 13]), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(second.grad, torch.tensor([60 / 13]), rtol=2e-5, atol=2e-5)


def test_update_clips_before_steps_and_reports_last_epoch_statistics(monkeypatch):
    torch.manual_seed(7)
    policy = ppo.PolicyNetwork(4, 2, hidden_dim=8)
    value_network = ppo.ValueNetwork(4, hidden_dim=8)
    with torch.no_grad():
        for parameter in (*policy.parameters(), *value_network.parameters()):
            parameter.zero_()

    states = torch.zeros((2, 2, 4))
    actions = torch.tensor([[0, 0], [0, 1]])
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with torch.no_grad():
        old_log_probs = Categorical(logits=policy(states)).log_prob(actions)
    rollout = {
        "states": states.numpy(),
        "actions": actions.numpy(),
        "rewards": rewards.numpy(),
        "next_states": states.numpy().copy(),
        "terminated": torch.zeros((2, 2), dtype=torch.bool).numpy(),
        "episode_ends": torch.zeros((2, 2), dtype=torch.bool).numpy(),
        "old_log_probs": old_log_probs.numpy(),
    }

    max_norm = 0.45
    events = []
    recorded = {"actor": [], "critic": []}
    actor_ids = {id(parameter) for parameter in policy.parameters()}
    critic_ids = {id(parameter) for parameter in value_network.parameters()}
    original_clip = ppo.clip_gradients

    class CheckingSGD(torch.optim.SGD):
        def __init__(self, parameters, label):
            self.tracked_parameters = list(parameters)
            self.label = label
            super().__init__(self.tracked_parameters, lr=0.05)

        def step(self, closure=None):
            events.append(f"{self.label}_step")
            assert gradient_norm(self.tracked_parameters) <= max_norm + 1e-5
            return super().step(closure=closure)

    actor_optimizer = CheckingSGD(policy.parameters(), "actor")
    critic_optimizer = CheckingSGD(value_network.parameters(), "critic")

    def tracked_clip(parameters, max_grad_norm):
        parameters = list(parameters)
        ids = {id(parameter) for parameter in parameters}
        label = "actor" if ids == actor_ids else "critic"
        assert ids in (actor_ids, critic_ids)
        before = gradient_norm(parameters)
        assert before > 0.0  # proves clipping is after backward()
        recorded[label].append(before)
        events.append(f"{label}_clip")
        returned = original_clip(parameters, max_grad_norm)
        assert float(returned) == pytest.approx(before, rel=1e-5)
        return returned

    monkeypatch.setattr(ppo, "clip_gradients", tracked_clip)
    monkeypatch.setattr(
        ppo,
        "make_minibatch_indices",
        lambda num_samples, minibatch_size, device=None: [
            torch.arange(0, 2, device=device), torch.arange(2, 4, device=device)
        ],
    )
    # KL behavior was tested in Level 20; isolate it here.
    monkeypatch.setattr(
        ppo,
        "compute_kl_diagnostics",
        lambda new_log_probs, old_log_probs: {"approx_kl": 0.0, "signed_approx_kl": 0.0},
    )
    monkeypatch.setattr(ppo, "should_ppo_stop_for_kl", lambda *args, **kwargs: False)

    metrics = ppo.update_ppo_from_rollout(
        policy=policy,
        value_network=value_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        rollout=rollout,
        gamma=0.0,
        gae_lambda=0.95,
        entropy_coef=0.0,
        clip_eps=0.2,
        update_epochs=2,
        minibatch_size=2,
        target_kl=None,
        stop_multiplier=1.5,
        max_grad_norm=max_norm,
    )

    assert events == ["actor_clip", "actor_step", "critic_clip", "critic_step"] * 4
    for label in ("actor", "critic"):
        # Two minibatches from only the final completed epoch.
        final_norms = recorded[label][-2:]
        expected_mean = sum(final_norms) / len(final_norms)
        expected_fraction = sum(norm > max_norm for norm in final_norms) / len(final_norms)
        assert metrics[f"mean_{label}_grad_norm"] == pytest.approx(expected_mean)
        assert metrics[f"{label}_grad_clip_fraction"] == pytest.approx(expected_fraction)
        assert 0.0 <= metrics[f"{label}_grad_clip_fraction"] <= 1.0
