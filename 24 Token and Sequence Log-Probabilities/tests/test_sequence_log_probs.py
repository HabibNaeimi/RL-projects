import pytest
import torch
from torch.distributions import Categorical

from sequence_log_probs import (
    BOS_ID,
    EOS_ID,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    TinyAutoregressivePolicy,
    compute_response_log_probs,
    compute_sequence_log_prob,
    encode_tokens,
    generate_trajectory,
)


def make_policy():
    torch.manual_seed(123)

    return TinyAutoregressivePolicy(
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )


def slow_prefix_log_probs(
    policy,
    prompt_ids,
    response_ids,
    temperature,
):
    """Reference implementation using one forward pass per token."""
    prefix = prompt_ids.unsqueeze(0)
    log_probs = []

    for token_id in response_ids:
        logits = policy(prefix)
        next_token_logits = logits[:, -1, :]

        distribution = Categorical(
            logits=next_token_logits / temperature
        )

        token_log_prob = distribution.log_prob(
            token_id.reshape(1)
        )

        log_probs.append(token_log_prob.squeeze(0))

        prefix = torch.cat(
            [prefix, token_id.reshape(1, 1)],
            dim=1,
        )

    return torch.stack(log_probs)


def test_vectorized_scores_match_prefix_loop():
    policy = make_policy()

    prompt_ids = encode_tokens(
        ["<BOS>", "2", "+", "3", "="]
    )
    response_ids = encode_tokens(
        ["4", "+", "1", "<EOS>"]
    )

    temperature = 0.7

    actual = compute_response_log_probs(
        policy,
        prompt_ids,
        response_ids,
        temperature,
    )

    expected = slow_prefix_log_probs(
        policy,
        prompt_ids,
        response_ids,
        temperature,
    )

    assert actual.shape == response_ids.shape
    assert torch.allclose(actual, expected, atol=1e-6)


def test_sequence_log_prob_shape_and_gradient():
    policy = make_policy()

    prompt_ids = encode_tokens(["<BOS>", "2", "+"])
    response_ids = encode_tokens(["3", "=", "5", "<EOS>"])

    token_log_probs = compute_response_log_probs(
        policy,
        prompt_ids,
        response_ids,
    )
    sequence_log_prob = compute_sequence_log_prob(
        token_log_probs
    )

    assert token_log_probs.shape == (4,)
    assert sequence_log_prob.shape == ()
    assert sequence_log_prob.requires_grad

    assert torch.allclose(
        sequence_log_prob,
        token_log_probs.sum(),
    )

    policy.zero_grad()
    sequence_log_prob.backward()

    gradient = policy.output_head.weight.grad

    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_generated_and_recomputed_log_probs_match():
    policy = make_policy()
    prompt_ids = encode_tokens(
        ["<BOS>", "2", "+", "3", "="]
    )
    temperature = 0.8

    trajectory = generate_trajectory(
        policy=policy,
        prompt_ids=prompt_ids,
        eos_token_id=EOS_ID,
        max_new_tokens=5,
        deterministic=False,
        temperature=temperature,
    )

    recomputed = compute_response_log_probs(
        policy=policy,
        prompt_ids=trajectory["prompt_ids"],
        response_ids=trajectory["response_ids"],
        temperature=temperature,
    )

    assert not trajectory["old_log_probs"].requires_grad
    assert recomputed.requires_grad

    assert torch.allclose(
        recomputed,
        trajectory["old_log_probs"],
        atol=1e-6,
    )

    ratios = torch.exp(
        recomputed - trajectory["old_log_probs"]
    )

    assert torch.allclose(
        ratios,
        torch.ones_like(ratios),
        atol=1e-6,
    )


def test_one_token_eos_response_is_scored():
    policy = make_policy()

    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()

        policy.output_head.bias[EOS_ID] = 10.0

    prompt_ids = torch.tensor(
        [BOS_ID],
        dtype=torch.long,
    )

    trajectory = generate_trajectory(
        policy=policy,
        prompt_ids=prompt_ids,
        eos_token_id=EOS_ID,
        max_new_tokens=4,
        deterministic=True,
        temperature=1.0,
    )

    assert trajectory["response_ids"].shape == (1,)
    assert trajectory["response_ids"][0].item() == EOS_ID
    assert trajectory["terminated"] is True
    assert trajectory["truncated"] is False

    recomputed = compute_response_log_probs(
        policy,
        trajectory["prompt_ids"],
        trajectory["response_ids"],
        temperature=1.0,
    )

    assert recomputed.shape == (1,)
    assert torch.allclose(
        recomputed,
        trajectory["old_log_probs"],
        atol=1e-6,
    )


def test_empty_response_is_rejected():
    policy = make_policy()
    prompt_ids = encode_tokens(["<BOS>", "2"])

    with pytest.raises(ValueError):
        compute_response_log_probs(
            policy,
            prompt_ids,
            torch.empty(0, dtype=torch.long),
        )


def test_scoring_is_deterministic():
    policy = make_policy()
    prompt_ids = encode_tokens(["<BOS>", "2", "+"])
    response_ids = encode_tokens(["3", "=", "5"])

    first = compute_response_log_probs(
        policy, prompt_ids, response_ids
    )
    second = compute_response_log_probs(
        policy, prompt_ids, response_ids
    )

    assert torch.allclose(first, second)