import torch

from tiny_language_policy import (
    VOCAB,
    TOKEN_TO_ID,
    BOS_ID,
    EOS_ID,
    TinyAutoregressivePolicy,
    encode_tokens,
    generate_trajectory,
)


def make_policy():
    return TinyAutoregressivePolicy(
        vocab_size=len(VOCAB),
        embedding_size=8,
        hidden_dim=16,
    )


def force_output_token(policy, token_id):
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()

        policy.output_head.bias.fill_(-100.0)
        policy.output_head.bias[token_id] = 100.0


def test_forward_returns_one_distribution_per_position():
    policy = make_policy()

    input_ids = torch.tensor([
        [BOS_ID, TOKEN_TO_ID["2"], TOKEN_TO_ID["+"]],
        [BOS_ID, TOKEN_TO_ID["3"], TOKEN_TO_ID["+"]],
    ])

    logits = policy(input_ids)

    assert logits.shape == (2, 3, len(VOCAB))
    assert logits.requires_grad
    assert torch.isfinite(logits).all()


def test_eos_is_a_stored_terminating_action():
    policy = make_policy()
    force_output_token(policy, EOS_ID)

    prompt_ids = encode_tokens(["<BOS>", "2", "="])

    trajectory = generate_trajectory(
        policy=policy,
        prompt_ids=prompt_ids,
        eos_token_id=EOS_ID,
        max_new_tokens=4,
        deterministic=True,
    )

    assert trajectory["response_ids"].tolist() == [EOS_ID]
    assert trajectory["old_log_probs"].shape == (1,)
    assert len(trajectory["prefixes"]) == 1
    assert trajectory["terminated"] is True
    assert trajectory["truncated"] is False


def test_generation_limit_produces_truncation_and_aligned_prefixes():
    policy = make_policy()
    forced_token = TOKEN_TO_ID["1"]
    force_output_token(policy, forced_token)

    prompt_ids = encode_tokens(["<BOS>", "2", "="])

    trajectory = generate_trajectory(
        policy=policy,
        prompt_ids=prompt_ids,
        eos_token_id=EOS_ID,
        max_new_tokens=3,
        deterministic=True,
    )

    assert trajectory["response_ids"].tolist() == [
        forced_token,
        forced_token,
        forced_token,
    ]
    assert trajectory["old_log_probs"].shape == (3,)
    assert not trajectory["old_log_probs"].requires_grad
    assert torch.isfinite(trajectory["old_log_probs"]).all()

    for step, prefix in enumerate(trajectory["prefixes"]):
        expected = torch.cat([
            prompt_ids,
            trajectory["response_ids"][:step],
        ])
        assert torch.equal(prefix, expected)

    assert trajectory["terminated"] is False
    assert trajectory["truncated"] is True