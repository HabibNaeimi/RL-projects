import math
import torch
from torch import nn
from torch.distributions import Categorical

SEED = 32268


VOCAB = [
    "<BOS>",
    "<EOS>",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "+",
    "=",
]

TOKEN_TO_ID = {
    token: token_id
    for token_id, token in enumerate(VOCAB)
}

ID_TO_TOKEN = {
    token_id: token
    for token, token_id in TOKEN_TO_ID.items()
}

BOS_ID = TOKEN_TO_ID["<BOS>"]
EOS_ID = TOKEN_TO_ID["<EOS>"]

VOCAB_SIZE = len(VOCAB)
EMBEDDING_DIM = 16
HIDDEN_DIM = 32
MAX_NEW_TOKENS = 6

TEMPERATURE = 1.0


class TinyAutoregressivePolicy(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_size,
            hidden_dim,
    ):
        """
        The layers should initialize here, 
            then configure and pass to each other 
            in forward() method.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
        )
        self.gru = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.output_head = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.vocab_size
        )

    def forward(self, input_ids):
        """
        input_ids: [batch_size, sequence_length], dtype=torch.long

        returns:
            logits: [batch_size, sequence_length, vocab_size]
        """
        assert input_ids.ndim == 2
        assert input_ids.dtype == torch.long

        # Embedding: [B, L] -> [B, L, embedding_size]
        embedded = self.embedding(input_ids)

        # gru_output: [B, L, hidden_dim] : all position outputs
        # final_hidden (_ in here): [1, B, hidden_dim] : the final recurrent hidden state
        gru_output, _ = self.gru(embedded)

        # [B, L, hidden_dim] -> [B, L, vocab_size] : Converts each hidden state into vocabulary logits
        logits = self.output_head(gru_output)
        assert logits.shape == (
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
        )
        return logits



def encode_tokens(tokens):
    token_ids = [TOKEN_TO_ID[token] for token in tokens]
    long_tensor = torch.tensor(
        token_ids, 
        dtype=torch.long,
    )
    return long_tensor

def decode_ids(token_ids):
    raw_tokens = [ID_TO_TOKEN[token_id.item()] for token_id in token_ids]
    final_text = " ".join(raw_tokens)
    return final_text


@torch.no_grad()
def generate_trajectory(
    policy,
    prompt_ids,
    eos_token_id,
    max_new_tokens,
    deterministic=False,
    temperature=TEMPERATURE
):
    """
    prompt_ids: [prompt_length]

    Returns a dictionary containing:
        prompt_ids:     [P]
        response_ids:   [R]
        old_log_probs:  [R]
        prefixes:       list of R variable-length tensors
        terminated:     Python bool
        truncated:      Python bool
    """
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be one-dimensional.")
    if prompt_ids.numel() == 0:
        raise ValueError("The prompt must not be empty.")
    
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite.")
    
    prefix = prompt_ids.unsqueeze(0)          # shape: [1, P]
    response_ids = []
    old_log_probs = []
    prefixes = []
    terminated = False

    for _ in range(max_new_tokens):
        logits = policy(prefix)               # shape: [1, current_length, vocab_size]
        next_token_logits = logits[:, -1, :]  # shape: [1, vocab_size]
        distribution = Categorical(logits=next_token_logits / temperature)

        if deterministic:
            next_token = torch.argmax(next_token_logits, dim=1)
        else:
            next_token = distribution.sample()

        log_prob = distribution.log_prob(next_token)

        prefixes.append(prefix.squeeze(0).clone())
        response_ids.append(next_token.item())
        old_log_probs.append(float(log_prob.item()))

        prefix = torch.cat(
            [prefix, next_token.unsqueeze(1)], dim=1,
        )
        if next_token.item() == eos_token_id:
            terminated = True
            break

    truncated = not terminated
    response_ids_tensor = torch.tensor(response_ids, dtype=torch.long)
    old_log_prob_tensor = torch.tensor(old_log_probs, dtype=torch.float32)

    return {
        "prompt_ids": prompt_ids.clone(),
        "response_ids": response_ids_tensor,
        "old_log_probs": old_log_prob_tensor,
        "prefixes": prefixes,
        "terminated": terminated,
        "truncated": truncated,
    }



def compute_response_log_probs(
    policy,
    prompt_ids,
    response_ids,
    temperature=TEMPERATURE,
):
    """
    Args:
        prompt_ids:   [P], torch.long
        response_ids: [R], torch.long

    Returns:
        token_log_probs: [R], gradient-connected
    """
    # Prompt and response validation
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be one-dimensional.")

    if response_ids.ndim != 1:
        raise ValueError("response_ids must be one-dimensional.")

    if prompt_ids.dtype != torch.long:
        raise TypeError("prompt_ids must have dtype torch.long.")

    if response_ids.dtype != torch.long:
        raise TypeError("response_ids must have dtype torch.long.")

    if prompt_ids.numel() == 0:
        raise ValueError("The prompt must not be empty.")

    if response_ids.numel() == 0:
        raise ValueError("The response must not be empty.")

    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite.")
    
    prompt_length = prompt_ids.shape[0]
    response_length = response_ids.shape[0]

    model_input_ids = torch.cat([prompt_ids, response_ids[:-1]], dim=0)
    assert model_input_ids.shape == (
        prompt_length + response_length - 1,
    )

    batched_inputs = model_input_ids.unsqueeze(0)
    all_logits = policy(batched_inputs)
    assert all_logits.shape == (
        1,
        prompt_length + response_length - 1,
        policy.vocab_size,
    )

    response_logits = all_logits[:, prompt_length - 1:, :].squeeze(0)
    assert response_logits.shape == (
        response_length,
        policy.vocab_size,
    )

    distribution = Categorical(logits=response_logits / temperature)
    token_log_probs = distribution.log_prob(response_ids)

    assert token_log_probs.shape == (response_length,)
    assert token_log_probs.requires_grad
    assert torch.isfinite(token_log_probs).all()

    return token_log_probs


def compute_sequence_log_prob(token_log_probs):
    """
    Args:
        token_log_probs: [R]

    Returns:
        sequence_log_prob: scalar tensor
    """
    if token_log_probs.ndim != 1:
        raise ValueError("token_log_probs must be one-dimensional.")

    if token_log_probs.numel() == 0:
        raise ValueError("token_log_probs must not be empty.")
    
    sequence_log_prob = token_log_probs.sum()

    assert sequence_log_prob.shape == ()
    assert sequence_log_prob.requires_grad

    return sequence_log_prob


def main():
    torch.manual_seed(SEED)

    policy = TinyAutoregressivePolicy(
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )

    prompt_ids = encode_tokens(
        ["<BOS>", "1", "+", "2", "="]
    )
    trajectory = generate_trajectory(
        policy=policy,
        prompt_ids=prompt_ids,
        eos_token_id=EOS_ID,
        max_new_tokens=MAX_NEW_TOKENS,
        deterministic=False,
        temperature=TEMPERATURE,
    )

    recomputed_log_probs = compute_response_log_probs(
        policy=policy,
        prompt_ids=trajectory["prompt_ids"],
        response_ids=trajectory["response_ids"],
        temperature=TEMPERATURE,
    )

    sequence_log_prob = compute_sequence_log_prob(
        token_log_probs=recomputed_log_probs
    )

    ratios = torch.exp(recomputed_log_probs - trajectory["old_log_probs"])

    # Safety assertions
    R = trajectory["response_ids"].shape[0]

    assert trajectory["old_log_probs"].shape == (R,)
    assert recomputed_log_probs.shape == (R,)
    assert sequence_log_prob.shape == ()

    assert not trajectory["old_log_probs"].requires_grad
    assert recomputed_log_probs.requires_grad
    assert sequence_log_prob.requires_grad

    assert torch.allclose(
        recomputed_log_probs,
        trajectory["old_log_probs"],
        atol=1e-6,
    )

    assert torch.allclose(
        ratios,
        torch.ones_like(ratios),
        atol=1e-6,
    )


    print("Prompt:", decode_ids(trajectory["prompt_ids"]))
    print("Response:", decode_ids(trajectory["response_ids"]))
    print("Old log-probabilities:", trajectory["old_log_probs"])
    print("Terminated by EOS:", trajectory["terminated"])
    print("Truncated by limit:", trajectory["truncated"])
    print("Per-token probability Ratios:", ratios)


if __name__ == "__main__":
    main()
