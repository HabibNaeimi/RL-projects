# Level 23 — Tiny Autoregressive Language Policy

Level 23 is the transition from reinforcement learning in environments such as CartPole to reinforcement learning with language models.

Instead of choosing a physical action from an environment state, the policy chooses one token from a vocabulary. The prompt and previously generated tokens form the current state, each generated token is an action, and the complete response is a trajectory.

This level builds the mechanics required for later LLM reinforcement learning:

- token encoding and decoding;
- an `Embedding → GRU → Linear` policy;
- autoregressive token generation;
- stochastic and deterministic action selection;
- end-of-sequence termination;
- maximum-length truncation;
- storage of action log-probabilities; and
- exact alignment between prefixes, actions, and log-probabilities.

There is deliberately no optimizer, task reward, verifier, policy update, or plotting in this level.

## Project files

```text
23 Tiny Autoregressive Policy/
├── tiny_language_policy.py
└── README.md
```

A test file can be added later as:

```text
tests/
└── test_tiny_language_policy.py
```

## Installation

The implementation only requires PyTorch:

```bash
python -m pip install torch pytest
```

`matplotlib` is not used in this level.

## Configuration

| Setting | Value |
|---|---:|
| Random seed | 32,268 |
| Vocabulary size | 10 |
| Embedding dimension | 16 |
| GRU hidden dimension | 32 |
| Maximum new tokens | 6 |
| Default temperature | 0.1 |
| Prompt | `<BOS> 2 + 3 =` |
| Generation mode in `main` | Stochastic |

## From environmental RL to language-model RL

The core RL concepts have not disappeared. Their representation has changed.

| Environmental RL | Autoregressive language policy |
|---|---|
| State | Prompt plus generated tokens |
| Action | Next token |
| Policy | Distribution over the vocabulary |
| Transition | Append the selected token to the prefix |
| Trajectory | Complete generated response |
| Natural termination | Generate `<EOS>` |
| Time-limit truncation | Reach `max_new_tokens` without `<EOS>` |
| Stored action log-probability | Log-probability of each generated token |
| Episode return | Not introduced in this level |

At generation step $t$, define the state as:

$$
S_t = (x_1,\ldots,x_P,y_1,\ldots,y_{t-1}),
$$

where $x_1,\ldots,x_P$ are the prompt tokens and $y_1,\ldots,y_{t-1}$ are the response tokens already generated.

The action is the next token:

$$
A_t = y_t.
$$

The transition is deterministic once the action has been chosen:

$$
S_{t+1} = (S_t,A_t).
$$

The uncertainty comes from the policy's token selection, not from the prefix update.

## Vocabulary and token IDs

The toy vocabulary is intentionally small:

| Token | ID |
|---|---:|
| `<BOS>` | 0 |
| `<EOS>` | 1 |
| `0` | 2 |
| `1` | 3 |
| `2` | 4 |
| `3` | 5 |
| `4` | 6 |
| `5` | 7 |
| `+` | 8 |
| `=` | 9 |

`TOKEN_TO_ID` converts token strings to integer IDs, while `ID_TO_TOKEN` performs the inverse mapping.

For the demonstration prompt:

```text
<BOS> 2 + 3 =
```

the encoded tensor is:

```text
[0, 4, 8, 5, 9]
```

The implementation is a deliberately minimal tokenizer. It has no unknown token, subword splitting, padding, or batch tokenization. Passing a token outside `VOCAB` raises a key lookup error.

## Policy architecture

`TinyAutoregressivePolicy` contains three trainable layers:

```python
self.embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_size,
)

self.gru = nn.GRU(
    input_size=embedding_size,
    hidden_size=hidden_dim,
    batch_first=True,
)

self.output_head = nn.Linear(
    in_features=hidden_dim,
    out_features=vocab_size,
)
```

### Tensor shapes

Let:

- $B$ be the batch size;
- $L$ be the current sequence length;
- $E=16$ be the embedding dimension;
- $H=32$ be the GRU hidden dimension; and
- $V=10$ be the vocabulary size.

The forward pass has these shapes:

| Stage | Input shape | Output shape |
|---|---|---|
| Token IDs | — | `[B, L]` |
| Embedding | `[B, L]` | `[B, L, E]` |
| GRU outputs | `[B, L, E]` | `[B, L, H]` |
| Output head | `[B, L, H]` | `[B, L, V]` |

The embedding layer is a learnable lookup table: each token ID selects one row from a matrix with shape $V\times E$. See the official [PyTorch `Embedding` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html).

With `batch_first=True`, the GRU accepts and returns tensors whose first two dimensions are batch and sequence. It produces:

- `gru_output` with shape `[B, L, H]`, containing an output for every position; and
- `final_hidden` with shape `[1, B, H]` for this one-layer, one-direction GRU.

This implementation uses `gru_output` because a vocabulary prediction is needed at every sequence position. The final hidden tensor is returned by the GRU but is not otherwise needed. See the official [PyTorch `GRU` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html).

### Number of trainable parameters

The embedding contains:

$$
N_E = VE = 10\times16 = 160
$$

parameters.

A one-layer GRU contains three input projections, three recurrent projections, and two bias vectors for each of its three gates:

$$
N_G = 3HE + 3H^2 + 6H.
$$

For $E=16$ and $H=32$:

$$
N_G
= 3\times32\times16
+ 3\times32^2
+ 6\times32
= 4800.
$$

The output head contains:

$$
N_O = HV + V = 32\times10+10 = 330.
$$

Therefore, the complete policy has:

$$
N = N_E+N_G+N_O = 160+4800+330 = 5290
$$

trainable parameters.

## Autoregressive next-token policy

For a prefix $S_t$, the network produces one logit $z_{t,k}$ for every vocabulary token $k$.

The code only uses the logits at the final prefix position:

```python
logits = policy(prefix)
next_token_logits = logits[:, -1, :]
```

This final position summarizes the entire prefix through the GRU hidden state.

With temperature $\tau>0$, the categorical probability of token $k$ is:

$$
\pi_\theta(k\mid S_t;\tau)
= \frac{
\exp(z_{t,k}/\tau)
}{
\sum_{j=0}^{V-1}\exp(z_{t,j}/\tau)
}.
$$

The implementation constructs this policy directly from scaled logits:

```python
distribution = Categorical(
    logits=next_token_logits / temperature
)
```

PyTorch's `Categorical` distribution accepts logits, samples integer category indices, and computes their log-probabilities. See the official [categorical-distribution documentation](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical).

### Effect of temperature

- $\tau=1$ preserves the original relative logits.
- $0<\tau<1$ makes the distribution sharper.
- $\tau>1$ makes the distribution flatter.

This experiment uses $\tau=0.1$, so even modest logit differences become strong probability differences.

The code rejects zero, negative, infinite, and non-numeric temperatures because they do not define a valid scaling value.

## Autoregressive factorization

For prompt $x$ and a response containing $R$ generated tokens, the response probability factorizes as:

$$
\pi_\theta(y_1,\ldots,y_R\mid x)
= \prod_{t=1}^{R}
\pi_\theta(y_t\mid x,y_1,\ldots,y_{t-1}).
$$

Its log-probability is:

$$
\log\pi_\theta(y_1,\ldots,y_R\mid x)
= \sum_{t=1}^{R}
\log\pi_\theta(y_t\mid x,y_1,\ldots,y_{t-1}).
$$

This is why token log-probabilities are recorded separately: the probability of a response is built from a sequence of conditional action probabilities.

## Trajectory generation

`generate_trajectory` starts from the prompt and repeats the following process:

1. add a batch dimension to obtain a prefix with shape `[1, P]`;
2. run the complete current prefix through the policy;
3. select the final-position logits;
4. create a temperature-scaled categorical policy;
5. sample a token or choose the largest-logit token;
6. compute the selected token's log-probability;
7. save the pre-action prefix, token ID, and log-probability;
8. append the token to the prefix; and
9. stop if the token is `<EOS>`.

If `<EOS>` is not selected, the loop continues until `max_new_tokens` actions have been generated.

### Stochastic selection

With `deterministic=False`:

```python
next_token = distribution.sample()
```

The action is sampled from the categorical policy:

$$
A_t \sim \pi_\theta(\cdot\mid S_t;\tau).
$$

This is the appropriate behavior for collecting diverse trajectories during RL training.

### Deterministic selection

With `deterministic=True`:

```python
next_token = torch.argmax(next_token_logits, dim=1)
```

The action is the token with the largest logit. Dividing logits by a positive temperature does not change which token has the largest value, although temperature still changes the stored log-probability of that token.

## Correct prefix-action alignment

The prefix must be stored before the newly chosen token is appended:

```python
prefixes.append(prefix.squeeze(0).clone())
response_ids.append(next_token.item())
old_log_probs.append(float(log_prob.item()))
```

For generated response $y_1,y_2,\ldots,y_R$, the alignment is:

| Stored index | Prefix used as state | Stored action |
|---:|---|---|
| 0 | Prompt only | $y_1$ |
| 1 | Prompt plus $y_1$ | $y_2$ |
| 2 | Prompt plus $y_1,y_2$ | $y_3$ |
| $\ldots$ | $\ldots$ | $\ldots$ |
| $R-1$ | Prompt plus $y_1,\ldots,y_{R-1}$ | $y_R$ |

Let $n_P$, $n_A$, and $n_L$ denote the numbers of stored prefixes, response IDs, and old log-probabilities. Then:

$$
n_P=n_A=n_L=R.
$$

The length of stored prefix $t$ is:

$$
P+t-1,
\qquad 1\leq t\leq R.
$$

`clone()` is important because each list element must preserve the prefix as it existed at that step.

## Stored old log-probabilities

For selected token $A_t$, the stored value is:

$$
\ell_t^0
= \log\pi_{\theta_0}(A_t\mid S_t;\tau),
$$

where $\theta_0$ denotes the policy parameters used to generate the trajectory.

These values are called old log-probabilities because a later policy update may change the parameters from $\theta_0$ to $\theta$. They provide the fixed behavior-policy reference needed to compare the updated policy with the policy that produced the data.

For example, a future importance ratio can be computed as:

$$
r_t(\theta)
= \exp\left(
\ell_t(\theta)-\ell_t^0
\right).
$$

Level 23 only stores $\ell_t^0$. It does not perform that update.

## Why generation uses no gradients

`generate_trajectory` is decorated with:

```python
@torch.no_grad()
```

Trajectory collection does not call `backward()`, so retaining its computation graph would waste memory. PyTorch documents `no_grad` as a context that disables gradient calculation for inference-like computation. See the official [`torch.no_grad` documentation](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html).

This does not prevent future learning. During an update, the stored prefixes and actions can be passed through the current policy again without `no_grad` to compute new differentiable log-probabilities.

## Termination and truncation

The two ending conditions carry different meanings.

### Natural termination

If the generated token equals `EOS_ID`:

```python
if next_token.item() == eos_token_id:
    terminated = True
    break
```

The policy chose to finish the response.

### Length truncation

After the loop:

```python
truncated = not terminated
```

If no `<EOS>` token appeared within `max_new_tokens`, the response ended because of the external length limit.

In this implementation:

| Condition | `terminated` | `truncated` |
|---|---:|---:|
| `<EOS>` generated | `True` | `False` |
| Token limit reached first | `False` | `True` |

This distinction is the language-generation equivalent of separating natural episode termination from a time-limit cutoff in Gymnasium.

## Returned trajectory

`generate_trajectory` returns:

| Field | Type and shape | Meaning |
|---|---|---|
| `prompt_ids` | Long tensor `[P]` | Original prompt |
| `response_ids` | Long tensor `[R]` | Generated actions |
| `old_log_probs` | Float tensor `[R]` | Behavior-policy token log-probabilities |
| `prefixes` | List of $R$ tensors | Pre-action states with increasing lengths |
| `terminated` | Python `bool` | Whether `<EOS>` was generated |
| `truncated` | Python `bool` | Whether the token limit ended generation |

The prompt is cloned before it is returned, and every stored prefix is also cloned.

## Observed run

With seed 32,268, temperature 0.1, and the untrained randomly initialized policy, the reported run was:

```text
Prompt: <BOS> 2 + 3 =
Response: 4 + <BOS> <EOS>
Old log-probabilities: tensor([-0.7336, -0.5543, -2.9979, -1.3069])
Terminated by EOS: True
Truncated by limit: False
```

The response contains four generated tokens, so:

$$
R=4.
$$

The rounded token log-probabilities sum to:

$$
-0.7336-0.5543-2.9979-1.3069
= -5.5927.
$$

The output demonstrates that:

- encoding and decoding are working;
- one token and one log-probability were stored per step;
- generation stopped immediately after `<EOS>`;
- `terminated` and `truncated` have the correct values; and
- the generated `<BOS>` token is allowed because the implementation does not mask it.

The first generated token is `4`, but $2+3=5$. The model has not learned arithmetic, and the complete response is not a valid answer. This output is a seeded sample from random initial parameters.

Exact stochastic tokens can also differ across PyTorch versions or devices even when the same seed is used.

## Running the level

From the Level 23 directory:

```bash
python -m py_compile tiny_language_policy.py
python tiny_language_policy.py
```

The first command validates Python syntax. The second constructs the policy and generates one trajectory.

No learning curve or CartPole-style evaluation function is needed because this level performs no training.

## Recommended tests

A focused test suite should verify:

### Token mappings

- every vocabulary token round-trips through encode and decode;
- the result of `encode_tokens` has dtype `torch.long`; and
- an unknown token is rejected.

### Forward pass

- input shape `[B, L]` produces logits `[B, L, V]`;
- changing batch size or sequence length preserves the shape contract;
- non-long inputs are rejected; and
- non-two-dimensional inputs are rejected.

### Forced EOS termination

Configure the output head so `<EOS>` has the largest logit, then check:

- exactly one response token is generated;
- that token is `EOS_ID`;
- `terminated is True`;
- `truncated is False`;
- one prefix is stored; and
- the log-probability is finite.

### Forced truncation

Configure the output head to prefer a non-EOS token, then check:

- response length equals `max_new_tokens`;
- `terminated is False`;
- `truncated is True`;
- prefix lengths are $P,P+1,\ldots,P+R-1$; and
- the three trajectory collections all have length $R$.

### Validation

Check that generation rejects:

- a two-dimensional `prompt_ids` tensor;
- an empty prompt;
- zero or negative `max_new_tokens`;
- zero or negative temperature;
- infinite temperature; and
- a non-numeric temperature.

Run the tests with:

```bash
pytest -q
```

## Common mistakes

### Using the wrong embedding argument

The embedding vocabulary argument is `num_embeddings`, not `num_embedding`. It must equal the number of token IDs that can be passed to the layer.

### Passing token strings directly to PyTorch

`nn.Embedding` expects integer token IDs. Token strings must first be mapped through `TOKEN_TO_ID`.

### Applying layers through `nn.Sequential`

`nn.GRU` returns both its sequence output and final hidden state. Defining the layers separately makes this tuple explicit and allows the code to pass only `gru_output` into the linear head.

### Selecting the wrong logits

Generation needs:

```python
logits[:, -1, :]
```

This selects the vocabulary logits for the final token position in the current prefix.

### Appending before storing the prefix

If the new action is appended first, the stored state already contains the action it is supposed to predict. That breaks state-action alignment.

### Recording only stochastic trajectories

Both stochastic and deterministic branches should continue through the same log-probability, storage, append, and termination logic.

### Comparing a tensor directly as a Python condition

Using `next_token.item()` creates the scalar integer needed for the EOS comparison.

### Treating truncation as EOS

Reaching the token budget means the external limit ended the response. It does not mean the policy selected `<EOS>`.

### Interpreting random output as model capability

This policy has not been trained. A plausible token can occur by chance, especially with a tiny vocabulary.

## Scope and limitations

This level intentionally does not include:

- supervised language-model training;
- a reward or verifier;
- advantages or returns;
- a value model;
- PPO, GRPO, or another optimizer;
- a reference policy or KL penalty;
- masking of invalid tokens such as a generated `<BOS>`;
- batching of variable-length prompts;
- cached GRU hidden states;
- transformer attention; or
- real-world tokenization.

Recomputing the complete prefix at every step is inefficient but transparent. A production generator would normally cache recurrent or transformer state.

## Connection to PPO and RLVR

Levels 16–22 treated a CartPole action as one categorical choice. Level 23 applies the same idea repeatedly:

$$
S_t
\longrightarrow
\pi_\theta(\cdot\mid S_t)
\longrightarrow
A_t.
$$

The response is therefore an RL trajectory whose action space is the vocabulary.

For RL with verifiable rewards, a later verifier can score the completed response. That sequence-level reward must then be connected back to the token actions that produced it. The stored prefixes, actions, and old log-probabilities are the data required to perform such an update.

This level does not claim that RLVR is simply PPO with a verifier. It establishes only the autoregressive trajectory representation shared by many later LLM reinforcement-learning methods.

## Key takeaway

A language model can be viewed as a policy that repeatedly maps a token prefix to a categorical distribution over the next token.

Level 23 implements that policy-environment loop in its smallest useful form:

$$
S_t
\longrightarrow
\pi_\theta(\cdot\mid S_t)
\longrightarrow
A_t
\longrightarrow
S_{t+1}.
$$

The important result is not the generated arithmetic response. It is the correctly aligned trajectory:

$$
(S_t,A_t,\ell_t^0)
$$

for every generated token, together with a correct record of whether the response terminated with `<EOS>` or was truncated by the length limit.

## References

- [PyTorch `Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [PyTorch `GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [PyTorch categorical distributions](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical)
- [PyTorch `torch.no_grad`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)
