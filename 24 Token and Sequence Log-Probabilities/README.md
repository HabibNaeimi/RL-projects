# Level 24 — Token and Sequence Log-Probabilities

This level turns a generated trajectory into the probability quantities needed for policy-gradient training. The policy first generates a response one token at a time and stores the log-probability of every sampled token. It then scores that fixed response again in one gradient-connected forward pass.

The main ideas are:

- recompute the log-probability of each response token with teacher forcing;
- preserve the causal next-token alignment;
- sum token log-probabilities into one sequence log-probability;
- distinguish detached rollout data from differentiable current-policy values;
- verify old and recomputed log-probabilities with per-token probability ratios.

The policy is still randomly initialized. This level checks probability bookkeeping and tensor alignment; it does not train the model or teach arithmetic.

## Files

- `sequence_log_probs.py` — tiny autoregressive policy, trajectory generation, response rescoring, sequence scoring, and consistency checks

## Requirements

- Python 3.10 or newer
- PyTorch

Install PyTorch if it is not already available:

```bash
pip install torch
```

Run the example:

```bash
python sequence_log_probs.py
```

## Configuration

| Setting | Value | Purpose |
|---|---:|---|
| Random seed | `32268` | Makes model initialization and sampling reproducible |
| Vocabulary size | `10` | Two special tokens, six digits, `+`, and `=` |
| Embedding size | `16` | Width of each token embedding |
| GRU hidden size | `32` | Width of the recurrent state |
| Maximum new tokens | `6` | Generation limit |
| Temperature | `1.0` | Leaves the logits at their original scale |

The vocabulary is:

```text
<BOS> <EOS> 0 1 2 3 4 5 + =
```

The example prompt is:

```text
<BOS> 2 + 3 =
```

## Model

`TinyAutoregressivePolicy` contains three layers:

1. `nn.Embedding` maps token IDs to vectors.
2. `nn.GRU` processes the sequence from left to right.
3. `nn.Linear` maps every GRU output to one logit per vocabulary token.

For batch size $B$, sequence length $L$, embedding size $E$, hidden size $H$, and vocabulary size $V$, the shapes are:

| Stage | Shape |
|---|---|
| Token IDs | `[B, L]` |
| Embeddings | `[B, L, E]` |
| GRU outputs | `[B, L, H]` |
| Vocabulary logits | `[B, L, V]` |

Every output position predicts the token that comes immediately after that position.

## Autoregressive response probability

Let the prompt be $X=(x_1,\ldots,x_P)$ and the generated response be $Y=(y_1,\ldots,y_R)$. An autoregressive policy factorizes the response probability as:

$$
\pi_\theta(Y\mid X)
=
\prod_{t=1}^{R}
\pi_\theta(y_t\mid X,y_1,\ldots,y_{t-1}).
$$

The token log-probability at response step $t$ is:

$$
\ell_t(\theta)
=
\log \pi_\theta(y_t\mid X,y_1,\ldots,y_{t-1}).
$$

If $z_{t,a}$ is the logit for token $a$ and $\tau$ is the temperature, the policy probability is:

$$
p_t(a)
=
\frac{\exp(z_{t,a}/\tau)}
{\sum_{j=1}^{V}\exp(z_{t,j}/\tau)}.
$$

The selected token's log-probability is therefore:

$$
\ell_t(\theta)
=
\frac{z_{t,y_t}}{\tau}
-
\log\sum_{j=1}^{V}\exp(z_{t,j}/\tau).
$$

`Categorical(logits=...)` performs this normalization internally, so the code can obtain all selected-token log-probabilities with:

```python
distribution = Categorical(logits=response_logits / temperature)
token_log_probs = distribution.log_prob(response_ids)
```

`response_ids` contains category IDs. Passing the logits themselves to `log_prob` would be incorrect.

## Generation and recomputation serve different roles

| Phase | Input processing | Gradient tracking | Result |
|---|---|---|---|
| Rollout generation | One forward pass per sampled token | Disabled by `@torch.no_grad()` | Response tokens and detached old log-probabilities |
| Response recomputation | One teacher-forced pass over the fixed response | Enabled | Current, differentiable token log-probabilities |

During generation, the policy samples a token, appends it to the prefix, and repeats. The sampled log-probabilities are converted to Python numbers and later rebuilt as a tensor, so they are fixed rollout data:

```python
old_log_probs.append(float(log_prob.item()))
```

During recomputation, the stored response is not sampled again. It is treated as the target sequence, and the policy scores every stored token while preserving the computation graph.

This distinction is essential for later policy updates:

- old log-probabilities describe the behavior policy that produced the trajectory;
- recomputed log-probabilities describe the current policy;
- gradients must flow only through the current values.

## Teacher forcing

Suppose the prompt contains $P$ tokens and the response contains $R$ tokens. To score all response tokens, the model receives:

$$
(x_1,\ldots,x_P,y_1,\ldots,y_{R-1}).
$$

In code:

```python
model_input_ids = torch.cat(
    [prompt_ids, response_ids[:-1]],
    dim=0,
)
```

The final response token is excluded from the input because it is a target, not context for another response token. The combined input has length:

$$
P+R-1.
$$

### Why every response token can still be scored

The prompt's last position predicts $y_1$. After $y_1$ is included in the input, its position predicts $y_2$, and so on.

| Logit position | Available context | Target |
|---:|---|---|
| $P-1$ | $x_1,\ldots,x_P$ | $y_1$ |
| $P$ | $x_1,\ldots,x_P,y_1$ | $y_2$ |
| $P+1$ | $x_1,\ldots,x_P,y_1,y_2$ | $y_3$ |
| $\ldots$ | $\ldots$ | $\ldots$ |
| $P+R-2$ | $x_1,\ldots,x_P,y_1,\ldots,y_{R-1}$ | $y_R$ |

The response-aligned logits therefore begin at `prompt_length - 1`:

```python
response_logits = all_logits[
    :,
    prompt_length - 1:,
    :,
].squeeze(0)
```

This slice contains exactly $R$ rows, one for every response token.

### Why the full response is not used as input

Concatenating the complete response would produce an input of length $P+R$ and one unnecessary prediction after $y_R$. More importantly, using the wrong slice could shift each target onto the wrong logit or accidentally let a target token appear in its own context.

The correct causal pairing is:

```text
input through prompt end       -> first response token
input through response token 1 -> second response token
input through response token 2 -> third response token
...
```

## Shape trace for this run

The prompt has five tokens and the generated response has four:

```text
prompt   = <BOS> 2 + 3 =
response = 4 + <BOS> <EOS>
```

Therefore, $P=5$, $R=4$, and $V=10$.

| Tensor | Construction | Shape |
|---|---|---|
| `prompt_ids` | Complete prompt | `[5]` |
| `response_ids` | Complete sampled response | `[4]` |
| `response_ids[:-1]` | Response context only | `[3]` |
| `model_input_ids` | Prompt plus response context | `[8]` |
| `batched_inputs` | Add batch dimension | `[1, 8]` |
| `all_logits` | Policy output at every input position | `[1, 8, 10]` |
| `response_logits` | Positions `4` through `7` | `[4, 10]` |
| `token_log_probs` | Log-probability of each target ID | `[4]` |
| `sequence_log_prob` | Sum over response tokens | `[]` |

The scalar tensor shape is `[]`, represented in PyTorch by `torch.Size([])`.

## Sequence log-probability

Because the response probability is a product of conditional token probabilities, its log-probability is a sum:

$$
L_Y(\theta)
=
\sum_{t=1}^{R}\ell_t(\theta)
=
\log \pi_\theta(Y\mid X).
$$

The implementation is intentionally simple:

```python
sequence_log_prob = token_log_probs.sum()
```

The corresponding sequence probability is:

$$
\pi_\theta(Y\mid X)=\exp(L_Y(\theta)).
$$

The sum is the log-probability of the whole response. A mean could be useful as a length-normalized comparison score, but it would not be the response's sequence log-probability.

The `<EOS>` token is included in the response and in the sum. Stopping is a policy decision, so its probability is part of the trajectory probability.

## Old-to-current probability ratios

Let $\ell_t^0$ be the detached log-probability recorded during generation. The per-token probability ratio is:

$$
r_t(\theta)
=
\frac{\pi_\theta(y_t\mid X,y_1,\ldots,y_{t-1})}
{\exp(\ell_t^0)}
=
\exp(\ell_t(\theta)-\ell_t^0).
$$

The code uses the numerically stable log-space form:

```python
ratios = torch.exp(
    recomputed_log_probs - trajectory["old_log_probs"]
)
```

No update occurs between generation and recomputation, and both phases use the same temperature. Consequently:

$$
\ell_t(\theta)=\ell_t^0
\quad\Longrightarrow\quad
r_t(\theta)=1.
$$

The response-level ratio could also be written as:

Let $L_Y^0=\sum_{t=1}^{R}\ell_t^0$ be the sum of the stored rollout log-probabilities. Then:

$$
r_Y(\theta)
=
\exp(L_Y(\theta)-L_Y^0)
=
\prod_{t=1}^{R}r_t(\theta).
$$

This level computes the ratios but does not yet define rewards, advantages, clipping, or an optimization loss.

## Gradient behavior

The equality of old and recomputed values is numerical, not computational.

| Tensor | `requires_grad` | Reason |
|---|---:|---|
| `trajectory["old_log_probs"]` | `False` | Saved rollout record |
| `recomputed_log_probs` | `True` | Produced by the current policy with gradient tracking |
| `sequence_log_prob` | `True` | Sum of differentiable token log-probabilities |
| `ratios` | `True` | Depends on the recomputed values |

That is why the printed ratio tensor includes `grad_fn=<ExpBackward0>` even though every displayed value is `1.`. A later parameter update can move the recomputed values and ratios away from their rollout values.

## Example output

With seed `32268` and the original prompt, the program reports:

```text
Prompt: <BOS> 2 + 3 =
Response: 4 + <BOS> <EOS>
Old log-probabilities: tensor([-2.0503, -2.0403, -2.2699, -2.1064])
Terminated by EOS: True
Truncated by limit: False
Ratios of recompute log-probabilities: tensor([1., 1., 1., 1.], grad_fn=<ExpBackward0>)
```

Using the rounded printed values, the sequence log-probability is approximately:

$$
L_Y
\approx
-2.0503-2.0403-2.2699-2.1064
=
-8.4669.
$$

The corresponding sequence probability is approximately:

$$
\exp(-8.4669)\approx 0.00021032.
$$

The code uses full-precision tensors; the calculation above uses values rounded to four decimal places.

## Why another prompt can sample the same response

Changing the prompt to `<BOS> 1 + 2 =` may still produce:

```text
Response: 4 + <BOS> <EOS>
```

This is normal for this example:

1. The policy is randomly initialized and has not learned arithmetic.
2. `torch.manual_seed(SEED)` resets the random number stream when the program starts.
3. Different prompts can produce different probability vectors while the same sequence of random draws still lands on the same token IDs.
4. The changed token log-probabilities show that the prompt did affect the conditional distributions.

For example, the second prompt produced different old log-probabilities:

```text
tensor([-2.0692, -1.9784, -2.2891, -2.1091])
```

The sampled tokens happened to match, but their probabilities did not. The all-one ratios are still expected because each run compares its rollout values with an immediate recomputation by the same unchanged policy.

With ten vocabulary items, a perfectly uniform distribution would assign log-probability:

$$
\log(1/10)\approx -2.3026.
$$

Values near `-2` to `-2.3` are therefore unsurprising for this small, randomly initialized policy.

## Termination and truncation

- `terminated=True` means generation sampled `<EOS>`.
- `truncated=True` means generation reached `MAX_NEW_TOKENS` without sampling `<EOS>`.

Only one flag is true in this implementation because:

```python
truncated = not terminated
```

When `<EOS>` is sampled, it remains in `response_ids` and its log-probability remains in both the token-level and sequence-level scores.

## Validation performed by the program

The safety assertions verify that:

- old and recomputed log-probability tensors both have shape `[R]`;
- the sequence log-probability is a scalar;
- old rollout values are detached;
- recomputed token and sequence values require gradients;
- every log-probability is finite;
- immediate recomputation matches the recorded rollout values within `1e-6`;
- every immediate old-to-current probability ratio is one within `1e-6`.

The reported focused test run passes five tests covering these core behaviors.

## Padding and masks

This implementation scores one unpadded response, so it does not need a padding token or attention mask.

For a batch of responses with different lengths, shorter examples would normally be padded and accompanied by a binary validity mask $m_{b,t}$. A masked sequence log-probability would be:

$$
L_b(\theta)
=
\sum_{t=1}^{R_{\max}}
m_{b,t}\ell_{b,t}(\theta).
$$

Only positions with mask value one should contribute to sequence scores, ratios, losses, or metrics. Padding and batched variable-length scoring are conceptual extensions, not features of the current file.

## Common mistakes

### Starting the response slice at `prompt_length`

The first response token is predicted by the prompt's final position, so slicing from `prompt_length` drops that prediction and shifts the alignment.

Correct:

```python
response_logits = all_logits[:, prompt_length - 1:, :]
```

### Feeding the entire response

For a response longer than one token, the input should end at `response_ids[-2]`. For a one-token response, it ends at the final prompt token. In both cases, the last response token is a target and does not need to become context for another response target.

### Scoring logits instead of token IDs

`Categorical.log_prob` receives the selected category IDs:

```python
distribution.log_prob(response_ids)
```

### Recomputing inside `torch.no_grad()`

That would produce the right values but remove the gradient path required for learning.

### Sampling a new response during recomputation

The current policy must score the stored rollout actions. Sampling again would compare different trajectories.

### Averaging token log-probabilities

The sequence log-probability is their sum. Averaging changes its meaning.

### Excluding `<EOS>`

The stopping action is part of the generated response and must be scored.

### Using a different temperature

Old and recomputed values describe the same policy only when their sampling distributions use the same temperature.

## What this level establishes

After completing this level, the implementation can:

- generate and store an autoregressive trajectory;
- retain detached behavior-policy log-probabilities;
- score the same fixed response in one teacher-forced pass;
- align each response token with the correct causal logit;
- preserve gradients through current token and sequence log-probabilities;
- compute old-to-current per-token probability ratios;
- distinguish a coincidentally repeated sample from an unchanged probability distribution.

These are the probability-building blocks required before a language-model policy can be updated from sequence-level feedback.

## References

- [PyTorch Categorical distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical)
- [PyTorch `torch.no_grad`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)
- [PyTorch `torch.cat`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html)
