# Level 16: PPO Probability Ratios and Clipped Policy Updates

This level extends the GAE actor–critic implementation from Level 15 with the central policy-update mechanism of **Proximal Policy Optimization (PPO)**: probability ratios and the clipped surrogate objective.

The agent still collects an on-policy CartPole rollout and computes GAE advantages. The important change is that it stores the action log-probabilities produced by the rollout policy, then compares them with the probabilities assigned by the updated policy. PPO uses this comparison to reuse the same rollout for several optimization epochs while limiting incentives for excessively large policy changes.

## Learning objectives

- Store fixed action log-probabilities from the rollout policy.
- Re-evaluate the same stored actions under the current policy.
- Compute log-probability differences and probability ratios.
- Clip ratios to a controlled interval.
- Construct PPO's conservative surrogate objective.
- Understand how the sign of the advantage changes clipping behavior.
- Reuse one rollout for multiple actor and critic update epochs.
- Interpret mean ratio, ratio range, and clip fraction.
- Distinguish this educational implementation from full production PPO.

## Actor and critic

The actor represents the categorical policy

$$
\pi_\theta(a\mid s)
$$

with a `4 → 32 → 2` network:

```text
observation → Linear(4, 32) → Tanh → Linear(32, 2) → logits
```

The critic estimates

$$
V_\phi(s)
$$

with a separate `4 → 32 → 1` network:

```text
observation → Linear(4, 32) → Tanh → Linear(32, 1) → value
```

The networks have separate Adam optimizers and do not share parameters.

## Rollout policy and current policy

During episode collection, the actor samples action $A_t$ from the policy available at that time. The implementation stores the corresponding detached log-probability

$$
\ell_t^{\mathrm{old}}
=\log\pi_{\theta_{\mathrm{old}}}(A_t\mid S_t).
$$

The stored value is fixed across every PPO update epoch for that rollout.

During optimization, the current actor is evaluated again on the stored state and the **same stored action**:

$$
\ell_t^{\mathrm{new}}
=\log\pi_\theta(A_t\mid S_t).
$$

No new action is sampled when calculating the PPO loss. Comparing probabilities for different actions would not produce a valid policy ratio.

## Probability ratio

The PPO probability ratio is

$$
r_t(\theta)
=\frac{
\pi_\theta(A_t\mid S_t)
}{
\pi_{\theta_{\mathrm{old}}}(A_t\mid S_t)
}.
$$

The implementation calculates it from log-probabilities:

$$
r_t(\theta)
=\exp\left(
\ell_t^{\mathrm{new}}-
\ell_t^{\mathrm{old}}
\right).
$$

In code:

```python
log_ratios = new_actions_log_probs - old_actions_log_probs
ratios = torch.exp(log_ratios)
```

The ratio has a direct interpretation:

| Ratio | Meaning |
| ---: | --- |
| $r_t=1$ | The action probability is unchanged. |
| $r_t>1$ | The current policy makes the stored action more likely. |
| $r_t<1$ | The current policy makes the stored action less likely. |

Before the first actor update, the current and rollout policies are identical, so ratios should be approximately `1`. They may move away from `1` during later update epochs.

## GAE advantages

Level 16 retains the termination-aware GAE calculation from Level 15. With $d_t=1$ for a true terminal transition and $d_t=0$ otherwise,

$$
m_t=1-d_t,
$$

$$
\delta_t
=R_{t+1}
+\gamma m_tV_\phi(S_{t+1})
-V_\phi(S_t),
$$

and

$$
\hat{A}_t
=\delta_t
+\gamma\lambda m_t\hat{A}_{t+1}.
$$

The raw advantages form fixed critic targets:

$$
\hat{V}_t^{\mathrm{target}}
=\hat{A}_t+V_\phi(S_t).
$$

The actor uses advantages normalized once before the multi-epoch update:

$$
\widetilde{A}_t
=\frac{
\hat{A}_t-\mu_{\hat{A}}
}{
\sigma_{\hat{A}}+\epsilon_{\mathrm{num}}
}.
$$

The normalized advantages, raw advantages, and value targets remain fixed throughout all PPO epochs for that episode.

## Ratio clipping

With clipping parameter $\epsilon$, the ratio is bounded for the clipped surrogate:

$$
\bar{r}_t
=\min\left(
\max\left(r_t,1-\epsilon\right),
1+\epsilon
\right).
$$

This project uses

$$
\epsilon=0.2,
$$

so

$$
\bar{r}_t\in[0.8,1.2].
$$

In PyTorch:

```python
clipped_ratios = torch.clamp(
    ratios,
    min=1.0 - clip_eps,
    max=1.0 + clip_eps,
)
```

Clipping does not force the original ratio itself to stay inside this interval. It changes the optimization objective so that certain policy changes beyond the interval stop receiving additional benefit.

## PPO clipped surrogate objective

The unclipped surrogate is

$$
U_t(\theta)=r_t(\theta)\widetilde{A}_t.
$$

The clipped surrogate is

$$
C_t(\theta)=\bar{r}_t(\theta)\widetilde{A}_t.
$$

PPO selects the more conservative value:

$$
L_t^{\mathrm{CLIP}}(\theta)
=\min\left(U_t(\theta),C_t(\theta)\right).
$$

The actor maximizes this objective. Because PyTorch optimizers minimize losses, the implemented actor loss negates its mean and adds entropy regularization:

$$
L_{\mathrm{actor}}
=-\frac{1}{T}\sum_{t=0}^{T-1}
L_t^{\mathrm{CLIP}}(\theta)
-\beta\frac{1}{T}\sum_{t=0}^{T-1}
\mathcal{H}\left(\pi_\theta(\cdot\mid S_t)\right).
$$

The entropy coefficient is

$$
\beta=0.01.
$$

## Why the minimum depends on advantage sign

The clipping rule is intentionally asymmetric.

### Positive advantage

If $\widetilde{A}_t>0$, the sampled action was better than expected. The optimizer tries to increase its probability. Once $r_t$ rises above $1+\epsilon$, the clipped term prevents further improvement in the objective from that increase.

### Negative advantage

If $\widetilde{A}_t<0$, the sampled action was worse than expected. The optimizer tries to decrease its probability. Once $r_t$ falls below $1-\epsilon$, the clipped term prevents further improvement from that decrease.

The elementwise minimum also preserves penalties for movement in the harmful direction. PPO therefore clips excessive beneficial movement without hiding detrimental policy changes.

## Critic loss

The critic uses the fixed GAE value targets:

$$
L_{\mathrm{critic}}
=\frac{1}{T}\sum_{t=0}^{T-1}
\left(
V_\phi(S_t)-\hat{V}_t^{\mathrm{target}}
\right)^2.
$$

In code:

```python
critic_loss = F.mse_loss(predicted_values, value_targets)
```

The targets are computed once before the PPO epoch loop. The critic predictions are recomputed during every epoch so that gradients reflect the critic's current parameters.

## Multiple update epochs

Each collected episode is reused for

$$
K=4
$$

optimization epochs.

During every epoch:

1. evaluate the current actor on the stored states;
2. obtain new log-probabilities for the stored actions;
3. calculate ratios and clipped surrogate terms;
4. calculate the scalar entropy-regularized actor loss;
5. recompute critic predictions and critic loss;
6. clear the actor and critic gradients;
7. backpropagate both losses;
8. update both networks.

The old log-probabilities, advantages, and value targets never change within this loop. The new log-probabilities and critic predictions do change because the networks are updated.

## Critical implementation invariants

A valid PPO update must preserve the following rules:

- `old_log_probs` are collected under `torch.no_grad()` and remain detached.
- Each old log-probability is stored as one scalar, producing shape `[T]`.
- `new_log_probs` are computed with `distribution.log_prob(actions)` for the stored actions.
- No new action is sampled inside the PPO update loop.
- Advantages and value targets remain detached.
- The actor loss is a scalar obtained with `.mean()`.
- The entropy term is also averaged and multiplied by `entropy_coef`.
- Each optimizer follows `zero_grad() → backward() → step()`.
- The critic gradients are cleared **before** `critic_loss.backward()`.

These are algorithmic requirements, not merely formatting preferences.

## Tensor shapes

For an episode containing $T$ transitions:

| Tensor | Shape |
| --- | --- |
| `states`, `next_states` | `[T, 4]` |
| `actions` | `[T]` |
| `rewards` | `[T]` |
| `terminated` | `[T]` |
| `old_log_probs`, `new_log_probs` | `[T]` |
| policy `logits` | `[T, 2]` |
| rollout values and critic predictions | `[T]` |
| raw and normalized advantages | `[T]` |
| `value_targets` | `[T]` |
| ratios and all surrogate terms | `[T]` |
| actor and critic losses | scalar |

## Termination versus truncation

Episode collection stops on either true termination or time-limit truncation. GAE stops bootstrapping only on true termination.

| Ending | Bootstrap? | Reason |
| --- | --- | --- |
| CartPole failure (`terminated=True`) | No | The underlying task ended. |
| 500-step time limit (`truncated=True`) | Yes | The rollout ended because of an external time limit. |

This matches Level 15 and avoids incorrectly assigning zero future value to a valid truncated state.

## Training configuration

| Parameter | Value |
| --- | ---: |
| Environment | `CartPole-v1` |
| Training episodes | `1000` |
| Discount factor $\gamma$ | `0.99` |
| GAE parameter $\lambda$ | `0.95` |
| PPO clipping parameter $\epsilon$ | `0.2` |
| Update epochs per episode | `4` |
| Entropy coefficient $\beta$ | `0.01` |
| Actor learning rate | `3e-4` |
| Critic learning rate | `1e-3` |
| Hidden dimension | `32` |
| Rollout size | One complete episode |
| Minibatches | None; full episode used each epoch |
| Report interval | `50` episodes |
| Moving-average window | `50` episodes |
| Evaluation episodes | `20` |
| Base seed | `32268` |

The return curve is saved as:

```text
ppo_clipped_cartpole.png
```

## Implementation map

| Component | Purpose |
| --- | --- |
| `PolicyNetwork` | Produces categorical action logits. |
| `ValueNetwork` | Predicts one state value for each observation. |
| `collect_episode(...)` | Collects a rollout and stores old selected-action log-probabilities. |
| `compute_gae(...)` | Computes detached TD errors, GAE advantages, and value targets. |
| `episode_to_tensors(...)` | Converts aligned rollout data, including old log-probabilities, into tensors. |
| `calculate_ppo_terms(...)` | Computes log-ratios, ratios, clipped ratios, and surrogate objectives. |
| `update_ppo(...)` | Reuses one episode for four clipped actor–critic update epochs. |
| `train(...)` | Repeats rollout collection and PPO updates for `1000` episodes. |
| `moving_average(...)` | Smooths training returns over complete windows. |
| `plot_training_history(...)` | Saves raw and moving-average return curves. |
| `evaluate_policy(...)` | Evaluates the learned stochastic actor on separately seeded episodes. |

## PPO diagnostics

The final update epoch reports the following ratio diagnostics.

### Mean ratio

$$
\frac{1}{T}\sum_t r_t
$$

should usually remain reasonably close to `1`. A value near `1` does not mean that every individual action probability is unchanged; increases and decreases can average together.

### Ratio range

The minimum and maximum ratios show the most extreme probability changes in the rollout. The reported **original** ratios may be outside `[0.8, 1.2]`; only `clipped_ratios` are mathematically restricted to that interval.

### Clip fraction

The implementation measures

$$
f_{\mathrm{clip}}
=\frac{1}{T}\sum_t
\mathbf{1}
\left[
\left|r_t-1\right|>\epsilon
\right].
$$

Interpretation:

| Clip fraction | Meaning |
| ---: | --- |
| `0.0` | No sampled ratio crossed the clipping boundary in that epoch. |
| Between `0` and `1` | Some rollout samples were outside the interval. |
| Near `1.0` | Most samples moved beyond the interval, suggesting a very aggressive update. |

A clip fraction of `0.0` is valid. Clipping is a safety mechanism and does not have to activate during every update, especially with a small actor learning rate and only four epochs.

## Interpreting the other metrics

- **Episode return:** the main behavior metric; `CartPole-v1` has a maximum return of `500`.
- **Actor loss:** may remain small or change sign because normalized advantages contain positive and negative values.
- **Critic loss:** can be comparatively large as episode lengths and value targets grow; it should not be judged independently of returns.
- **Policy entropy:** for two actions, its maximum is $\log 2\approx0.693$. Lower entropy indicates a more confident policy.
- **Final update epoch:** should report `4`, confirming that the same rollout was reused for all configured epochs.

Training quality should be judged using return trends, stochastic evaluation performance, finite losses, ratio diagnostics, and evidence that both networks update.

## Validation checks

A focused test suite should verify:

- ratios equal `1` when old and new log-probabilities are equal;
- `exp(new_log_prob - old_log_prob)` gives the expected probability ratio;
- clipped ratios stay inside $[1-\epsilon,1+\epsilon]$;
- the unclipped, clipped, and conservative surrogates match hand calculations;
- positive- and negative-advantage cases select the correct conservative term;
- old log-probabilities and advantages remain detached;
- gradients flow through new log-probabilities but not old log-probabilities;
- stored actions, rather than newly sampled actions, are used during updates;
- actor and critic losses are finite scalars;
- both networks' parameters change after a PPO update;
- old log-probabilities and value targets remain fixed across epochs;
- clip fraction and ratio-range diagnostics are calculated correctly.

For example, let

$$
r=[1.2,0.7],
\qquad
\widetilde{A}=[1,-1],
\qquad
\epsilon=0.2.
$$

Then

$$
\bar{r}=[1.2,0.8],
$$

$$
U=[1.2,-0.7],
$$

$$
C=[1.2,-0.8],
$$

and the elementwise conservative surrogate is

$$
\min(U,C)=[1.2,-0.8].
$$

## Run the project

Install the dependencies:

```bash
python -m pip install numpy gymnasium torch matplotlib pytest
```

From the Level 16 directory:

```bash
python ppo_clipped_cartpole.py
python -m pytest -q
```

## Level 15 versus Level 16

| Property | Level 15: GAE actor–critic | Level 16: clipped PPO |
| --- | --- | --- |
| Rollout signal | GAE advantages | GAE advantages |
| Policy objective | $\log\pi_\theta(A_t\mid S_t)\widetilde{A}_t$ | Clipped probability-ratio surrogate |
| Stored old log-probabilities | Not required | Required |
| Rollout reuse | One update | Four update epochs |
| Large-update control | No PPO constraint | Clipped surrogate objective |
| Ratio diagnostics | None | Mean, range, and clip fraction |

Level 16 keeps GAE and changes how the actor uses its advantages.

## Connection to language-model PPO and RLVR

In language-model reinforcement learning, each generated token can be treated as an action. The rollout model records the selected token's old log-probability, while the trainable policy recomputes its new log-probability. Their ratio can enter the same clipped PPO objective used here.

Verifiable rewards can supply outcome signals for generated solutions, while a value model and advantage estimator assign credit across token positions. Real LLM training additionally requires attention masks, variable-length sequence handling, batching, large-model optimization, reward design, and often reference-policy regularization.

This CartPole project isolates the probability-ratio and clipping mechanics that later appear in larger RL fine-tuning systems. It does not itself fine-tune a language model.

## Scope and limitations

This is a compact educational PPO implementation. It includes:

- on-policy rollout collection;
- GAE and advantage normalization;
- old and new selected-action log-probabilities;
- probability ratios;
- clipped policy surrogates;
- entropy regularization;
- repeated full-rollout update epochs;
- a separately trained critic.

It does not include:

- multi-environment rollout batches;
- minibatch sampling or shuffling;
- value-function clipping;
- gradient-norm clipping;
- target-KL early stopping;
- learning-rate schedules;
- observation or reward normalization;
- production-scale checkpointing.

Evaluation remains stochastic because actions are sampled from the categorical policy rather than chosen with `argmax`.

## Main takeaway

PPO compares the current policy with the rollout policy through

$$
r_t(\theta)
=\exp\left(
\log\pi_\theta(A_t\mid S_t)
-\log\pi_{\theta_{\mathrm{old}}}(A_t\mid S_t)
\right),
$$

then uses the conservative objective

$$
\boxed{
L_t^{\mathrm{CLIP}}(\theta)
=\min\left(
r_t(\theta)\widetilde{A}_t,
\bar{r}_t(\theta)\widetilde{A}_t
\right)
}.
$$

This permits several optimization passes over one rollout while reducing the incentive for the actor to move too far from the policy that collected the data.
