# Level 15: Actor–Critic with Generalized Advantage Estimation (GAE)

This level extends the actor–critic implementation from Level 14 with **Generalized Advantage Estimation (GAE)**. Instead of updating from one transition at a time, the agent collects a complete CartPole episode, computes a multi-step advantage estimate backward through the rollout, and then updates the actor and critic once using the complete episode batch.

GAE introduces a tunable bias–variance tradeoff through the parameter $\lambda$. It is also an important bridge from basic actor–critic methods to algorithms such as PPO.

## Learning objectives

- Compute termination-aware one-step TD errors for a complete rollout.
- Accumulate TD errors backward to obtain GAE advantages.
- Understand the roles of the discount factor $\gamma$ and GAE parameter $\lambda$.
- Construct fixed critic targets from detached rollout values and advantages.
- Normalize advantages for the actor without changing the critic targets.
- Preserve the correct actor and critic gradient paths.
- Bootstrap across time-limit truncation but not true termination.
- Connect GAE to PPO and reinforcement learning for language models.

## Actor and critic

The actor represents the categorical policy

$$
\pi_\theta(a\mid s).
$$

For CartPole, it uses a `4 → 32 → 2` network:

```text
observation → Linear(4, 32) → Tanh → Linear(32, 2) → logits
```

The critic estimates the state value

$$
V_\phi(s),
$$

using a separate `4 → 32 → 1` network:

```text
observation → Linear(4, 32) → Tanh → Linear(32, 1) → value
```

The networks do not share parameters. Each has its own Adam optimizer and learning rate.

## Why use GAE?

Level 14 used the one-step TD error directly as the actor's advantage estimate. That signal has relatively low variance, but it can be biased because it depends heavily on the critic's next-state estimate.

A long-horizon Monte Carlo advantage uses more observed rewards and less bootstrapping, but usually has higher variance. GAE combines TD errors over several future steps and lets $\lambda$ control the balance between these two behaviors.

## One-step TD errors

Let $d_t=1$ when a transition truly terminates the task and $d_t=0$ otherwise. Define the bootstrap mask

$$
m_t=1-d_t.
$$

The one-step TD error is

$$
\delta_t
=R_{t+1}
+\gamma m_t V_\phi(S_{t+1})
-V_\phi(S_t).
$$

For a true terminal transition, $m_t=0$, so the next-state value is removed. For a continuing or time-limit-truncated transition, $m_t=1$, so bootstrapping is retained.

In vectorized form, the implementation calculates

```python
td_errors = rewards - values + gamma * bootstrap_mask * next_values
```

## Generalized Advantage Estimation

GAE accumulates the TD errors backward through the rollout:

$$
\hat{A}_t^{\mathrm{GAE}}
=\delta_t
+\gamma\lambda m_t
\hat{A}_{t+1}^{\mathrm{GAE}}.
$$

The recursion starts after the final collected transition with

$$
\hat{A}_T^{\mathrm{GAE}}=0.
$$

The implementation therefore iterates through the episode in reverse:

```python
for t in reversed(range(T)):
    gae = td_errors[t] + gamma * gae_lambda * bootstrap_mask[t] * gae
    advantages[t] = gae
```

For a sequence without an intervening true termination, the recursive estimate corresponds to a discounted sum of future TD errors:

$$
\hat{A}_t^{\mathrm{GAE}}
=\delta_t
+(\gamma\lambda)\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}
+\cdots.
$$

## Effect of $\lambda$

| Setting | Behavior | Typical tradeoff |
| --- | --- | --- |
| $\lambda=0$ | $\hat{A}_t=\delta_t$ | More bootstrapping, lower variance, more bias |
| $0<\lambda<1$ | Weighted mixture of future TD errors | Intermediate bias and variance |
| $\lambda\approx1$ | Long-horizon estimate | Less bias from one-step bootstrapping, higher variance |

This project uses

$$
\gamma=0.99,
\qquad
\lambda=0.95.
$$

When an episode ends because of CartPole's time limit rather than a true terminal state, the final target still includes the critic's estimate of the state beyond the cutoff. Therefore, $\lambda=1$ is not necessarily identical to an unbootstrapped Monte Carlo return in a truncated rollout.

## Fixed value targets

After computing the raw advantages, the critic target is

$$
\hat{V}_t^{\mathrm{target}}
=\hat{A}_t^{\mathrm{GAE}}+V_\phi(S_t).
$$

Both the rollout values and GAE calculation are evaluated under `torch.no_grad()`. Consequently, the resulting advantages and value targets are fixed learning targets rather than gradient-bearing critic predictions.

This distinction is essential:

- `predicted_values` must require gradients because the critic is trained through them;
- `value_targets` must not require gradients because they define where the prediction should move.

## Advantage normalization

The raw GAE advantages are normalized before they are used by the actor:

$$
\widetilde{A}_t
=\frac{
\hat{A}_t-\mu_{\hat{A}}
}{
\sigma_{\hat{A}}+\epsilon_{\mathrm{num}}
}.
$$

The implementation uses the population standard deviation through `std(unbiased=False)` and adds the floating-point epsilon for numerical stability.

Normalization changes the scale of the actor update and often improves optimization stability. It does **not** replace the raw advantages used to construct the critic targets.

## Actor loss

The policy is re-evaluated on all stored states so that the selected-action log-probabilities remain connected to the actor parameters. The actor loss is

$$
L_{\mathrm{actor}}
=-\frac{1}{T}\sum_{t=0}^{T-1}
\log\pi_\theta(A_t\mid S_t)
\widetilde{A}_t
-\beta\frac{1}{T}\sum_{t=0}^{T-1}
\mathcal{H}\left(\pi_\theta(\cdot\mid S_t)\right).
$$

The entropy term encourages continued exploration. This project uses

$$
\beta=0.01.
$$

The normalized advantages are detached, so actor-loss backpropagation updates the policy but not the critic.

## Critic loss

The critic is regressed onto the fixed GAE value targets with mean squared error:

$$
L_{\mathrm{critic}}
=\frac{1}{T}\sum_{t=0}^{T-1}
\left(
V_\phi(S_t)-\hat{V}_t^{\mathrm{target}}
\right)^2.
$$

In PyTorch, this is implemented as

```python
critic_loss = F.mse_loss(predicted_values, value_targets)
```

Here, `F` is the imported alias for `torch.nn.functional`.

## Episode update sequence

Each training episode follows this order:

1. reset `CartPole-v1` with the episode seed;
2. sample actions from the categorical actor;
3. store states, actions, rewards, next states, and true-termination flags;
4. stop collection on either termination or truncation;
5. convert the complete episode to tensors;
6. evaluate detached current-state and next-state values;
7. compute TD errors, raw GAE advantages, and fixed value targets;
8. normalize the advantages used by the actor;
9. recompute policy logits, log-probabilities, and entropies with gradients enabled;
10. compute gradient-bearing critic predictions;
11. calculate actor and critic losses;
12. clear both optimizers' gradients, backpropagate, and step both optimizers.

This is an on-policy, complete-episode rollout update. It is no longer the per-transition online update used in Level 14.

## Tensor shapes

For an episode containing $T$ transitions:

| Tensor | Shape |
| --- | --- |
| `states` | `[T, 4]` |
| `next_states` | `[T, 4]` |
| `actions` | `[T]` |
| `rewards` | `[T]` |
| `terminated` | `[T]` |
| policy `logits` | `[T, 2]` |
| `values`, `next_values` | `[T]` |
| `td_errors` | `[T]` |
| raw and normalized advantages | `[T]` |
| `value_targets` | `[T]` |
| `log_probs`, `entropies` | `[T]` |

`ValueNetwork.forward()` uses `squeeze(-1)` to convert an output shaped `[T, 1]` into `[T]` without accidentally removing the batch dimension.

## Termination versus truncation

Episode collection stops when

$$
\mathrm{terminated}\lor\mathrm{truncated}
$$

is true. However, the GAE bootstrap mask depends only on true termination.

| Ending | Bootstrap? | Reason |
| --- | --- | --- |
| CartPole failure (`terminated=True`) | No | The underlying task reached a terminal state. |
| 500-step time limit (`truncated=True`) | Yes | The environment wrapper stopped the rollout, but the underlying state is not terminal. |

The episode dictionary therefore stores `terminated`, while truncation is used to stop collection but not to zero the next-state value.

## Training configuration

| Parameter | Value |
| --- | ---: |
| Environment | `CartPole-v1` |
| Training episodes | `1000` |
| Discount factor $\gamma$ | `0.99` |
| GAE parameter $\lambda$ | `0.95` |
| Entropy coefficient $\beta$ | `0.01` |
| Actor learning rate | `3e-4` |
| Critic learning rate | `1e-3` |
| Hidden dimension | `32` |
| Updates | One complete-episode batch per episode |
| Report interval | `50` episodes |
| Moving-average window | `50` episodes |
| Evaluation episodes | `20` |
| Base seed | `32268` |

The return curve is saved as:

```text
gae_actor_critic_cartpole_training.png
```

## Implementation map

| Component | Purpose |
| --- | --- |
| `PolicyNetwork` | Produces categorical action logits. |
| `ValueNetwork` | Predicts one scalar state value per observation. |
| `collect_episode(...)` | Collects a complete on-policy rollout without retaining interaction graphs. |
| `episode_to_tensors(...)` | Converts rollout lists into aligned PyTorch tensors. |
| `compute_gae(...)` | Computes detached TD errors, GAE advantages, and value targets backward through time. |
| `update_actor_critic(...)` | Normalizes actor advantages, calculates both losses, and updates both networks. |
| `train(...)` | Repeats rollout collection and learning for `1000` episodes. |
| `moving_average(...)` | Smooths the return history over complete windows. |
| `plot_training_history(...)` | Saves raw and moving-average training returns. |
| `evaluate_policy(...)` | Evaluates the learned stochastic policy over separate seeded episodes. |

## Interpreting the reported metrics

The script reports recent returns, actor loss, critic loss, policy entropy, and the mean raw advantage every 50 episodes.

- **Episode return:** the most direct measure of behavior. In `CartPole-v1`, the maximum episode return is `500`.
- **Actor loss:** it can be small, change sign, or oscillate around zero because the advantages are normalized to approximately zero mean. Its magnitude should not be interpreted like a supervised-learning error.
- **Critic loss:** it can be numerically large, particularly as episodes become longer and value targets increase. A temporary increase does not by itself prove that learning has failed.
- **Policy entropy:** with two actions, the maximum is $\log 2\approx0.693$. Values near this maximum indicate an uncertain policy; lower values indicate more confident action selection.
- **Mean raw advantage:** positive values often mean the critic underestimated the observed rollout outcomes, while negative values suggest overestimation.

Training should primarily be judged using the return curve, the final moving average, evaluation returns, finite losses, and evidence that both networks' parameters change.

## Validation checks

The implementation contains shape and gradient assertions for the central tensors. A focused test suite should additionally verify:

- correct TD errors for continuing and terminal transitions;
- zero next-state bootstrapping on true termination;
- retained bootstrapping on time-limit truncation;
- exact reverse-time GAE recursion on a small hand-calculated rollout;
- $\lambda=0$ reducing GAE to one-step TD errors;
- detached advantages and value targets;
- normalized actor advantages with finite values;
- gradient-connected policy log-probabilities and critic predictions;
- parameter changes in both networks after an update;
- finite returned metrics and CartPole's return–length equality.

For example, consider

$$
\gamma=0.9,
\qquad
\lambda=0.5,
$$

with rewards `[1, 1]`, values `[0.5, 0.25]`, next values `[0.25, 0]`, and termination flags `[False, True]`. Then

$$
\delta_0=1+0.9(0.25)-0.5=0.725,
$$

$$
\delta_1=1-0.25=0.75,
$$

and the backward recursion gives

$$
\hat{A}_1=0.75,
$$

$$
\hat{A}_0=0.725+(0.9)(0.5)(0.75)=1.0625.
$$

The corresponding fixed value targets are `[1.5625, 1.0]`.

## Run the project

Install the dependencies:

```bash
python -m pip install numpy gymnasium torch matplotlib pytest
```

From the Level 15 directory:

```bash
python gae_actor_critic_cartpole.py
python -m pytest -q
```

## Level 14 versus Level 15

| Property | Level 14: TD(0) actor–critic | Level 15: GAE actor–critic |
| --- | --- | --- |
| Advantage | One-step TD error $\delta_t$ | Discounted combination of future TD errors |
| Update timing | Every environment transition | Once after each complete episode |
| Update data | One transition | Complete episode batch |
| Bias–variance control | Primarily determined by one-step bootstrapping | Explicitly controlled by $\lambda$ |
| Advantage normalization | No | Yes, for the actor |
| Critic target | One-step TD target | GAE-derived value target |

## Connection to PPO and RLVR

GAE is commonly used to construct the advantages that train PPO's policy and value function. This level already implements several important ingredients:

- on-policy rollout collection;
- value-function predictions;
- termination-aware bootstrapping;
- GAE advantages;
- advantage normalization;
- actor and critic losses;
- entropy regularization.

It is **not PPO** because it does not use an old-policy probability ratio, a clipped surrogate objective, minibatches, or multiple optimization epochs over the same rollout.

In reinforcement learning for language models, a generated sequence can be viewed as a trajectory and each generated token as an action. A reward or verifiable outcome can be propagated backward through token positions using returns or advantages. The environment, model scale, masking, reward design, and optimization details are different, but the core idea of assigning credit to earlier actions is directly related.

## Scope

This is a single-environment, on-policy actor–critic implementation with one update per complete CartPole episode. It is not parallel A2C, PPO, an off-policy replay method, or an LLM fine-tuning system.

Evaluation remains stochastic because actions are sampled from the learned categorical policy instead of selected with `argmax`.

## Main takeaway

GAE replaces a single one-step advantage with a controlled multi-step estimate:

$$
\boxed{
\hat{A}_t^{\mathrm{GAE}}
=\delta_t
+\gamma\lambda(1-d_t)
\hat{A}_{t+1}^{\mathrm{GAE}}
}.
$$

The parameter $\lambda$ determines how much future TD information contributes to the current policy update, providing a practical bridge between one-step actor–critic learning and the advantage estimation used by PPO.
