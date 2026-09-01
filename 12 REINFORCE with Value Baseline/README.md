# Level 12: REINFORCE with a Learned Value Baseline

This level adds a value network to REINFORCE. The policy still learns from complete-episode Monte Carlo returns, but each return is compared with the critic's prediction to form an advantage estimate. The baseline aims to reduce policy-gradient variance without changing which actions are preferred in expectation.

## Learning objectives

- Distinguish policy logits from scalar state-value predictions.
- Train separate policy and value networks.
- Construct the advantage estimate $G_t-V_\phi(S_t)$.
- Detach advantages in the policy loss.
- Train the value network with mean squared error.
- Understand why this remains episode-based REINFORCE.

## Networks

Both networks receive the four-dimensional CartPole observation.

### Policy network

```text
4 observations → Linear(4, 32) → Tanh → Linear(32, 2) → action logits
```

It represents

$$
\pi_\theta(a\mid s).
$$

### Value network

```text
4 observations → Linear(4, 32) → Tanh → Linear(32, 1) → scalar value
```

It estimates

$$
V_\phi(s)\approx\mathbb{E}_{\pi_\theta}[G_t\mid S_t=s].
$$

`squeeze(-1)` converts batched output from shape `(T, 1)` to `(T,)`, aligning it with the return tensor.

## Returns and advantages

Returns are still calculated after the complete episode:

$$
G_t=R_{t+1}+\gamma G_{t+1}.
$$

The sampled advantage is

$$
\hat{A}_t=G_t-V_\phi(S_t).
$$

Its sign has a direct interpretation:

- $\hat{A}_t>0$: the outcome was better than the critic predicted;
- $\hat{A}_t<0$: the outcome was worse than predicted;
- $\hat{A}_t\approx0$: the outcome was close to expectation.

## Policy loss

The policy loss is

$$
L_{\text{policy}}(\theta)
=-\sum_{t=0}^{T-1}
\log\pi_\theta(A_t\mid S_t)
\hat{A}_t^{\mathrm{detach}}.
$$

In code:

```python
advantages = returns - predicted_values
fixed_advantages = advantages.detach()
policy_loss = -(log_probs_tensor * fixed_advantages).sum()
```

Detaching the advantages is essential. The actor should treat them as fixed weights; policy-loss backpropagation must not update the value network or manipulate its predictions merely to reduce the actor's loss.

## Value loss

The value network is trained to regress toward Monte Carlo returns:

$$
L_{\text{value}}(\phi)
=\frac{1}{T}\sum_{t=0}^{T-1}
\left(V_\phi(S_t)-G_t\right)^2.
$$

This loss supplies gradients to the critic independently of the detached policy advantages.

## Why a baseline helps

Raw REINFORCE weights every log-probability by $G_t$. The value baseline removes the expected return already attributable to being in state $S_t$, leaving a more decision-specific signal:

$$
G_t\quad\longrightarrow\quad G_t-V_\phi(S_t).
$$

A state-dependent, action-independent baseline preserves the expected policy gradient while potentially reducing its variance. The benefit depends on how accurately the value network predicts returns.

## Training configuration

| Parameter | Value |
| --- | ---: |
| Training episodes | `800` |
| Discount factor $\gamma$ | `0.99` |
| Policy learning rate | `1e-2` |
| Value learning rate | `1e-3` |
| Episodes per update | `1` |
| Report interval | `50` |
| Moving-average window | `50` |
| Evaluation episodes | `20` |
| Base seed | `32268` |

Both optimizers have their gradients cleared, both losses are backpropagated, and then both optimizers step once per completed episode.

Training tracks episode returns, policy losses, value losses, and mean absolute advantages. The learning curve is saved as:

```text
reinforce_value_baseline_curve.png
```

## Implementation map

| Component | Purpose |
| --- | --- |
| `PolicyNetwork` | Produces the categorical policy logits. |
| `ValueNetwork` | Predicts one expected return per observation. |
| `compute_loss_with_baseline(...)` | Creates advantages and separate actor/critic losses. |
| `update_policy_and_value(...)` | Updates both networks after one episode. |
| `train_reinforce_with_baseline(...)` | Runs the full on-policy training loop. |
| `evaluate_policy(...)` | Samples from the learned policy without gradients. |

## Run the project

Install the dependencies:

```bash
python -m pip install numpy gymnasium torch matplotlib pytest
```

From the Level 12 directory:

```bash
python reinforce_value_baseline_cartpole.py
python -m pytest -q
```

## Tests

The tests verify:

- value-network output shape, finiteness, and gradient connectivity;
- exact advantage, policy-loss, and value-loss calculations;
- gradient separation between the policy and value predictions;
- parameter changes in both networks;
- finite update metrics.

The hand-calculated test uses

$$
\hat{A}=[3-1,1-2]=[2,-1]
$$

and confirms that policy-loss backpropagation leaves the value predictions untouched.

## Scope

This is REINFORCE with a learned Monte Carlo baseline, not online TD actor–critic. Updates still wait for a complete episode, and the critic target is $G_t$, not a bootstrapped one-step target. Advantages are not normalized, and only one episode is used per update.

Evaluation samples from the stochastic policy rather than choosing `argmax` actions.

## Main takeaway

The value network does not choose actions. It evaluates the current situation so the policy can learn from whether an action's outcome was better or worse than expected:

$$
\boxed{\hat{A}_t=G_t-V_\phi(S_t)}.
$$
