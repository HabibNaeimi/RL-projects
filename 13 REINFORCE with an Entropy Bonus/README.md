# Level 13 — REINFORCE with a Value Baseline and Entropy Bonus

This level extends Level 12 by adding policy entropy to the actor loss. The value baseline supplies a lower-variance learning signal, while the entropy bonus discourages the categorical policy from becoming deterministic too early.

## Learning objectives

- Compute categorical policy entropy.
- Interpret entropy as uncertainty in action selection.
- Add entropy regularization with the correct sign.
- Keep entropy connected to policy parameters.
- Track mean entropy during training.
- Understand the exploration–exploitation role of the entropy coefficient.

## Base algorithm

The policy and value networks retain the Level 12 architectures. Complete-episode discounted returns are

$$
G_t=R_{t+1}+\gamma G_{t+1},
$$

and advantages are

$$
\hat{A}_t=G_t-V_\phi(S_t).
$$

The detached advantage is used in the REINFORCE term:

$$
L_{\text{REINFORCE}}
=-\sum_t
\log\pi_\theta(A_t\mid S_t)
\operatorname{stopgrad}(\hat{A}_t).
$$

The value network is trained with

$$
L_{\text{value}}
=\frac{1}{T}\sum_t
\left(V_\phi(S_t)-G_t\right)^2.
$$

## Policy entropy

For a discrete policy, entropy at state \(s\) is

$$
\mathcal{H}\!\left(\pi_\theta(\cdot\mid s)\right)
=-\sum_a\pi_\theta(a\mid s)
\log\pi_\theta(a\mid s).
$$

CartPole has two actions, so

$$
0\le\mathcal{H}\le\log 2\approx0.693.
$$

| Policy probabilities | Entropy | Interpretation |
| --- | ---: | --- |
| `[1.0, 0.0]` or `[0.0, 1.0]` | Approximately `0` | Nearly deterministic |
| `[0.5, 0.5]` | \(\log2\approx0.693\) | Maximum uncertainty |

`Categorical.entropy()` returns a scalar tensor connected to the policy network. It is stored for every episode step so it can affect the gradient.

## Entropy-regularized policy loss

The implemented loss is

$$
L_{\text{policy}}
=L_{\text{REINFORCE}}
-\beta\sum_t\mathcal{H}
\left(\pi_\theta(\cdot\mid S_t)\right),
$$

where \(\beta\ge0\) is the entropy coefficient. Because the optimizer minimizes the loss, subtracting entropy encourages it to increase.

The project uses

$$
\beta=0.01.
$$

Setting \(\beta=0\) recovers the Level 12 policy loss. A negative coefficient is rejected with `ValueError` because it would reverse the intended regularization behavior.

## Effect of the coefficient

- A coefficient that is too small has little influence.
- A moderate coefficient can preserve useful exploration while the policy learns.
- A coefficient that is too large can keep the policy unnecessarily random and weaken exploitation.

This implementation uses a fixed coefficient; it does not anneal entropy regularization over training.

## Training configuration

| Parameter | Value |
| --- | ---: |
| Training episodes | `800` |
| Discount factor \(\gamma\) | `0.99` |
| Entropy coefficient \(\beta\) | `0.01` |
| Policy learning rate | `1e-2` |
| Value learning rate | `1e-3` |
| Episodes per update | `1` |
| Report interval | `50` |
| Moving-average window | `50` |
| Evaluation episodes | `20` |
| Base seed | `32268` |

Training records episode returns, policy losses, value losses, mean absolute advantages, and mean policy entropy. The return curve is saved as:

```text
reinforce_entropy_curve.png
```

## Implementation map

| Component | Purpose |
| --- | --- |
| `choose_action(...)` | Returns action, log-probability, detached probabilities, and attached entropy. |
| `collect_episode(...)` | Stores one entropy tensor per interaction step. |
| `compute_losses_with_baseline_and_entropy(...)` | Combines advantage weighting, entropy regularization, and value regression. |
| `update_policy_and_value_with_entropy(...)` | Updates both networks after a complete episode. |
| `train_reinforce_with_entropy(...)` | Trains and tracks entropy across episodes. |

## Run the project

Install the dependencies:

```bash
python -m pip install numpy gymnasium torch matplotlib pytest
```

From the Level 13 directory:

```bash
python reinforce_entropy_cartpole.py
python -m pytest -q
```

## Tests

The test suite verifies:

- value-network outputs and two-network parameter updates;
- the Level 12 loss when \(\beta=0\);
- maximum entropy \(\log2\) for a uniform two-action policy;
- the exact entropy-regularized loss for a hand-calculated example;
- entropy gradients with the correct negative sign;
- storage, finiteness, scalar shape, and gradient connectivity of episode entropies;
- continued gradient separation from value predictions.

For the numerical entropy test,

$$
L_{\text{policy}}=-0.3-0.1(0.5+0.7)=-0.42.
$$

## Scope

This remains complete-episode REINFORCE with a Monte Carlo value baseline. Entropy does not turn it into TD actor–critic, PPO, or an off-policy method. Evaluation continues to sample from the stochastic policy.

## Main takeaway

The policy is now optimized for both return-informed behavior and controlled uncertainty:

$$
\boxed{
L_{\text{policy}}
=-\sum_t\log\pi_\theta(A_t\mid S_t)\hat{A}_t
-\beta\sum_t\mathcal{H}_t
}.
$$

