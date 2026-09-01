# Level 10: One REINFORCE Policy Update on CartPole

This level takes the stochastic policy from Level 9 and performs one complete REINFORCE update. It connects episode rewards to stored action log-probabilities, computes discounted returns, backpropagates the policy-gradient loss, and verifies that the policy parameters change.

## Learning objectives

- Compute a discounted return for every episode step.
- Build a scalar REINFORCE loss from attached log-probabilities.
- Understand why policy-gradient code minimizes the negative objective.
- Execute the PyTorch update sequence correctly.
- Verify a real parameter change after one optimizer step.

## Policy and environment

The project uses `CartPole-v1` and the same policy network as Level 9:

```text
4 observations → Linear(4, 32) → Tanh → Linear(32, 2) → Categorical policy
```

Actions are sampled on-policy:

$$
A_t\sim\pi_\theta(\cdot\mid S_t).
$$

The configuration is:

| Parameter | Value |
| --- | ---: |
| Discount factor $\gamma$ | `0.99` |
| Optimizer | Adam |
| Learning rate | `1e-2` |
| Seed | `32268` |
| Episodes used | `1` |
| Optimizer steps | `1` |

## Discounted returns

For a trajectory of length $T$, the reward-to-go from step $t$ is

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1}.
$$

The implementation computes it backward:

$$
G_t=R_{t+1}+\gamma G_{t+1}.
$$

For example, if rewards are `[1, 1, 1]` and $\gamma=0.5$, then

$$
[G_0,G_1,G_2]=[1.75,1.5,1.0].
$$

`compute_discounted_returns()` returns a detached `float32` tensor of shape `(T,)`. Returns are learning targets, not differentiable model outputs.

## REINFORCE objective and loss

A sampled policy-gradient estimator is

$$
\widehat{\nabla_\theta J(\theta)}
=\sum_{t=0}^{T-1}G_t\nabla_\theta
\log\pi_\theta(A_t\mid S_t).
$$

PyTorch optimizers minimize a loss, so the code uses the negative sampled objective:

$$
L_{\text{policy}}(\theta)
=-\sum_{t=0}^{T-1}
G_t\log\pi_\theta(A_t\mid S_t).
$$

When $G_t>0$, minimizing this loss increases the log-probability of the sampled action. Actions followed by larger returns receive stronger reinforcement.

The list of scalar log-probabilities is combined with

```python
log_probs_tensor = torch.stack(log_probs)
```

which preserves the autograd graph and creates a tensor aligned with the returns.

## One policy update

`policy_update()` performs the standard PyTorch sequence:

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

The complete data flow is

$$
\text{current policy}
\rightarrow\text{one fresh episode}
\rightarrow\{G_t\}
\rightarrow L_{\text{policy}}
\rightarrow\nabla_\theta L
\rightarrow\text{updated policy}.
$$

The main program clones all parameters before and after the update and reports whether at least one changed.

## Implementation map

| Function | Purpose |
| --- | --- |
| `PolicyNetwork` | Produces two CartPole action logits. |
| `choose_action(...)` | Samples an action and preserves its log-probability. |
| `collect_episode(...)` | Collects one complete on-policy trajectory. |
| `compute_discounted_returns(...)` | Computes reward-to-go targets backward. |
| `compute_reinforce_loss(...)` | Builds the negative policy-gradient objective. |
| `policy_update(...)` | Runs zero-gradient, backward, and optimizer-step operations. |

## Run the project

Install the dependencies:

```bash
python -m pip install numpy gymnasium torch pytest
```

From the Level 10 directory:

```bash
python reinforce_cartpole.py
python -m pytest -q
```

## Tests

In addition to the Level 9 policy and trajectory checks, the tests verify:

- exact discounted returns for a small hand-calculated example;
- the numerical REINFORCE loss for known log-probabilities and returns;
- expected gradients with respect to individual log-probabilities;
- finite loss values;
- a genuine change in policy parameters after the optimizer step.

For the hand-calculated loss,

$$
-[(-0.2)(2)+(-0.7)(1)]=1.1.
$$

## Scope and expected behavior

This level demonstrates one mechanically correct update; it is not a training experiment. One episode is a noisy sample, and one update is not expected to produce a reliable improvement in CartPole performance.

The implementation uses raw discounted returns with no baseline, return normalization, entropy bonus, batching, or gradient clipping.

## Main takeaway

REINFORCE converts outcomes into a learning signal by weighting each sampled action's log-probability with its later return:

$$
\boxed{L_{\text{policy}}=-\sum_tG_t\log\pi_\theta(A_t\mid S_t)}.
$$

