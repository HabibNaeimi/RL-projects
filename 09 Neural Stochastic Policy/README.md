# Level 9: Neural Stochastic Policy with PyTorch

This level replaces a tabular action-selection rule with a neural stochastic policy for `CartPole-v1`. It focuses on the forward pass, categorical action sampling, log-probabilities, trajectory collection, and autograd connectivity. No policy optimization is performed yet.

## Learning objectives

- Represent a policy with `torch.nn.Module`.
- Convert a NumPy observation into a PyTorch tensor.
- Produce action logits rather than hard-coded action values.
- Build and sample a categorical distribution.
- Retain differentiable log-probabilities for a future policy-gradient loss.
- Detach diagnostic probabilities from the computation graph.
- Collect one complete CartPole episode.

## CartPole environment

`CartPole-v1` has a four-dimensional continuous observation:

$$
S_t=[x_t,\dot{x}_t,\theta_t,\dot{\theta}_t],
$$

containing cart position, cart velocity, pole angle, and pole angular velocity.

The discrete action space contains:

| Action | Effect |
| ---: | --- |
| `0` | Push the cart left. |
| `1` | Push the cart right. |

CartPole returns reward `1` at every executed step. Therefore,

$$
\text{episode return}=\sum_{t=0}^{T-1}R_{t+1}=T,
$$

so episode return and episode length are equal. An episode ends because of a failure condition or the 500-step time limit.

## Policy network

The implemented architecture is

```text
4 observations → Linear(4, 32) → Tanh → Linear(32, 2) → action logits
```

For observation $s$, the policy network produces logits

$$
z_\theta(s)=[z_0,z_1].
$$

The categorical action probabilities are the softmax of those logits:

$$
\pi_\theta(a\mid s)
=\frac{\exp(z_a)}{\sum_{j=0}^{1}\exp(z_j)}.
$$

`PolicyNetwork.forward()` returns raw logits. `Categorical(logits=logits)` performs the probability normalization internally in a numerically stable way.

## Stochastic action selection

`choose_action()` performs:

```python
observation_tensor = torch.from_numpy(observation).float()
logits = policy(observation_tensor)
distribution = Categorical(logits=logits)
action_tensor = distribution.sample()
log_prob = distribution.log_prob(action_tensor)
```

The sampled action follows

$$
A_t\sim\pi_\theta(\cdot\mid S_t).
$$

Its log-probability is

$$
\log\pi_\theta(A_t\mid S_t).
$$

The returned values serve different purposes:

| Return value | Gradient status | Purpose |
| --- | --- | --- |
| `action_tensor.item()` | Python integer | Passed to `env.step()`. |
| `log_prob` | Attached to autograd | Used later to train the policy. |
| `probs.detach()` | Detached tensor | Safe inspection without retaining a training dependency. |

Sampling itself is nondifferentiable. Policy-gradient methods learn through the differentiable log-probability of the sampled action.

## Episode collection

`collect_episode()` resets the environment with a seed and interacts until either `terminated` or `truncated` is true. It returns:

```python
{
    "observations": ...,
    "actions": ...,
    "rewards": ...,
    "log_probs": ...,
    "episode_return": ...,
    "episode_length": ...,
}
```

The stored sequences are time-aligned: index $t$ describes $S_t$, $A_t$, $R_{t+1}$, and $\log\pi_\theta(A_t\mid S_t)$.

The global seed is `32268`, applied to NumPy and PyTorch; the environment reset also receives this seed.

## Implementation map

| Component | Purpose |
| --- | --- |
| `PolicyNetwork` | Maps a four-value observation to two action logits. |
| `choose_action(...)` | Samples from the neural policy and returns its log-probability. |
| `collect_episode(...)` | Collects one on-policy trajectory. |
| `main()` | Creates CartPole, builds the policy, and prints one episode. |

## Run the project

Install the dependencies:

```bash
python -m pip install numpy gymnasium torch pytest
```

From the Level 9 directory:

```bash
python cartpole_policy.py
python -m pytest -q
```

## Tests

The test suite checks that:

- the action is valid;
- the two probabilities lie in $[0,1]$ and sum to `1`;
- `log_prob` is a finite scalar that requires gradients;
- a diagnostic repeatedly samples 200 actions from the same policy state;
- episode lists have consistent lengths;
- CartPole return equals episode length;
- backpropagating through a stored log-probability reaches policy parameters.

The gradient-connectivity test uses

$$
L=-\log\pi_\theta(A_t\mid S_t)
$$

only to verify the computation graph. It does not update the policy.

## Scope and expected behavior

The policy begins with random neural-network weights, so one collected episode is not expected to balance the pole well. There is no optimizer or learning loop in this level.

## Main takeaway

A stochastic neural policy does not output an action directly. It parameterizes a distribution:

$$
S_t\rightarrow z_\theta(S_t)\rightarrow\pi_\theta(\cdot\mid S_t)
\rightarrow A_t,\log\pi_\theta(A_t\mid S_t).
$$

The attached log-probabilities are the bridge from sampled behavior to the REINFORCE update in Level 10.
