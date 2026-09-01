# Level 11: Training REINFORCE Across Episodes

This level turns the single policy update from Level 10 into a complete on-policy training experiment on `CartPole-v1`. Each episode is collected with the current policy, used for one REINFORCE update, and then discarded before the next episode is generated.

## Learning objectives

- Train a stochastic policy across many episodes.
- Preserve the on-policy data-collection requirement.
- Track episode returns and policy losses.
- Calculate and plot a moving-average learning curve.
- Evaluate the trained policy without modifying it.
- Observe the high variance of REINFORCE with raw returns.

## Configuration

| Parameter | Value |
| --- | ---: |
| Environment | `CartPole-v1` |
| Training episodes | `800` |
| Episodes per update | `1` |
| Discount factor \(\gamma\) | `0.99` |
| Optimizer | Adam |
| Learning rate | `1e-2` |
| Report interval | `50` episodes |
| Moving-average window | `50` episodes |
| Evaluation episodes | `20` |
| Base seed | `32268` |

The policy architecture remains:

```text
4 observations → Linear(4, 32) → Tanh → Linear(32, 2) → Categorical policy
```

## REINFORCE update

For each trajectory, discounted reward-to-go is computed as

$$
G_t=R_{t+1}+\gamma G_{t+1}.
$$

The policy loss is

$$
L_{\text{REINFORCE}}
=-\sum_{t=0}^{T-1}
G_t\log\pi_\theta(A_t\mid S_t).
$$

Its sampled gradient corresponds to

$$
\widehat{\nabla_\theta J(\theta)}
=\sum_{t=0}^{T-1}
G_t\nabla_\theta\log\pi_\theta(A_t\mid S_t).
$$

The implementation uses raw discounted returns: there is no baseline, return normalization, advantage normalization, or entropy regularization.

## Why training is on-policy

The training cycle is

$$
\pi_{\theta_k}
\rightarrow\text{collect a fresh episode}
\rightarrow\text{update }\theta_k
\rightarrow\pi_{\theta_{k+1}}
\rightarrow\text{discard the old episode}.
$$

The collected log-probabilities describe the policy that generated that episode. After the optimizer changes the policy, the code does not reuse the old data.

## Reproducible episode seeds

When `base_seed` is provided, training episode \(i\) uses

$$
\text{episode seed}_i=\text{base seed}+i.
$$

This gives each episode a different initial environment state while keeping the overall experiment reproducible under the same software and hardware conditions.

## Training history

`train_reinforce()` returns:

```python
{
    "episode_returns": [...],
    "losses": [...],
}
```

Every 50 episodes, it prints the mean of the most recent returns and the newest policy loss.

The moving average for window size \(W\) is

$$
\bar{R}_i=\frac{1}{W}\sum_{j=i-W+1}^{i}R_j.
$$

`plot_training_history()` saves raw returns and their 50-episode moving average to:

```text
reinforce_learning_curve.png
```

## Evaluation

Evaluation uses a separate CartPole environment and runs inside `torch.no_grad()`, so no computation graph or gradient update is created. The evaluation seeds begin after the training range:

```python
base_seed = SEED + NUM_EPISODES
```

Actions are still sampled from the learned categorical policy. Evaluation is therefore stochastic-policy evaluation, not greedy `argmax` evaluation.

The program reports:

- mean return over the first 50 training episodes;
- mean return over the final 50 training episodes;
- mean return over 20 evaluation episodes.

## Implementation map

| Function | Purpose |
| --- | --- |
| `collect_episode(...)` | Collects a fresh on-policy CartPole trajectory. |
| `compute_discounted_returns(...)` | Computes reward-to-go for every step. |
| `compute_reinforce_loss(...)` | Builds the raw-return policy loss. |
| `policy_update(...)` | Applies one optimizer update. |
| `train_reinforce(...)` | Repeats collection and learning across episodes. |
| `moving_average(...)` | Smooths return history for interpretation. |
| `plot_training_history(...)` | Saves the learning curve. |
| `evaluate_policy(...)` | Measures the stochastic policy without updates. |

## Run the project

Install the dependencies:

```bash
python -m pip install numpy gymnasium torch matplotlib pytest
```

From the Level 11 directory:

```bash
python reinforce_train_cartpole.py
python -m pytest -q
```

## Tests

The test suite covers policy outputs, sampling, episode alignment, gradient connectivity, discounted returns, the numerical REINFORCE loss, parameter updates, moving-average values, training-history lengths, finite metrics, and read-only evaluation.

## Expected behavior and limitations

Returns should show whether the policy is learning, but the curve may be noisy or unstable. This is expected because each update uses only one episode and raw Monte Carlo returns. The scalar policy loss is also not a direct performance metric: its scale changes with episode length, returns, and the probabilities assigned by the current policy.

## Main takeaway

REINFORCE can train a neural policy using only sampled trajectories and returns, but its Monte Carlo gradient estimate can have high variance. Level 12 introduces a learned state-value baseline to improve the learning signal.
