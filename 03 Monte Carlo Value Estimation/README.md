# Level 3: Every-Visit Monte Carlo State-Value Estimation

This level uses complete episodes to estimate how valuable each state is under the fixed random policy from Level 2. It implements **Monte Carlo prediction**: learning a value function for a policy without changing that policy.

## Learning objectives

- Understand the state-value function \(V_\pi(s)\).
- Use complete sampled returns as learning targets.
- Implement every-visit Monte Carlo prediction.
- Update sample means incrementally without storing all past returns.
- Separate policy evaluation from policy improvement.

## Environment and policy

The same five-state GridWorld is used:

| Setting | Value |
| --- | --- |
| Start state | `0` |
| Terminal goal | `4` |
| Actions | `LEFT = 0`, `RIGHT = 1` |
| Nonterminal reward | `0` |
| Goal-entry reward | `1` |
| Discount factor | \(\gamma=0.9\) |
| Policy | 50/50 random |

The policy remains

$$
\pi(a\mid s)=0.5
$$

for both actions in every nonterminal state. Since the policy is fixed, this level performs prediction, not control.

## State-value function

The value of state \(s\) under policy \(\pi\) is the expected return after visiting that state and then following \(\pi\):

$$
V_\pi(s)=\mathbb{E}_\pi\left[G_t\mid S_t=s\right].
$$

For an individual sampled episode, the observed return is

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
$$

Monte Carlo prediction treats \(G_t\) as a sample target for \(V_\pi(S_t)\).

## Every-visit Monte Carlo update

Every occurrence of a state in an episode is used. If state \(s\) appears several times in the same trajectory, each visit produces a separate update.

For every visited state:

$$
N(s)\leftarrow N(s)+1,
$$

$$
V(s)\leftarrow V(s)+\frac{G_t-V(s)}{N(s)}.
$$

This is an incremental sample mean. After \(N(s)\) visits, it is equivalent to averaging all returns previously observed after visits to \(s\):

$$
V(s)=\frac{1}{N(s)}\sum_{i=1}^{N(s)}G_i(s).
$$

The difference \(G_t-V(s)\) is the current value-prediction error. The effective learning rate \(1/N(s)\) becomes smaller as more evidence is collected.

## Why complete episodes are required

The target \(G_t\) contains all rewards from time \(t\) to the end of the episode. Therefore, the notebook first generates a trajectory, then computes its returns backward, and only then updates \(V\).

This method does not bootstrap: its target does not contain another learned value estimate.

## Implementation map

| Function or array | Purpose |
| --- | --- |
| `run_episode()` | Generates experience under the random policy. |
| `compute_returns(...)` | Calculates \(G_t\) for every transition. |
| `V` | Stores one estimated value per state. |
| `visit_counts` | Stores the number of updates made for each state. |
| `update_values_from_episode(...)` | Applies every-visit incremental-mean updates. |
| `train_mc(...)` | Repeats episode generation and value updates. |

The terminal state remains at zero because the stored transitions contain the state *before* each action. The episode ends upon entering state `4`, so there is no later action or return to estimate from that terminal state.

## Experiment

The notebook trains separate estimators using:

```text
1, 100, 1,000, and 10,000 episodes
```

For each experiment, the value estimates and visit counts are reinitialized. This makes the outputs a comparison of different sample sizes rather than checkpoints from one continuous training run.

## Run the notebook

Install the required packages:

```bash
python -m pip install jupyter numpy
```

Then run from the repository root:

```bash
jupyter notebook "03 Monte Carlo Value Estimation/Monte Carlo Value Estimation.ipynb"
```

## Expected behavior

- Estimates can be noisy after only one or a few episodes.
- With more episodes, the estimates become more stable.
- States nearer the goal should generally have higher values because the terminal reward is discounted over fewer future steps.
- Earlier states are visited more often because every episode begins at state `0` and the random policy can revisit the left side many times.
- The terminal state's stored estimate remains `0.0`.

Exact values vary between runs because the trajectories are randomly generated and no fixed random seed is used.

## Monte Carlo prediction versus control

This notebook answers:

> How good are states when the agent follows this random policy?

It does not answer:

> Which actions should the agent take to improve its behavior?

Answering the second question requires action values and policy improvement, introduced in Level 4.

## Main takeaway

Monte Carlo prediction learns directly from experienced outcomes:

$$
\text{complete episode}\rightarrow\text{return for each visit}\rightarrow\text{updated state values}.
$$

It replaces a single observed return with a progressively more reliable average over many visits.

