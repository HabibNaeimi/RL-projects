# Level 5: Q-Learning; One-Step Off-Policy TD Control

This level replaces episode-end Monte Carlo updates with one-step temporal-difference learning. Q-learning updates an action value immediately after every transition and learns a greedy target policy while the agent continues to explore.

## Learning objectives

- Understand temporal-difference targets and TD errors.
- Learn by bootstrapping from the next state's current estimates.
- Update \(Q(S_t,A_t)\) after every environment step.
- Distinguish the exploratory behavior policy from the greedy target policy.
- Evaluate a learned policy without changing its values.

## Environment and hyperparameters

The same deterministic five-state GridWorld is used:

| Setting | Value |
| --- | ---: |
| Start state | `0` |
| Terminal goal | `4` |
| Actions | `LEFT = 0`, `RIGHT = 1` |
| Goal-entry reward | `1` |
| Other rewards | `0` |
| Training episodes | `5000` |
| Learning rate \(\alpha\) | `0.1` |
| Discount factor \(\gamma\) | `0.9` |
| Exploration rate \(\varepsilon\) | `0.1` |
| Maximum steps per episode | `100` |

## Epsilon-greedy behavior policy

Training actions are selected using

$$
A_t=
\begin{cases}
\text{a random action}, & \text{with probability }\varepsilon,\\
\arg\max_a Q(S_t,a), & \text{with probability }1-\varepsilon.
\end{cases}
$$

Random tie-breaking prevents the initial all-zero table from always selecting the same action.

## Q-learning update

For a nonterminal transition \((S_t,A_t,R_{t+1},S_{t+1})\), the Q-learning target is

$$
y_t=R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a').
$$

The temporal-difference error is

$$
\delta_t=y_t-Q(S_t,A_t),
$$

and the update is

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\delta_t.
$$

Combining these equations gives

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha
\left[R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)\right].
$$

For a terminal transition, there is no future action value to estimate, so

$$
y_t=R_{t+1}.
$$

This terminal case prevents the algorithm from bootstrapping beyond the end of an episode.

## Why this is temporal-difference learning

The algorithm compares two estimates separated by one time step:

$$
Q(S_t,A_t)
\quad\text{and}\quad
R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a').
$$

It therefore learns from the **difference** between the current estimate and a one-step-ahead target. Unlike Monte Carlo learning, it does not wait for the complete return \(G_t\).

## Why Q-learning is off-policy

Two policies appear in the update:

| Role | Policy |
| --- | --- |
| Behavior policy | Epsilon-greedy; generates experience and still explores. |
| Target policy | Greedy; represented by \(\max_{a'}Q(S_{t+1},a')\). |

The action actually taken next does not determine the target. Q-learning updates toward the best estimated next action even when the behavior policy may explore instead. That separation makes the method off-policy.

## Online episode flow

Within each episode, the notebook repeatedly performs:

1. choose \(A_t\) epsilon-greedily;
2. execute it and observe \(S_{t+1},R_{t+1}\);
3. calculate the Q-learning target;
4. calculate \(\delta_t\);
5. update \(Q(S_t,A_t)\) immediately;
6. continue from \(S_{t+1}\), unless it is terminal.

The notebook records total episode reward, episode length, and every TD error.

## Implementation map

| Function | Purpose |
| --- | --- |
| `step(...)` | Implements GridWorld transitions and termination. |
| `greedy_action(state, Q)` | Selects the action with the largest current estimate. |
| `choose_action(state, Q, epsilon)` | Implements the behavior policy. |
| `q_learning_update(...)` | Builds the target, computes the TD error, and updates one table entry. |
| `run_episode(...)` | Interacts with the environment and learns online. |
| `evaluation(...)` | Runs a greedy policy without learning. |

## Run the notebook

Install the dependencies:

```bash
python -m pip install jupyter numpy
```

Then run from the repository root:

```bash
jupyter notebook "05 Q-Learning/Q-learning_gridworld.ipynb"
```

## Expected learned structure

The optimal action is `RIGHT` in every nonterminal state, so training should produce

$$
Q(s,\text{RIGHT})>Q(s,\text{LEFT}),
\qquad s\in\{0,1,2,3\}.
$$

The transition from state `3` directly into the goal has target \(1\), so its learned value should approach

$$
Q(3,\text{RIGHT})=1.
$$

With deterministic dynamics and this reward structure, the optimal right-action values are

$$
Q^*(0,\text{RIGHT})=\gamma^3=0.729,
$$

$$
Q^*(1,\text{RIGHT})=\gamma^2=0.81,
$$

$$
Q^*(2,\text{RIGHT})=\gamma=0.9,
$$

$$
Q^*(3,\text{RIGHT})=1.
$$

Learned values should approach these quantities with sufficient exploration and training.

## Tests and evaluation

The notebook verifies that:

- basic environment transitions are correct;
- `RIGHT` has a larger learned value than `LEFT` in every nonterminal state;
- the goal transition approaches value `1`;
- greedy evaluation does not modify the Q-table.

A successful greedy evaluation should reach the goal in exactly four actions and obtain total reward `1` in every episode.

## Main takeaway

Q-learning replaces a complete sampled return with a bootstrapped one-step target:

$$
\text{act}\rightarrow\text{observe one transition}\rightarrow\text{update immediately}.
$$

The use of \(\max_{a'}Q(S_{t+1},a')\) lets an exploratory agent learn the value of a greedy target policy.

