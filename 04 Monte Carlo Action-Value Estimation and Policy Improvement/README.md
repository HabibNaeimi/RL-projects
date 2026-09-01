# Level 4: Monte Carlo Action-Value Estimation and Policy Improvement

This level extends Monte Carlo prediction from state values $V(s)$ to action values $Q(s,a)$. Estimating actions separately makes it possible to derive a greedy policy and compare it with the random behavior policy.

## Learning objectives

- Understand the action-value function $Q_\pi(s,a)$.
- Implement every-visit Monte Carlo updates for state–action pairs.
- Extract a greedy action from learned action values.
- Perform policy improvement after evaluating a fixed policy.
- Compare policies using average episode length.

## Environment

The notebook retains the five-state GridWorld from Levels 2 and 3:

| Setting | Value |
| --- | --- |
| States | `0` through `4` |
| Start state | `0` |
| Terminal goal | `4` |
| Actions | `LEFT = 0`, `RIGHT = 1` |
| Reward | `1` when entering the goal; otherwise `0` |
| Discount factor | $\gamma=0.9$ |
| Data-generating policy | 50/50 random |

The action-value and count tables have shape

$$
|\mathcal{S}|\times|\mathcal{A}|=5\times2.
$$

## Action-value function

For a policy $\pi$, the value of selecting action $a$ in state $s$ and then continuing under $\pi$ is

$$
Q_\pi(s,a)=\mathbb{E}_\pi\left[G_t\mid S_t=s,A_t=a\right].
$$

Unlike $V_\pi(s)$, which averages over the policy's possible actions, $Q_\pi(s,a)$ preserves the value of each decision separately.

## Every-visit Monte Carlo action-value update

After a complete episode, the notebook computes

$$
G_t=R_{t+1}+\gamma G_{t+1}
$$

backward through the trajectory. For every visited pair $(S_t,A_t)$, it applies

$$
N(S_t,A_t)\leftarrow N(S_t,A_t)+1,
$$

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)
+\frac{G_t-Q(S_t,A_t)}{N(S_t,A_t)}.
$$

This is an incremental average of all returns observed after that state–action pair. Because this is **every-visit** Monte Carlo, repeated occurrences of the same pair within one episode are all used.

## Policy improvement

Once $Q$ has been estimated, the greedy action in each nonterminal state is

$$
\pi_{\text{greedy}}(s)=\arg\max_a Q(s,a).
$$

In this environment, moving right leads toward the only rewarding state, so the learned greedy action should be `RIGHT` for states `0` through `3` after sufficient sampling.

The notebook then compares:

- the original random policy used to generate the Monte Carlo data;
- a fixed greedy policy that repeatedly selects the learned action.

The evaluation metric is average episode length. The shortest possible path from state `0` to state `4` takes four actions, so a successful greedy policy should average four steps, while the random policy should take longer.

## Scope of this implementation

This notebook evaluates a fixed random policy and performs a greedy improvement **after** estimating its action values. The behavior policy is not changed during training. It is therefore a clear demonstration of Monte Carlo action-value prediction plus one policy-improvement step, rather than a full iterative Monte Carlo control algorithm.

A full Monte Carlo control method would repeatedly improve the policy while continuing to guarantee exploration, for example through an epsilon-soft policy.

## Implementation map

| Function or array | Purpose |
| --- | --- |
| `run_episode(...)` | Generates a trajectory using either the random policy or an optional fixed action. |
| `compute_returns(...)` | Computes one discounted return per transition. |
| `Q` | Stores the estimated value of each state–action pair. |
| `visit_counts` | Counts updates for each state–action pair. |
| `update_q_from_episode(...)` | Performs every-visit Monte Carlo updates. |
| `train_mc_q(...)` | Trains new action-value tables from sampled episodes. |
| `greedy_action(state, Q)` | Returns the action with the largest learned value. |
| `evaluation(...)` | Compares random and greedy episode lengths. |

## Experiment

Independent estimates are trained with:

```text
1, 10, 100, and 1,000 episodes
```

Each experiment starts from a zero-valued table. More episodes should produce more stable estimates and a more reliable greedy policy.

## Run the notebook

Install the dependencies:

```bash
python -m pip install jupyter numpy
```

Then run from the repository root:

```bash
jupyter notebook "04 Monte Carlo Action-Value Estimation and Policy Improvement/Monte Carlo Action-Value Estimation and Policy Improvement.ipynb"
```

## Expected behavior

- Visit counts grow for both actions because the training policy is random.
- Estimates near the goal stabilize faster because their returns depend on fewer future transitions.
- For each nonterminal state, $Q(s,\text{RIGHT})$ should eventually exceed $Q(s,\text{LEFT})$.
- The extracted greedy policy should move right.
- Greedy evaluation should reach the goal in four steps, compared with a longer average trajectory under the random policy.

Exact Monte Carlo estimates differ between runs because episodes are sampled randomly and no fixed seed is used.

## Main takeaway

State values say whether a situation is promising; action values say which decision is preferable in that situation. This change makes policy improvement possible:

$$
\text{sampled returns}\rightarrow Q(s,a)\rightarrow\arg\max_a Q(s,a)\rightarrow\text{improved policy}.
$$

