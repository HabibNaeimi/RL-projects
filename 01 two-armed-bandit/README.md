# Level 1: Two-Action Stochastic Bandit

This level implements a two-action multi-armed bandit from scratch using only Python. It introduces the smallest complete reinforcement-learning loop:

1. choose an action;
2. receive a stochastic reward;
3. update the estimated value of that action;
4. use the improved estimates to make later decisions.

Unlike later levels, this problem has no states, episodes, or delayed rewards. Each action produces an immediate reward.

## Learning objectives

- Represent learned action values as $Q(a)$.
- Balance exploration and exploitation with an $\varepsilon$-greedy policy.
- Interpret the prediction error $R_t-Q_t(A_t)$.
- Update only the value of the selected action.
- Verify that repeated interaction improves the agent's behavior.

## Environment

The agent chooses between two actions. The reward is binary:

$$
R_t \in \{0,1\}.
$$

Each action has a hidden probability of producing reward $1$:

| Action | Hidden reward probability |
| --- | ---: |
| `0` | $0.2$ |
| `1` | $0.8$ |

The agent does not receive these probabilities directly. It must estimate them from sampled rewards.

The true value of an action is its expected immediate reward:

$$
q_*(a)=\mathbb{E}[R_t\mid A_t=a].
$$

For this Bernoulli environment, the true values are $q_*(0)=0.2$ and $q_*(1)=0.8$.

## Agent

The learned estimate for action $a$ is denoted by $Q_t(a)$. Both estimates begin at zero:

$$
Q_0(0)=Q_0(1)=0.
$$

### Epsilon-greedy action selection

At every step, the agent explores with probability $\varepsilon$ and otherwise exploits its current estimates:

$$
A_t=
\begin{cases}
\text{a uniformly random action}, & \text{with probability } \varepsilon,\\
\arg\max_a Q_t(a), & \text{with probability } 1-\varepsilon.
\end{cases}
$$

This implementation uses $\varepsilon=0.1$. Ties between the two estimated values are broken randomly.

### Constant-step-size value update

After selecting $A_t$ and observing $R_t$, the prediction error is

$$
\delta_t=R_t-Q_t(A_t).
$$

The estimate for the selected action is updated using

$$
Q_{t+1}(A_t)=Q_t(A_t)+\alpha\delta_t,
$$

or equivalently,

$$
Q_{t+1}(A_t)=Q_t(A_t)+\alpha\left[R_t-Q_t(A_t)\right].
$$

The learning rate is $\alpha=0.1$. The value of the action that was not selected remains unchanged.

## Implementation map

| Component | Function or variable | Purpose |
| --- | --- | --- |
| Hidden environment | `reward_probabilities` | Stores the reward probability for each action. |
| Reward sampling | `get_reward(action)` | Samples a binary reward for the selected action. |
| Greedy choice | `greedy_action(q_values)` | Selects the action with the larger estimate and handles ties randomly. |
| Behavior policy | `choose_action(q_values, epsilon)` | Implements epsilon-greedy exploration. |
| Learning rule | `update_q(...)` | Applies the constant-step-size update. |
| Diagnostics | `action_counts`, `total_reward` | Tracks behavior and accumulated reward. |

The training configuration is:

| Parameter | Value |
| --- | ---: |
| Training steps | `1000` |
| Learning rate $\alpha$ | `0.1` |
| Exploration rate $\varepsilon$ | `0.1` |

## Training loop

For each interaction step, the notebook performs

$$
Q_t \rightarrow A_t \rightarrow R_t \rightarrow \delta_t \rightarrow Q_{t+1}.
$$

The notebook prints intermediate estimates and action counts every 100 steps, followed by the learned values, total reward, and final greedy action.

## Run the notebook

From the repository root:

```bash
jupyter notebook "two-armed-bandit/Two-Action-Bandit.ipynb"
```

The implementation uses Python's standard library; Jupyter is the only notebook dependency.

## Expected behavior

Because action `1` returns reward more frequently, its estimate should usually become larger than the estimate for action `0`:

$$
Q(1)>Q(0).
$$

Consequently, action `1` should be selected much more often and should become the final greedy action. Exact estimates and counts change between runs because rewards, exploration, and tie-breaking are stochastic and no fixed random seed is used.

The final sampling check repeatedly selects action `0`; its empirical mean reward should be close to $0.2$ when enough samples are collected.

## Main takeaway

The agent is never told which action is better. It discovers this through the repeated cycle

$$
\text{action} \rightarrow \text{reward} \rightarrow \text{prediction error} \rightarrow \text{updated estimate}.
$$

This same learning pattern reappears in later value-based and policy-based RL methods, although the prediction targets become more sophisticated.

