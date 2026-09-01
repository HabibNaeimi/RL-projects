# Level 6 — SARSA: On-Policy Temporal-Difference Control

This level implements SARSA, a one-step temporal-difference control algorithm. Like Q-learning, SARSA learns online after each transition. Its defining difference is that the update uses the value of the **next action actually selected by the current behavior policy**.

The name comes from the five values used in one update:

$$
S_t,\ A_t,\ R_{t+1},\ S_{t+1},\ A_{t+1}.
$$

## Learning objectives

- Implement an on-policy TD control update.
- Carry both the current action and next selected action through the interaction loop.
- Understand why SARSA's target contains \(Q(S_{t+1},A_{t+1})\).
- Handle terminal transitions without bootstrapping.
- Compare SARSA with Q-learning at the level of one equation.

## Environment and hyperparameters

The notebook uses the same deterministic GridWorld as Level 5:

| Setting | Value |
| --- | ---: |
| States | `0` through `4` |
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

Actions are selected by an epsilon-greedy policy:

$$
A_t\sim\pi_\varepsilon(\cdot\mid S_t).
$$

## SARSA update

For a nonterminal transition, the SARSA target is

$$
y_t=R_{t+1}+\gamma Q(S_{t+1},A_{t+1}),
$$

where \(A_{t+1}\) is sampled from the same epsilon-greedy policy used to interact with the environment.

The TD error is

$$
\delta_t=y_t-Q(S_t,A_t),
$$

and the update is

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\delta_t.
$$

Equivalently,

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha
\left[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\right].
$$

For a terminal transition, there is no next decision, so the target becomes

$$
y_t=R_{t+1}.
$$

## Why SARSA is on-policy

The same epsilon-greedy policy both generates experience and defines the update target. If the policy selects an exploratory next action, the value of that exploratory action appears in the target.

SARSA therefore estimates action values for the policy it is currently following:

$$
Q\approx Q_{\pi_\varepsilon}.
$$

As the Q-table changes, the epsilon-greedy policy changes with it, so learning and policy improvement occur together.

## SARSA versus Q-learning

Both algorithms use the same current transition. The difference is the next-state value inside the target:

| Algorithm | Nonterminal target | What it learns toward |
| --- | --- | --- |
| Q-learning | \(R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')\) | Greedy target policy |
| SARSA | \(R_{t+1}+\gamma Q(S_{t+1},A_{t+1})\) | Current epsilon-greedy policy |

Q-learning asks, “What if I take the best estimated action next?” SARSA asks, “What is the value of the action my current policy actually selected next?”

## Episode flow

SARSA must choose the first action before entering the main loop. Each nonterminal step then follows this order:

1. execute the current pair \((S_t,A_t)\);
2. observe \(R_{t+1}\) and \(S_{t+1}\);
3. choose \(A_{t+1}\) epsilon-greedily;
4. update \(Q(S_t,A_t)\) using \(Q(S_{t+1},A_{t+1})\);
5. shift `state = next_state` and `action = next_action`;
6. repeat.

This action-carrying structure ensures that the action used in the target is the same action executed on the following iteration.

## Implementation map

| Function | Purpose |
| --- | --- |
| `step(...)` | Applies actions and returns the next state, reward, and terminal flag. |
| `greedy_action(state, Q)` | Selects the highest-valued action. |
| `choose_action(state, Q, epsilon)` | Samples from the epsilon-greedy behavior policy. |
| `sarsa_update(...)` | Computes the SARSA target and TD error, then updates one table entry. |
| `run_sarsa_episode(...)` | Maintains the SARSA interaction sequence and learns online. |
| `evaluation(...)` | Evaluates the final greedy policy without updating the Q-table. |

The training loop records total reward, episode length, and the TD error from every step.

## Run the notebook

Install the dependencies:

```bash
python -m pip install jupyter numpy
```

Then run from the repository root:

```bash
jupyter notebook "06 SARSA/SARSA-gridworld.ipynb"
```

## Expected behavior

For this simple environment, moving right is the shortest route to the only reward. After sufficient training:

$$
Q(s,\text{RIGHT})>Q(s,\text{LEFT}),
\qquad s\in\{0,1,2,3\}.
$$

The value of moving from state `3` directly into the goal should approach

$$
Q(3,\text{RIGHT})=1.
$$

Because SARSA learns values for an epsilon-greedy policy rather than directly using a greedy maximum in every target, its nonterminal estimates need not exactly match Q-learning's values while \(\varepsilon>0\).

During greedy evaluation, the learned policy should reach the goal in four steps and receive total reward `1`. The notebook copies the Q-table before evaluation and verifies that evaluation does not change it.

## Tests

The notebook checks:

- normal and terminal environment transitions;
- the learned ordering between `RIGHT` and `LEFT`;
- the value of the direct goal transition;
- read-only evaluation of the learned table.

Exact training traces vary because exploration and tie-breaking are random and no fixed seed is used.

## Main takeaway

SARSA's essential idea is contained in the next action:

$$
\boxed{\text{learn from the action the behavior policy actually selected}}
$$

That single choice changes Q-learning's off-policy target into an on-policy TD-control update.

