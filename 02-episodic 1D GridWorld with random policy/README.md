# Level 2: Episodic 1D GridWorld with a Random Policy

This level moves from a stateless bandit to an episodic environment with states, actions, terminal transitions, trajectories, and delayed returns. The policy is intentionally random: the objective is to understand how experience is represented before introducing value learning.

## Learning objectives

- Implement the agent–environment interaction loop.
- Distinguish the current state from the next state.
- generate a complete episode and store its trajectory;
- stop an episode at a terminal state or a maximum step limit;
- compute a discounted return for every transition;
- understand temporal credit assignment.

## Environment

The environment contains five states:

| State | Meaning |
| ---: | --- |
| `0` | Start state |
| `1`, `2`, `3` | Intermediate states |
| `4` | Goal and terminal state |

There are two actions:

| Constant | Value | Effect |
| --- | ---: | --- |
| `LEFT` | `0` | Move one state left. |
| `RIGHT` | `1` | Move one state right. |

Movement is clipped at the boundaries, so taking `LEFT` in state `0` leaves the agent in state `0`. Reaching state `4` returns reward `1` and terminates the episode; all other transitions return reward `0`.

Formally, the reward is

$$
R_{t+1}=
\begin{cases}
1, & S_{t+1}=4,\\
0, & \text{otherwise}.
\end{cases}
$$

## Random policy

The fixed policy selects `LEFT` or `RIGHT` with equal probability:

$$
\pi(a\mid s)=0.5, \qquad a\in\{\text{LEFT},\text{RIGHT}\}.
$$

The policy does not learn in this level. Its only purpose is to generate experience.

## Episode trajectory

An episode begins in state `0` and continues until the goal is reached or `max_steps=100` is exhausted. Each stored transition has the form

$$
(S_t,A_t,R_{t+1}),
$$

represented in Python as

```python
(state, action, reward)
```

The next state is used to continue the interaction loop, but it is not stored directly in the tuple.

## Discounted returns

The return from time step $t$ is the discounted sum of all subsequent rewards:

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1},
$$

where $T$ is the end of the episode and $\gamma\in[0,1]$ is the discount factor.

The same quantity can be computed backward using the recursion

$$
G_t=R_{t+1}+\gamma G_{t+1},
$$

with the last return initialized from the final reward. The notebook uses $\gamma=0.9$.

Because the only positive reward occurs at the goal, transitions closer to that reward receive larger returns. For an episode that reaches the goal in four moves, the rewards and returns are:

| Time step | Reward | Return with $\gamma=0.9$ |
| ---: | ---: | ---: |
| 0 | 0 | $0.9^3=0.729$ |
| 1 | 0 | $0.9^2=0.81$ |
| 2 | 0 | $0.9$ |
| 3 | 1 | $1$ |

This is temporal credit assignment: an earlier action can receive credit for a reward observed several steps later.

## Implementation map

| Function | Purpose |
| --- | --- |
| `step(state, action)` | Applies an action and returns `(next_state, reward, terminated)`. |
| `choose_action()` | Samples an action from the fixed random policy. |
| `run_episode(max_steps=100)` | Generates one complete trajectory. |
| `compute_returns(trajectory, gamma=0.9)` | Computes one return aligned with each trajectory element. |

The backward return calculation is efficient because each previously computed return is reused instead of repeatedly summing the remaining rewards.

## Run the notebook

From the repository root:

```bash
jupyter notebook "02-episodic 1D GridWorld with random policy/1d-gridworld.ipynb"
```

The implementation uses Python's standard library; Jupyter is the only notebook dependency.

## Tests and inspection

The notebook checks important transitions, including:

- `LEFT` at the left boundary;
- a normal move to the right;
- entry into the terminal goal state;
- a normal move to the left.

It also prints every trajectory element beside its corresponding return. These two lists must have equal lengths, and the return at index $t$ must describe the future reward following the transition at index $t$.

## Expected behavior

Episode length varies because the policy is random. Some episodes reach the goal quickly, while others repeatedly move left or remain at the boundary. If the goal is not reached within 100 steps, the trajectory is safely truncated.

No values or policy parameters are updated in this level. The output is experience plus correctly aligned returns—the data required by Monte Carlo learning in the next level.

## Main takeaway

In an episodic RL problem, the immediate reward is not the complete learning signal. The return $G_t$ connects each earlier state–action decision to the rewards that follow it.

