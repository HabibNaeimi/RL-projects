# Level 7 — Gymnasium Basics with FrozenLake

This level introduces the Gymnasium environment API through random interaction with `FrozenLake-v1`. There is deliberately no learning algorithm yet: the objective is to understand observations, actions, rewards, episode boundaries, and the return values of `reset()` and `step()`.

## Learning objectives

- Create a standard Gymnasium environment.
- Inspect discrete observation and action spaces.
- Use `env.reset()` and `env.step(action)` correctly.
- Distinguish `terminated` from `truncated`.
- Run multiple episodes with a random policy.
- Interpret sparse rewards and episode statistics.

## Environment

The notebook creates

```python
env = gym.make("FrozenLake-v1", is_slippery=False)
```

Setting `is_slippery=False` makes each chosen action deterministic. The standard 4 × 4 map contains 16 discrete observations:

| 0: `S` | 1: `F` | 2: `F` | 3: `F` |
| --- | --- | --- | --- |
| 4: `F` | 5: `H` | 6: `F` | 7: `H` |
| 8: `F` | 9: `F` | 10: `F` | 11: `H` |
| 12: `H` | 13: `F` | 14: `F` | 15: `G` |

where:

- `S` is the start;
- `F` is a safe frozen tile;
- `H` is a hole;
- `G` is the goal.

The spaces reported by Gymnasium are:

```text
observation_space = Discrete(16)
action_space      = Discrete(4)
```

The action encoding is:

| Action | Meaning |
| ---: | --- |
| `0` | Left |
| `1` | Down |
| `2` | Right |
| `3` | Up |

## Reward and episode endings

FrozenLake uses a sparse reward:

$$
R_{t+1}=
\begin{cases}
1, & S_{t+1}=\text{goal},\\
0, & \text{otherwise}.
\end{cases}
$$

An episode can end in two different ways:

| Flag | Meaning |
| --- | --- |
| `terminated` | The environment reached a genuine terminal state, such as a hole or the goal. |
| `truncated` | An external limit ended the episode, such as Gymnasium's time limit. |

Interaction stops when either flag is true:

$$
\text{done}=\text{terminated}\lor\text{truncated}.
$$

The helper also limits its own loop to `max_steps=100`.

## Gymnasium interaction API

Resetting returns the initial observation and auxiliary information:

```python
observation, info = env.reset()
```

Executing an action returns five values:

```python
next_observation, reward, terminated, truncated, info = env.step(action)
```

The current observation must then be replaced with `next_observation` before the next decision.

## Random policy

Actions are sampled directly from the environment's action space:

```python
action = env.action_space.sample()
```

This corresponds to the uniform policy

$$
\pi(a\mid s)=\frac{1}{4},
\qquad a\in\{0,1,2,3\}.
$$

The policy does not use the observation and does not improve between episodes.

## Episode statistics

`do_random_episode()` returns:

- total episode reward;
- number of executed steps.

The undiscounted episode return is

$$
G_0=\sum_{t=0}^{T-1}R_{t+1}.
$$

Because the only possible positive reward is the goal reward, each episode return is either `0` or `1`. Therefore, the mean reward over many episodes is also the empirical success rate.

The notebook runs 20 random episodes and reports all lengths, unique lengths, mean length, rewards, and mean reward.

## Implementation map

| Component | Purpose |
| --- | --- |
| `gym.make(...)` | Creates deterministic `FrozenLake-v1`. |
| `env.observation_space` | Describes the 16 possible observations. |
| `env.action_space` | Describes and samples the four possible actions. |
| `do_random_episode(...)` | Runs one complete random episode. |
| `terminated or truncated` | Detects whether interaction must stop. |
| NumPy summaries | Report unique and mean rewards and lengths. |

## Run the notebook

Install the dependencies:

```bash
python -m pip install jupyter numpy gymnasium
```

From the repository root:

```bash
jupyter notebook "07 Gymnasium Basics/frozenlake_interaction.ipynb"
```

## Expected behavior

Random episodes usually fall into a hole before reaching the goal. The saved notebook run produced 20 zero-reward episodes; this is a plausible outcome, not evidence that the environment is broken.

Results vary because the environment and random action space are not explicitly seeded in this notebook.

## Main takeaway

Gymnasium standardizes the agent–environment loop:

$$
S_t\rightarrow A_t\rightarrow(R_{t+1},S_{t+1},\text{terminated},\text{truncated}).
$$

Random interaction supplies experience, but without a learning update it does not improve the policy. Level 8 adds Q-learning to the same environment.

