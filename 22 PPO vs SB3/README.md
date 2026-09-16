# Level 22 — PPO with Stable-Baselines3

Level 22 rebuilds the CartPole PPO experiment with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) (SB3). The purpose is not to replace the PPO implementation from Levels 16–21. It is to express the same experiment through a maintained RL library, inspect what the library does internally, and compare its behavior with the from-scratch agent under the same interaction budget.

The main question is:

> If the environment, rollout size, training budget, network width, and most PPO hyperparameters stay fixed, how does the SB3 implementation compare with the custom PPO implementation?

## Learning objectives

By the end of this level, you should be able to:

- configure an SB3 PPO agent instead of accepting all library defaults;
- map familiar PPO concepts to SB3 constructor arguments;
- collect rollouts from multiple environments;
- record completed-episode statistics through a callback;
- distinguish stochastic policy evaluation from deterministic evaluation;
- explain the important differences between the custom and SB3 update procedures; and
- compare implementations without confusing a framework comparison with an algorithm comparison.

## Project files

```text
22 PPO with Stable-Baselines3/
├── ppo_sb3_cartpole.py
├── README.md
└── ppo_sb3_cartpole.png       # recommended learning-curve artifact
```

The plot is only produced if plotting is implemented in the training script. It should show completed-episode returns and a moving average, using the same presentation as the earlier PPO levels.

## Installation

Install the packages inside the environment used for this repository:

```bash
python -m pip install stable-baselines3 gymnasium torch numpy matplotlib pytest
```

The package name used by `pip` contains a hyphen, while the Python import contains underscores and uses the plural word `baselines`:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
```

Neither `torchvision` nor `torchaudio` is required by this CartPole experiment.

## Required implementation checks

Before training, verify these details in `ppo_sb3_cartpole.py`:

1. All SB3 imports use `stable_baselines3`.
2. The callback constructor calls `super().__init__(verbose=0)`.
3. Evaluation converts the predicted action to a scalar before passing it to Gymnasium.
4. The fewer-than-50-episodes check actually raises its exception.
5. `clip_range_vf=None` is passed explicitly so the absence of value clipping is visible in the configuration.

The corresponding lines should look like this:

```python
class EpisodeReturnCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.episode_returns = []
        self.episode_lengths = []

    def _on_step(self):
        for info in self.locals["infos"]:
            episode = info.get("episode")
            if episode is not None:
                self.episode_returns.append(float(episode["r"]))
                self.episode_lengths.append(int(episode["l"]))
        return True
```

During evaluation:

```python
action, _ = model.predict(observation, deterministic=deterministic)
action_scalar = int(np.asarray(action).item())
observation, reward, terminated, truncated, _ = evaluation_env.step(
    action_scalar
)
```

The training-statistics guard should be:

```python
if number_of_completed_episodes < 50:
    raise ValueError("Training completed fewer than 50 episodes.")
```

## Experiment configuration

| Setting | Value |
|---|---:|
| Environment | `CartPole-v1` |
| Parallel environments | 4 |
| Steps per environment and rollout | 256 |
| Rollout size | 1,024 transitions |
| PPO updates | 100 |
| Total environment interactions | 102,400 |
| Minibatch size | 64 |
| Optimization epochs per rollout | 4 |
| Learning rate | 0.0003 |
| Discount factor | 0.99 |
| GAE factor | 0.95 |
| PPO clip range | 0.2 |
| Entropy coefficient | 0.01 |
| Value-loss coefficient | 0.5 |
| Maximum gradient norm | 0.5 |
| Target KL | 0.01 |
| Value-function clipping | Disabled |
| Evaluation episodes | 20 per evaluation mode |
| Device | CPU |
| Seed | 32,268 |

The rollout contains all transitions collected by all environment copies:

$$
N = 4 \times 256 = 1024.
$$

The complete training budget is:

$$
T = 100 \times 1024 = 102400.
$$

Each full optimization epoch contains:

$$
K = \frac{1024}{64} = 16
$$

minibatches. Therefore, each rollout permits at most:

$$
4 \times 16 = 64
$$

optimizer steps. Early stopping based on KL divergence can reduce this number.

SB3 documents `total_timesteps` as a lower bound in the general case. Here it is exactly divisible by the rollout size, and the callback never requests early termination, so the intended final count is exactly 102,400 interactions.

## Vectorized environment

```python
training_env = make_vec_env(
    ENV_ID,
    n_envs=NUM_ENVS,
    seed=SEED,
)
```

`make_vec_env` creates a vectorized environment and wraps each copy with SB3's episode monitor. With no custom vector class, it uses `DummyVecEnv`. The monitor adds an `episode` entry to an environment's `info` dictionary when an episode finishes. This allows the callback to collect complete returns and lengths without reconstructing them from partial rollouts. See the official [`make_vec_env` documentation](https://stable-baselines3.readthedocs.io/en/master/common/env_util.html) and [`Monitor` documentation](https://stable-baselines3.readthedocs.io/en/master/common/monitor.html).

The four environments may finish at different times. A PPO rollout boundary is therefore not necessarily an episode boundary. Training statistics should come from completed episodes, not from sums over arbitrary rollout slices.

## Policy and value networks

The policy configuration is:

```python
policy_kwargs = {
    "net_arch": {
        "pi": [32],
        "vf": [32],
    },
    "activation_fn": nn.Tanh,
    "ortho_init": False,
}
```

For CartPole's four-dimensional observation and two discrete actions, this creates:

- a policy branch: `4 → 32 → 2 action logits`;
- a value branch: `4 → 32 → 1 state value`; and
- `tanh` activation after each 32-unit hidden layer.

The policy and value branches have separate hidden layers, but SB3 owns them inside one actor-critic policy object and optimizes them with one optimizer. The default feature extractor for vector observations only flattens the input and has no trainable parameters. The [SB3 policy documentation](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html) explains how `net_arch` defines these branches.

Setting `ortho_init=False` makes the architecture closer to the earlier PyTorch implementation, but it does not guarantee identical initial parameters or identical training trajectories.

## PPO calculations

SB3 performs the familiar PPO calculations internally. The notation below connects the framework configuration to the equations implemented in previous levels.

### One-step temporal-difference residual

For transition $t$, let $D_{t+1}=1$ when no bootstrap value should be used and $D_{t+1}=0$ otherwise. The residual is:

$$
\delta_t
= R_{t+1}
+ \gamma(1-D_{t+1})V_{\phi_0}(S_{t+1})
- V_{\phi_0}(S_t).
$$

### Generalized advantage estimation

GAE is computed backward through the rollout:

$$
\hat{A}_t
= \delta_t
+ \gamma\lambda(1-D_{t+1})\hat{A}_{t+1}.
$$

The value target is:

$$
\hat{G}_t = \hat{A}_t + V_{\phi_0}(S_t).
$$

Here, $\gamma=0.99$ and $\lambda=0.95$.

### Minibatch advantage normalization

For a minibatch $M$ containing $m$ samples, current SB3 normalizes the sampled advantages inside that minibatch:

$$
\mu_M = \frac{1}{m}\sum_{i\in M}\hat{A}_i,
$$

$$
\sigma_M
= \sqrt{
\frac{1}{m-1}\sum_{i\in M}(\hat{A}_i-\mu_M)^2
},
$$

$$
\tilde{A}_i
= \frac{\hat{A}_i-\mu_M}{\sigma_M+10^{-8}}.
$$

This is an important comparison detail: the custom implementation normalized advantages once across the full rollout, while SB3 normalizes each sampled minibatch during training.

### Probability ratio

Let $\ell_i(\theta)$ be the current log probability of the stored action and let $\theta_0$ denote the rollout policy. The importance ratio is:

$$
r_i(\theta)
= \exp\left(\ell_i(\theta)-\ell_i(\theta_0)\right)
= \frac{\pi_\theta(A_i\mid S_i)}{\pi_{\theta_0}(A_i\mid S_i)}.
$$

Define the clipped ratio without relying on a custom math function:

$$
\bar{r}_i
= \min\left(
\max\left(r_i(\theta),1-\epsilon\right),
1+\epsilon
\right).
$$

With $\epsilon=0.2$, the policy loss is:

$$
L_\pi
= -\frac{1}{m}\sum_{i\in M}
\min\left(
r_i(\theta)\tilde{A}_i,
\bar{r}_i\tilde{A}_i
\right).
$$

### Value loss

Because `clip_range_vf=None`, this experiment does not clip the value prediction:

$$
L_V
= \frac{1}{m}\sum_{i\in M}
\left(V_\phi(S_i)-\hat{G}_i\right)^2.
$$

### Entropy bonus and combined loss

For CartPole's two-action categorical policy, sample $i$ has entropy:

$$
H_i
= -\sum_{a=0}^{1}
\pi_\theta(a\mid S_i)
\log\pi_\theta(a\mid S_i).
$$

SB3 minimizes one combined loss with one optimizer:

$$
L
= L_\pi
+ c_V L_V
- \frac{\beta}{m}\sum_{i\in M}H_i,
$$

where $c_V=0.5$ and $\beta=0.01$. After backpropagation, the global gradient norm across the policy object's parameters is limited to 0.5.

### Approximate KL early stopping

For each minibatch, define:

$$
\Delta_i = \ell_i(\theta)-\ell_i(\theta_0).
$$

The approximation used by SB3 is:

$$
\hat{K}
= \frac{1}{m}\sum_{i\in M}
\left[
\exp(\Delta_i)-1-\Delta_i
\right].
$$

With target $\tau=0.01$, optimization stops early when:

$$
\hat{K} > 1.5\tau = 0.015.
$$

In the current SB3 implementation, this check occurs for each minibatch before that minibatch's optimizer step. If the threshold is exceeded, that minibatch is not applied and the remaining epochs are skipped. This is more fine-grained than checking one full-rollout estimate after a complete epoch. These details can be verified in the official [SB3 PPO source](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py).

## Model construction

The configuration should be explicit so the comparison does not depend on unnoticed defaults:

```python
model = PPO(
    "MlpPolicy",
    training_env,
    policy_kwargs=policy_kwargs,
    learning_rate=LEARNING_RATE,
    n_steps=STEPS_PER_ENV,
    batch_size=MINIBATCH_SIZE,
    n_epochs=UPDATE_EPOCHS,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_range=CLIP_EPSILON,
    clip_range_vf=None,
    normalize_advantage=True,
    ent_coef=ENTROPY_COEF,
    vf_coef=VALUE_COEF,
    max_grad_norm=MAX_GRAD_NORM,
    target_kl=TARGET_KL,
    seed=seed,
    verbose=1,
    device="cpu",
)
```

## Episode-statistics callback

`EpisodeReturnCallback` is called after each vector-environment step. It inspects every environment's `info` dictionary and records an entry only when the monitor reports a completed episode.

The callback intentionally records both return and length. In `CartPole-v1`, every nonterminal step gives reward 1, so return and episode length usually match. Keeping both metrics is still useful because this equality does not hold in most RL environments.

SB3 exposes training state, including `self.locals`, to custom callbacks through `BaseCallback`. See the official [callback guide](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html).

## Training

```python
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callback,
    log_interval=10,
)
```

After training, report:

- number of completed training episodes;
- mean return over the first 50 completed episodes;
- mean return over the final 50 completed episodes;
- mean episode length over the first 50 episodes; and
- mean episode length over the final 50 episodes.

These measurements show whether learning occurred and make the SB3 run directly comparable with the custom PPO run.

## Evaluation

Evaluation uses a separate, non-vectorized `CartPole-v1` environment and does not update the model. Episode $j$ starts with seed:

$$
s_j = 32268 + 100 + j,
\qquad j=0,1,\ldots,19.
$$

The same 20 environment seeds are used for both modes:

| Mode | `deterministic` | Action selection | Purpose |
|---|---:|---|---|
| Stochastic | `False` | Sample from the learned categorical policy | Primary comparison with the custom stochastic policy |
| Deterministic | `True` | Choose the highest-probability action | Diagnostic of the policy's greedy behavior |

Report the mean and population standard deviation for both sets of returns. A high deterministic score does not replace stochastic evaluation: it answers a different question.

## Comparison with the custom PPO agent

The matched settings make this a useful implementation comparison, but it is not a perfectly controlled one. Several update details remain different.

| Detail | Custom PPO, Level 21 | SB3 PPO, Level 22 |
|---|---|---|
| Training interactions | 102,400 | 102,400 |
| Rollout geometry | 4 environments × 256 steps | 4 environments × 256 steps |
| Network branches | Separate actor and critic modules | Separate branches inside one policy object |
| Learning rates | Actor: 0.0003; critic: 0.001 | One shared rate: 0.0003 |
| Optimizers | Separate actor and critic optimizers | One combined optimizer |
| Advantage normalization | Once over the full rollout | Separately for each minibatch |
| Value loss | Independent critic update | Weighted by 0.5 in the combined loss |
| Value clipping | None | None |
| KL check | Full-rollout estimate after an epoch | Minibatch estimate before its optimizer step |
| Gradient clipping | Actor and critic handled separately | One global norm across the policy object |
| Episode bookkeeping | Custom vector loop | Monitor wrapper and callback |

Consequently, a performance difference cannot be attributed to one factor. Level 22 tests the complete SB3 training recipe configured to resemble the custom experiment.

## Results

Do not invent SB3 measurements before running the corrected script. Record the actual output here after a successful run:

| Measurement | Custom PPO, Level 21 | SB3 PPO, Level 22 |
|---|---:|---:|
| Training interactions | 102,400 | 102,400 |
| Final-50 training return | 487.98 | Record after running |
| Stochastic evaluation mean | 457.15 | Record after running |
| Stochastic evaluation standard deviation | — | Record after running |
| Deterministic evaluation mean | — | Record after running |
| Deterministic evaluation standard deviation | — | Record after running |

When interpreting the table, consider both evaluation variance and training variance. One seed is enough for this learning exercise, but it is not enough to claim that one implementation is generally superior.

## Recommended learning curve

Plot the completed-episode returns in their completion order and overlay a 50-episode moving average. For returns $R_1,R_2,\ldots$, the moving average at episode $k$ is:

$$
\bar{R}_k
= \frac{1}{50}\sum_{i=k-49}^{k}R_i,
\qquad k\geq 50.
$$

Save the figure as `ppo_sb3_cartpole.png`. The raw curve shows episode-to-episode variation, while the moving average makes the learning trend easier to compare with Level 21.

## Run the experiment

From this level's directory:

```bash
python -m py_compile ppo_sb3_cartpole.py
python ppo_sb3_cartpole.py
```

The first command catches syntax errors without starting training. The second command performs the full 102,400-interaction run.

If a test file is added, run it before the full experiment:

```bash
pytest -q
```

Useful automated checks include:

- rollout size equals `NUM_ENVS * STEPS_PER_ENV`;
- the total budget equals `ROLLOUT_SIZE * NUM_UPDATES`;
- the rollout size is divisible by the minibatch size;
- model fields match the constants in the script;
- the callback ignores unfinished episodes and records completed ones;
- stochastic and deterministic evaluation use the requested flag;
- evaluation seeds advance once per episode;
- evaluation closes its environment; and
- the script raises an error if fewer than 50 training episodes finish.

## Expected observations

A correct run should satisfy the following qualitative checks:

- training reaches exactly 102,400 environment interactions;
- many more than 50 CartPole episodes finish;
- later training returns are substantially higher than early returns;
- both evaluation modes produce finite returns between 1 and 500;
- the deterministic and stochastic scores may differ even on identical environment seeds; and
- repeated runs can still differ unless every source of nondeterminism is controlled.

These are sanity checks, not guaranteed numerical results.

## Why this level matters for RLVR and LLM training

Large-scale reinforcement learning rarely uses a training loop written entirely from scratch. It uses frameworks that own rollout storage, batching, loss construction, optimization, logging, and distributed execution. This level develops the habit that matters in those systems: map each library argument back to an equation and inspect important implementation details instead of treating the framework as a black box.

The CartPole policy is much smaller than a language model, and SB3 is not an LLM-training framework. However, the workflow transfers:

1. define an interaction budget;
2. collect behavior from a frozen rollout policy;
3. compute rewards, advantages, and value targets;
4. perform repeated minibatch updates;
5. limit destructive policy movement;
6. log training and evaluation separately; and
7. compare implementations under a clearly documented configuration.

## Key takeaway

Level 22 moves from implementing PPO mechanics to auditing and configuring a production-style PPO implementation. Stable-Baselines3 removes much of the training-loop code, but it does not remove the need to understand rollout geometry, advantage normalization, loss coefficients, KL early stopping, evaluation modes, or reproducibility.

The library is useful precisely because you now understand what it is doing for you.

## References

- [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Stable-Baselines3 PPO source](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py)
- [Stable-Baselines3 custom policy guide](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
- [Stable-Baselines3 callback guide](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)
- [Stable-Baselines3 environment utilities](https://stable-baselines3.readthedocs.io/en/master/common/env_util.html)
- [Gymnasium CartPole documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
