# Level 14: Online Actor–Critic with One-Step TD(0)

This level replaces complete-episode Monte Carlo targets with a bootstrapped one-step TD target. The policy becomes the actor, the value network becomes the critic, and both networks are updated after every environment transition.

## Learning objectives

- Construct a one-step TD target.
- Use TD error as an online advantage estimate.
- Separate actor and critic gradient paths.
- Update both networks once per environment step.
- Bootstrap across a time-limit truncation but not a true terminal state.
- Understand the bias–variance tradeoff between REINFORCE and actor–critic.

## Actor and critic

The actor represents the stochastic policy

$$
\pi_\theta(a\mid s),
$$

using the same `4 → 32 → 2` network as the previous levels.

The critic estimates

$$
V_\phi(s),
$$

using a separate `4 → 32 → 1` network. The networks do not share parameters and have separate Adam optimizers.

## One-step TD target

For a continuing transition, the critic target is

$$
y_t=R_{t+1}+\gamma V_\phi(S_{t+1}).
$$

For a true terminal transition,

$$
y_t=R_{t+1}.
$$

Let \(d_t=1\) when the transition truly terminates and \(d_t=0\) otherwise. The implementation combines both cases as

$$
y_t=R_{t+1}
+\gamma(1-d_t)V_\phi(S_{t+1})^{\mathrm{detach}}.
$$

The next-state value is detached because it is a fixed bootstrap target for this update. Gradients should train the current prediction \(V_\phi(S_t)\), not move the target through \(V_\phi(S_{t+1})\).

## TD error

The one-step temporal-difference error is

$$
\delta_t=y_t-V_\phi(S_t).
$$

It serves two roles:

- critic prediction error;
- actor advantage estimate.

A positive TD error means the transition was better than the critic expected; a negative error means it was worse.

## Actor loss

The actor uses the detached TD error:

$$
L_{\text{actor}}
=-\log\pi_\theta(A_t\mid S_t)
\delta_t^{\mathrm{detach}}
-\beta\mathcal{H}
\left(\pi_\theta(\cdot\mid S_t)\right).
$$

Detachment prevents actor-loss backpropagation from updating the critic. The entropy coefficient is \(\beta=0.01\).

## Critic loss

The scalar critic loss is

$$
L_{\text{critic}}=\delta_t^2
=\left(y_t-V_\phi(S_t)\right)^2.
$$

This loss updates the critic toward the detached TD target.

## Online update sequence

For every environment step:

1. sample \(A_t\) from the actor;
2. predict \(V_\phi(S_t)\);
3. execute the action and observe \(R_{t+1},S_{t+1}\);
4. predict and detach \(V_\phi(S_{t+1})\) through the TD target;
5. compute \(y_t\) and \(\delta_t\);
6. compute actor and critic losses;
7. clear both optimizers' gradients;
8. backpropagate both losses;
9. step both optimizers;
10. continue from \(S_{t+1}\).

Unlike Levels 10–13, learning does not wait for the episode to finish.

## Termination versus truncation

The interaction loop ends when

$$
\text{terminated}\lor\text{truncated}
$$

is true, but the TD target stops bootstrapping only for `terminated=True`.

| Ending | Bootstrap? | Reason |
| --- | --- | --- |
| CartPole failure (`terminated`) | No | The underlying task reached a terminal state. |
| 500-step time limit (`truncated`) | Yes | The external cutoff does not make the underlying state terminal. |

This distinction avoids treating a time limit as if the future value were necessarily zero.

## Training configuration

| Parameter | Value |
| --- | ---: |
| Training episodes | `1000` |
| Discount factor \(\gamma\) | `0.99` |
| Entropy coefficient \(\beta\) | `0.01` |
| Actor learning rate | `1e-3` |
| Critic learning rate | `1e-3` |
| Update frequency | Every environment step |
| Report interval | `50` episodes |
| Moving-average window | `50` episodes |
| Evaluation episodes | `20` |
| Base seed | `32268` |

Per episode, the code records return, length, mean actor loss, mean critic loss, mean absolute TD error, and mean entropy. Training history stores one aggregate value per episode. The return curve is saved as:

```text
actor_critic_td0_cartpole.png
```

## Monte Carlo versus TD(0)

| Property | REINFORCE with baseline | Online actor–critic TD(0) |
| --- | --- | --- |
| Critic target | Full return \(G_t\) | \(R_{t+1}+\gamma V(S_{t+1})\) |
| Update timing | After the complete episode | After every step |
| Bootstrapping | No | Yes |
| Typical target variance | Higher | Lower |
| Typical target bias | Lower | Higher because the target uses an estimate |

This is the central bias–variance tradeoff introduced by temporal-difference learning.

## Implementation map

| Function | Purpose |
| --- | --- |
| `compute_td_target(...)` | Builds a detached, termination-aware one-step target. |
| `compute_actor_critic_losses(...)` | Creates separated actor and critic losses. |
| `train_one_actor_critic_episode(...)` | Updates both networks after each transition. |
| `train_actor_critic(...)` | Repeats online learning and records episode metrics. |
| `collect_episode(...)` | Collects trajectories for evaluation, not training. |
| `evaluate_policy(...)` | Samples from the learned actor under `torch.no_grad()`. |

## Run the project

Install the dependencies:

```bash
python -m pip install numpy gymnasium torch matplotlib pytest
```

From the Level 14 directory:

```bash
python actor_critic_td0_cartpole.py
python -m pytest -q
```

## Tests

The tests verify:

- a continuing target of \(1+0.9(4)=4.6\);
- a terminal target of `1.0`;
- detachment of both TD targets;
- exact actor loss, critic loss, and TD error values;
- no critic gradient from actor-loss backpropagation;
- correct log-probability and entropy gradients;
- parameter changes in both networks during one training episode;
- finite episode metrics and CartPole's return–length equality.

## Scope

This is a single-environment, one-step online actor–critic implementation. It is not A2C, PPO, an n-step method, or GAE: there are no rollout batches, parallel workers, clipped objectives, importance ratios, or multi-step advantage estimates.

Evaluation remains stochastic because actions are sampled from the actor rather than selected with `argmax`.

## Main takeaway

The critic now teaches the actor immediately through the TD error:

$$
\boxed{
\delta_t=R_{t+1}+\gamma V_\phi(S_{t+1})-V_\phi(S_t)
}.
$$

This changes the learning pattern from complete-episode Monte Carlo updates to online bootstrapped learning.
