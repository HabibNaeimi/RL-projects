# Level 21: PPO with Global Gradient-Norm Clipping

This level extends the vectorized PPO trainer from Level 20 with **global L2 gradient-norm clipping** for both the actor and the critic.

The previous level monitored how far the current policy moved from the rollout policy and could skip remaining PPO epochs when the approximate KL became too large. Level 21 adds a different safeguard: after each minibatch backward pass, it measures the complete gradient norm of one network and rescales that network's gradients when the norm exceeds a configured limit.

Gradient clipping does not change rollout collection, GAE, PPO's clipped surrogate, or KL early stopping. It controls the size of each optimizer input before the parameters are updated.

## Learning objectives

- Compute one global L2 norm across all gradients of a network.
- Ignore parameters whose gradients are `None`.
- Support one-use parameter generators safely.
- Distinguish the norm before clipping from the norm after clipping.
- Clip actor and critic gradients independently.
- Place clipping after `backward()` and before `optimizer.step()`.
- Understand the difference between gradient clipping, PPO ratio clipping, and KL early stopping.
- Track mean pre-clipping norms and clipping frequencies across minibatches.
- Preserve final-epoch metric semantics when KL early stopping is active.

## What changes from Level 20?

| Component | Level 20 | Level 21 |
| --- | --- | --- |
| Vectorized collection | Four independent CartPole environments | Unchanged |
| GAE and fixed targets | Boundary-aware vectorized GAE | Unchanged |
| PPO objective | Clipped probability-ratio surrogate | Unchanged |
| KL monitoring | Full-rollout diagnostic after each epoch | Unchanged |
| KL early stopping | Can skip remaining epochs | Unchanged |
| Actor gradients | Passed directly to Adam | Globally clipped before every actor step |
| Critic gradients | Passed directly to Adam | Globally clipped before every critic step |
| Gradient diagnostics | Not reported | Pre-clipping means and clipping fractions |

## Project files

- `ppo_grad_clip_cartpole.py`: vectorized PPO, KL monitoring, gradient clipping, training, and evaluation.
- `test_ppo_grad_clip_cartpole.py`: minimal deterministic tests focused on the new gradient-clipping behavior.
- `ppo_grad_clip_cartpole.png`: generated training-return plot.

## Existing PPO pipeline

The actor represents the categorical policy $\pi_\theta(a\mid s)$ with a `4 → 32 → 2` network. The critic estimates $V_\phi(s)$ with a separate `4 → 32 → 1` network. They have separate parameters and separate Adam optimizers.

The vector environment collects

$$
N=4,\qquad T=256,\qquad B=TN=1024
$$

transitions per rollout. Here, $N$ is the number of independent environments, $T$ is the number of steps per environment, and $B$ is the flattened training-batch size.

For environment $n$ at time $t$, the termination-aware TD error is

$$
\delta_{t,n}
=R_{t+1,n}
+\gamma(1-d_{t,n})V_{\phi_0}(S^+_{t,n})
-V_{\phi_0}(S_{t,n}),
$$

where $d_{t,n}=1$ only for true task termination and $S^+_{t,n}$ is the actual next state before autoreset.

The independent GAE recurrence is

$$
\hat A_{t,n}
=\delta_{t,n}
+\gamma\lambda(1-e_{t,n})\hat A_{t+1,n},
\qquad
\hat A_{T,n}=0,
$$

where $e_{t,n}=1$ for either termination or truncation. Therefore, truncation allows value bootstrapping but stops the advantage trace at the episode boundary.

After GAE, the time and environment dimensions are flattened without changing state–action–target alignment. The actor advantages are normalized once over the complete rollout, while the critic uses fixed, unnormalized value targets.

## PPO losses

For flattened sample $i$, define the selected-action probability ratio

$$
r_i
=\exp\left(
\log\pi_\theta(A_i\mid S_i)
-\log\pi_{\theta_0}(A_i\mid S_i)
\right),
$$

where $\theta_0$ is the rollout policy and $\theta$ is the current policy.

The ratio used by the clipped surrogate is

$$
\bar r_i
=\min\left(
\max\left(r_i,1-\epsilon\right),
1+\epsilon
\right).
$$

For a minibatch $M_j$ containing $m_j$ samples, the actor minimizes

$$
L_\theta
=-\frac{1}{m_j}\sum_{i\in M_j}
\min\left(
r_i\tilde A_i,
\bar r_i\tilde A_i
\right)
-\frac{\beta}{m_j}\sum_{i\in M_j}
H\left(\pi_\theta(\cdot\mid S_i)\right).
$$

The critic minimizes

$$
L_\phi
=\frac{1}{m_j}\sum_{i\in M_j}
\left(V_\phi(S_i)-\hat V_i\right)^2.
$$

The gradient-clipping mechanism is applied after these scalar losses have produced gradients. It does not alter the loss formulas themselves.

## What is a global gradient norm?

Suppose a network contains parameter tensors indexed by $k$. Let $g_{k,q}$ be one scalar gradient inside parameter tensor $k$. The global L2 gradient norm is

$$
G
=\sqrt{
\sum_k\sum_q g_{k,q}^2
}.
$$

This is one norm over all available gradients in the network. It is not:

- one norm per parameter tensor;
- the sum or average of separate layer norms;
- the norm of model parameter values;
- the actor and critic gradients combined into one value.

The actor and critic therefore have separate pre-clipping norms:

$$
G_\theta
=\sqrt{\sum_k\sum_q g_{\theta,k,q}^2},
\qquad
G_\phi
=\sqrt{\sum_k\sum_q g_{\phi,k,q}^2}.
$$

Parameters whose `.grad` is `None` do not contribute. If none of the supplied parameters has a gradient, the norm is `0.0`.

## Global norm clipping

Let $C>0$ be the maximum allowed gradient norm. Conceptually, clipping uses the shared scale factor

$$
\alpha
=\min\left(
1,
\frac{C}{G+\eta}
\right),
$$

where $\eta$ is a small numerical stabilizer. Every gradient in that network is then rescaled by the same factor:

$$
g'_{k,q}=\alpha g_{k,q}.
$$

Therefore:

- if $G\leq C$, the gradients are left effectively unchanged;
- if $G>C$, their directions and relative proportions are preserved while their joint magnitude is reduced;
- the post-clipping global norm is at most approximately $C$, subject to floating-point tolerance.

For example, suppose the combined gradients are

$$
(3,4,12).
$$

Their pre-clipping norm is

$$
G=\sqrt{3^2+4^2+12^2}=13.
$$

With $C=5$, the common scale is approximately $5/13$, producing

$$
\left(
\frac{15}{13},
\frac{20}{13},
\frac{60}{13}
\right),
$$

whose norm is approximately $5$.

Clipping all components with one factor matters. Independently restricting individual gradient values would be element-wise clipping, which is a different operation and can change the gradient direction.

## Helper functions

### `get_gradient_norm(parameters)`

This educational helper manually computes the global L2 norm. It first converts the parameter iterable to a list so that a one-use generator such as `policy.parameters()` can be traversed safely. It then:

1. ignores parameters with no gradient;
2. flattens and combines the remaining gradients;
3. calculates their L2 norm;
4. returns the result as a Python float;
5. returns `0.0` if no gradients exist.

The function runs under `torch.no_grad()` because the diagnostic must not create another computation graph.

### `clip_gradients(parameters, max_grad_norm)`

This wrapper converts the iterable to a list, validates the threshold, and calls `torch.nn.utils.clip_grad_norm_` with:

```python
norm_type=2.0
error_if_nonfinite=True
```

The threshold must be finite and strictly positive. Conceptually, the validation condition is:

```python
if max_grad_norm <= 0 or not math.isfinite(max_grad_norm):
    raise ValueError(...)
```

The helper returns the **pre-clipping** global norm as a Python float. This point is important: a returned value greater than `max_grad_norm` is evidence that clipping was needed, not evidence that clipping failed.

With no gradients, the helper returns `0.0` and leaves the parameters unchanged.

## Correct placement in the update loop

Each network follows this order for every minibatch:

| Order | Actor | Critic |
| ---: | --- | --- |
| 1 | `actor_optimizer.zero_grad()` | `critic_optimizer.zero_grad()` |
| 2 | `actor_loss.backward()` | `critic_loss.backward()` |
| 3 | Clip `policy.parameters()` | Clip `value_network.parameters()` |
| 4 | Record the returned pre-clipping norm | Record the returned pre-clipping norm |
| 5 | `actor_optimizer.step()` | `critic_optimizer.step()` |

In code, the essential structure is:

```python
actor_optimizer.zero_grad()
actor_loss.backward()
actor_pre_clip_norm = clip_gradients(
    policy.parameters(),
    max_grad_norm,
)
actor_optimizer.step()

critic_optimizer.zero_grad()
critic_loss.backward()
critic_pre_clip_norm = clip_gradients(
    value_network.parameters(),
    max_grad_norm,
)
critic_optimizer.step()
```

Clipping before `backward()` would find no current gradients. Clipping after `step()` would be too late because the optimizer would already have used the unbounded gradients.

The actor and critic are clipped separately because they use different losses, parameter sets, and optimizers. Their norm values are not expected to have similar scales.

## Three different safeguards

Level 21 now contains three mechanisms called “clipping” or “control,” but they act on different quantities and at different times.

| Mechanism | Quantity controlled | When applied | Scope | Main purpose |
| --- | --- | --- | --- | --- |
| PPO ratio clipping | Surrogate objective through $r_i$ and $\bar r_i$ | During actor-loss calculation | Actor samples | Limits incentives from large selected-action ratio changes |
| KL early stopping | Approximate old-to-new policy divergence | After a complete PPO epoch | Full rollout and remaining epochs | Prevents further reuse after excessive policy drift |
| Gradient-norm clipping | Backpropagated gradients | After each backward pass, before each step | Actor and critic separately | Limits the magnitude supplied to each optimizer step |

None replaces the others. In particular, a small gradient norm does not guarantee a small policy KL, and PPO ratio clipping does not guarantee a bounded parameter-gradient norm.

## Gradient diagnostics

The gradient lists are reset at the start of every PPO epoch. One value and one Boolean flag must be appended for **every minibatch** after each backward pass.

If the final completed epoch contains $J$ minibatches, let $G_{\theta,j}$ and $G_{\phi,j}$ be the actor and critic pre-clipping norms for minibatch $j$. The reported means are

$$
\bar G_\theta
=\frac{1}{J}\sum_{j=1}^{J}G_{\theta,j},
\qquad
\bar G_\phi
=\frac{1}{J}\sum_{j=1}^{J}G_{\phi,j}.
$$

The clipping fractions are

$$
f_\theta
=\frac{1}{J}\sum_{j=1}^{J}I\left(G_{\theta,j}>C\right),
\qquad
f_\phi
=\frac{1}{J}\sum_{j=1}^{J}I\left(G_{\phi,j}>C\right),
$$

where $I$ equals one when the condition is true and zero otherwise.

The four returned diagnostics are:

| Metric | Meaning |
| --- | --- |
| `mean_actor_grad_norm` | Arithmetic mean of actor pre-clipping norms from the final completed epoch |
| `mean_critic_grad_norm` | Arithmetic mean of critic pre-clipping norms from the final completed epoch |
| `actor_grad_clip_fraction` | Fraction of final-epoch actor steps whose pre-clipping norm exceeded $C$ |
| `critic_grad_clip_fraction` | Fraction of final-epoch critic steps whose pre-clipping norm exceeded $C$ |

These metrics are averaged by minibatch update, not weighted by the number of samples in each minibatch. Each norm describes one optimizer step. This differs from actor and critic loss aggregation, which remains sample-weighted.

With the default batch and minibatch sizes:

$$
J=\frac{1024}{64}=16.
$$

The clipping fractions can therefore take values in increments of $1/16$ during a complete default epoch. A fraction of `0.25`, for example, means that four of the sixteen corresponding optimizer steps required clipping.

If KL early stopping occurs, these diagnostics still describe every minibatch in the **last fully completed epoch**. They do not combine gradient norms across all completed epochs.

## KL monitoring remains active

After each complete minibatch epoch, the actor is re-evaluated on the full flattened rollout. With

$$
\Delta_i
=\log\pi_\theta(A_i\mid S_i)
-\log\pi_{\theta_0}(A_i\mid S_i),
$$

the nonnegative approximate KL diagnostic is

$$
D_+
=\frac{1}{B}\sum_{i=1}^{B}
\left(e^{\Delta_i}-1-\Delta_i\right).
$$

Remaining epochs are skipped when

$$
D_+>c\tau,
$$

provided at least one epoch remains. The defaults are $\tau=0.01$ and $c=1.5$, giving a stopping threshold of `0.015`.

Gradient clipping occurs inside every completed minibatch, while KL is checked only after the entire epoch. Completed parameter updates are never rolled back.

## Optimizer-step accounting

For rollout size $B$, minibatch size $M$, and $Q$ completed epochs:

$$
J=\left\lceil\frac{B}{M}\right\rceil,
\qquad
N_\theta=N_\phi=QJ.
$$

With the default configuration, each epoch contains 16 actor steps and 16 critic steps. If all four epochs finish, `optimizer_steps` is `64`. This field counts actor/critic update pairs rather than adding the two optimizer counts together.

KL early stopping can reduce optimizer steps, but it does not change the number of interactions already collected. Each rollout still contains 1024 environment transitions.

## Training configuration

| Parameter | Default |
| --- | --- |
| Environment | `CartPole-v1` |
| Vector environment / autoreset | `SyncVectorEnv` / `SAME_STEP` |
| Number of environments | `4` |
| Steps per environment | `256` |
| Rollout batch size | `1024` |
| Minibatch size | `64` |
| Maximum PPO epochs | `4` |
| Training updates | `100` |
| Maximum gradient norm $C$ | `0.5` |
| Discount factor $\gamma$ | `0.99` |
| GAE parameter $\lambda$ | `0.95` |
| PPO ratio limit $\epsilon$ | `0.2` |
| Entropy coefficient $\beta$ | `0.01` |
| Target KL $\tau$ | `0.01` |
| KL multiplier $c$ | `1.5` |
| Actor learning rate | `3e-4` |
| Critic learning rate | `1e-3` |
| Hidden dimension | `32` |
| Evaluation episodes | `20` |
| Base seed | `32268` |

The same maximum norm is used for both networks, but clipping and diagnostics remain independent.

## Implementation map

| Component | Responsibility |
| --- | --- |
| `PolicyNetwork`, `ValueNetwork` | Separate categorical actor and scalar critic |
| `collect_vector_rollout(...)` | Vectorized experience and correct final-observation handling |
| `compute_gae(...)` | Independent boundary-aware GAE and fixed value targets |
| `flatten_vector_batch(...)` | Aligned flattening after temporal target construction |
| `calculate_ppo_terms(...)` | Probability ratios and conservative PPO surrogate |
| `compute_kl_diagnostics(...)` | Full-rollout signed and nonnegative KL estimates |
| `should_ppo_stop_for_kl(...)` | Complete-epoch early-stopping decision |
| `get_gradient_norm(...)` | Manual global L2 norm for understanding and verification |
| `clip_gradients(...)` | Validated in-place global norm clipping and pre-clipping norm return |
| `update_ppo_from_rollout(...)` | Minibatch losses, backward passes, clipping, steps, and final-epoch metrics |
| `train(...)` | Repeated rollouts, episode histories, interaction counts, and reporting |
| `evaluate_policy(...)` | Complete stochastic-policy evaluation episodes |

## Interpreting the output

- `mean_actor_grad_norm` and `mean_critic_grad_norm` are measured **before** clipping.
- A reported mean above `0.5` is possible even though every optimizer step receives clipped gradients.
- `actor_grad_clip_fraction` and `critic_grad_clip_fraction` must lie in `[0, 1]`.
- A clipping fraction of `0.0` means no pre-clipping norm in the reported epoch exceeded the threshold.
- A clipping fraction of `1.0` means every corresponding minibatch exceeded the threshold.
- A higher critic norm than actor norm is not automatically a problem because their losses and scales differ.
- `early_stopped=False` is normal when the KL remains below its threshold or no epochs remain to skip.
- `optimizer_steps=64` and `update_epoch=4` mean that the current rollout completed every planned minibatch update.

One seeded training run reached:

| Result | Value |
| --- | ---: |
| Environment interactions | `102400` |
| Completed training episodes | `581` |
| Mean of final 50 training returns | `487.98` |
| Mean evaluation return over 20 episodes | `457.15` |
| Update-100 approximate KL | `0.0016689253970980644` |
| Update-100 signed approximate KL | `0.00024457863764837384` |
| Update-100 completed PPO epochs | `4` |
| Update-100 optimizer steps | `64` |
| Update-100 early stopping | `False` |

The learning results are examples from one stochastic run, not guaranteed targets. Gradient diagnostics are intentionally omitted from this example table unless they were collected with one norm and clipping flag per minibatch.

## Minimal test coverage

The focused `test_ppo_grad_clip_cartpole.py` contains four tests:

1. `get_gradient_norm(...)` calculates one global L2 norm, ignores missing gradients, and accepts a one-use generator.
2. Both helpers handle a parameter collection with no gradients without changing model parameters.
3. `clip_gradients(...)` returns the pre-clipping norm and globally rescales the `(3, 4, 12)` example from norm `13` to norm `5`.
4. The real PPO update clips after each backward pass and before each optimizer step, then reports final-epoch actor and critic gradient means and fractions over every minibatch.

The fourth test also checks the exact operation order:

```text
actor clip → actor step → critic clip → critic step
```

repeated once for every minibatch.

The suite is intentionally limited to the new Level 21 behavior; earlier PPO, GAE, vector-environment, and KL mechanics were tested in previous levels.

## Run the project

Install dependencies in your Python environment:

```bash
python -m pip install numpy "gymnasium>=1.1" torch matplotlib pytest
```

From the Level 21 directory:

```bash
python ppo_grad_clip_cartpole.py
python -m pytest -q test_ppo_grad_clip_cartpole.py
```

The test module imports `ppo_grad_clip_cartpole`, so run it where that module is importable.

## Connection to language-model RL and RLVR

Gradient clipping transfers directly to larger policy-optimization settings. Language-model updates can produce occasional large gradient norms because rewards, advantages, sequence lengths, and token counts vary across batches.

The conceptual order remains the same:

1. compute the policy objective;
2. backpropagate;
3. obtain the usable gradients;
4. clip their global norm;
5. perform the optimizer step.

Mixed-precision training introduces an additional requirement: scaled gradients must be unscaled before their norm is clipped. This CartPole implementation does not use automatic mixed precision, distributed gradient synchronization, or token-level masking.

Gradient clipping also remains distinct from language-model reference-policy regularization. It constrains an optimizer input, not the divergence from a rollout or reference policy.

## Scope and limitations

Included: vectorized collection, independent GAE, shuffled minibatch PPO, entropy regularization, KL monitoring and early stopping, separate actor/critic global L2 clipping, and final-epoch gradient diagnostics.

Not included: separate actor and critic norm thresholds, per-layer clipping, element-wise value clipping, gradient accumulation, automatic mixed precision, distributed gradient synchronization, adaptive norm thresholds, value-function clipping, learning-rate schedules, checkpoints, or explicit GPU device management.

## Main takeaway

Gradient clipping belongs between backpropagation and the optimizer step:

`zero_grad()` → `backward()` → `clip_gradients(...)` → `optimizer.step()`

The returned norm is measured before clipping, and the final-epoch means and fractions must include every minibatch. This produces a useful stability diagnostic while ensuring that neither optimizer receives a gradient vector above the configured global-norm limit.
