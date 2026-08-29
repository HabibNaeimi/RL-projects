# REINFORCE with a Learned Value Baseline on CartPole

Stage E: Policy gradients / REINFORCE – ~3 levels
    Level 12: REINFORCE with a Learned Value Baseline

Implementing REINFORCE with a Learned Value Baseline using torch and Gymnasium on CartPole-v1 environment.
We're doing on-policy learning here, because the training data comes from the policy currently being optimized.

------------------------------------

* **Algorithm:**
observation sₜ
├── policy network → log π(aₜ|sₜ)
└── value network  → V(sₜ)

rewards → return Gₜ

advantage = Gₜ − V(sₜ)

policy loss ← log-probability × detached advantage
value loss  ← prediction error between V(sₜ) and Gₜ


* The policy predicts one score for every possible action: state → [action-0 logit, action-1 logit]
* The value network predicts one expected return for the state: state → expected return

* During training, we give each episode a different but reproducible environment seed: episode_seed = base_seed + episode_index


* Note that here:

    * the policy is continually changing;
    * the value network is chasing changing return targets;
    * only one episode is used per update;
    * advantages are not normalized;
    * the policy learning rate remains relatively aggressive.




* **Tests**
    * run tests using:  python -m pytest -q



