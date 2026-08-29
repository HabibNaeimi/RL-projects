# CartPole with Training REINFORCE Policy 

Stage D: Neural-network RL basics – ~3 levels
    Level 11: Train REINFORCE Across Episodes

Implementing REINFORCE Policy Training using torch and Gymnasium on CartPole-v1: environment.
We're doing on-policy learning here, because the training data comes from the policy currently being optimized.

------------------------------------

* **Algorithm:**
    * current policy
        * → collect fresh episode
        * → calculate REINFORCE loss
        * → update policy
        * → discard episode
        * → repeat with updated policy



During training, we give each episode a different but reproducible environment seed: episode_seed = base_seed + episode_index


* We are using:
    * one episode per update;
    * raw discounted returns;
    * no baseline;
    * no advantage normalization.


So the result should not be great! it demonstrates why policy-gradient variance matters.


* **Tests**
    * run tests using:  python -m pytest -q



