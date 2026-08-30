# REINFORCE with an Entropy Bonus on CartPole

Stage E: Policy gradients / REINFORCE – ~3 levels
    Level 13: REINFORCE with an Entropy Bonus

Implementing REINFORCE with a Learned Value Baseline and an Entropy Bonus using torch and Gymnasium on CartPole-v1 environment.
We're doing on-policy learning here, because the training data comes from the policy currently being optimized.

------------------------------------

For two actions, entropy lies between: 0 and log(2)=0.693. H=0: almost deterministic policy. H=0.693: probabilities exactly [0.5, 0.5].




* **Tests**
    * run tests using:  python -m pytest -q



