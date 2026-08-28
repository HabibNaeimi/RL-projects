# CartPole with One REINFORCE Policy Update

Stage D: Neural-network RL basics – ~3 levels
    Level 10: One REINFORCE Policy Update

Implementing One REINFORCE Policy Update policy using torch and Gymnasium on CartPole-v1: environment.

------------------------------------

* **Algorithm:**
    * reset environment
    * while episode is not finished:
        * convert observation to tensor
        * obtain action logits from policy
        * construct categorical distribution
        * sample action
        * calculate log-probability of sampled action
        * execute action
        * store observation, action, reward, and log-probability


* **Tests**
    * run tests using:  python -m pytest -q



