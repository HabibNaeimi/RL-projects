# Online Actor–Critic with One-Step TD Error

Stage F: Actor-Critic + PPO – ~4-5 levels
    Level 14: Online Actor–Critic with One-Step TD Error

------------------------------------

* **GOALS:**
    * Turn the policy into an actor.
    * Turn the value network into a critic.
    * Replace full-episode Monte Carlo returns with a one-step TD target.
    * Update both networks after every environment step.
    * Correctly distinguish terminal states from time-limit truncation.
    * Understand the bias–variance tradeoff between REINFORCE and actor–critic.


* **Algorithm:**
    * reset environment

    * while episode is not over:
        * sample action from actor
        * predict current value V(state)

        * execute action
        * observe reward and next state

        * predict V(next_state) without retaining its gradient
        * construct TD target
        * calculate TD error

        * actor loss = -log_prob * detached TD error
        *              - entropy coefficient * entropy

        * critic loss = TD error squared

        * update actor
        * update critic

        * move to next state


* **Tests**
    * run tests using:  python -m pytest -q



