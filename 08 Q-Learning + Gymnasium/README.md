# Q-Learning: FrozenLake

Stage C: Gymnasium — ~2 levels
    Level 8: Q-Learning with Gymnasium: FrozenLake

Implementing off-policy Q-Learning using Gymnasium on FrozenLake environment.

------------------------------------

Code Structure:
Gymnasium environment
        ↓
observation
        ↓
agent chooses action from Q
        ↓
env.step(action)
        ↓
reward + next observation
        ↓
Q-learning update
        ↓
repeat

------------------------------------

We're using FrozenLake, a small discrete environment. Conceptually it looks like:

S F F F
F H F H
F F F H
H F F G

where:	​
    S = start
    F = safe frozen tile
    H = hole
    G = goal

Agent actions:
    0 = LEFT
    1 = DOWN
    2 = RIGHT
    3 = UP

reward = 1 if it reaches the goal.
Also Falling into a hole ends the episode.



