# FrozenLake Interactions in Gymnasium

Stage C: Gymnasium — ~2 levels
    Level 7: Gymnasium: Using a Standard RL Environment

This level is all about learning basics of Gymnasium API.

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
the current policy is completely randome. 

ABout all zero rewards: we are executing a completely random policy on FrozenLake with only 20 episodes. 
    An episode can terminate because the agent reached the reward=1 or falls into a hole, so most random trajectories are bad.
    We can conclude that Random interaction alone doesn't improve anything!