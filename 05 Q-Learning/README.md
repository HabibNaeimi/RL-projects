# Q-Learning: Learning from One Step at a Time

Stages A–B: RL mechanics + tabular RL — ~6 levels
    Level 5: Q-Learning: Learning from One Step at a Time

Implementing every-visit Monte Carlo prediction with policy improvement.

Q-learning:

take ONE action
      ↓
observe reward + next state
      ↓
immediately update Q(s, a)
      ↓
continue episode


Using an estimate of the future to update an estimate of the present -> Bootstrapping