import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical

SEED = 32268
np.random.seed(SEED)
torch.manual_seed(SEED)

class PolicyNetwork(nn.Module):
    def __init__(
            self,
            obs_dim,
            n_actions,
            hidden_dim=32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_actions),
        )

    def forward(self, observation):
        """
        returns the logits produced by self.network
        """
        return self.network(observation)

def choose_action(policy, observation):
    """
        Returns:
            action: Python int
            log_prob: scalar tensor that remains connected to the policy
            probs: detached tensor used only for inspection
    """

    observation_tensor = torch.from_numpy(observation).float()
    logits = policy(observation_tensor)                                 # Creating logits from policy
    distribution = Categorical(logits=logits)                           # Creating distributions
    action_tensor = distribution.sample()                               # Sampling from policy, not env!
    log_prob = distribution.log_prob(action_tensor)
    probs = distribution.probs.detach()                                 # detached tensor
    return action_tensor.item(), log_prob, probs

def collect_episode(env, policy, seed=SEED):

    observation, info = env.reset(seed=seed)
    observations = []
    actions =[]
    rewards = []
    log_probs_list = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, log_probs, _ = choose_action(policy, observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        observations.append(observation.copy())
        log_probs_list.append(log_probs)
        observation = next_observation
    
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "log_probs": log_probs_list,
        "episode_return": float(sum(rewards)),
        "episode_length": len(rewards)
    }

def main():

    env = gym.make("CartPole-v1")                    # creating environment.
    env.action_space.seed(SEED)

    obs_dim = env.observation_space.shape[0]         # Observation space dimentions
    n_actions = env.action_space.n                   # Action counts

    print("Observation dimension:", obs_dim)
    print("Number of actions:", n_actions)

    policy = PolicyNetwork(obs_dim, n_actions)
    episode_results = collect_episode(env, policy)

    print("Episode return:", episode_results["episode_return"])
    print("Episode length:", episode_results["episode_length"])
    print('episode_results:')
    print(episode_results)

    env.close()


if __name__ == "__main__":
    main()


