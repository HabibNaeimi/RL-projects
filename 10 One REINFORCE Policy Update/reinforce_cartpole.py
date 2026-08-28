import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical

GAMMA = 0.99
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


def compute_discounted_returns(rewards, gamma=GAMMA):
    """
    Args:
        rewards: rewards ordered from t=0 to t=T-1
        gamma: discount factor
    Returns:
        A float32 tensor of shape (T,), ordered from G_0 to G_{T-1}.    
    """
    discounted_returns = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        discounted_returns.append(running_return)
    
    discounted_returns.reverse()
    discounted_tensor = torch.tensor(discounted_returns, dtype=torch.float32)
    return discounted_tensor


def compute_reinforce_loss(log_probs, returns):
    """
    Args:
        log_probs: list of scalar tensors connected to the policy
        returns: tensor of shape (T,)

    Returns:
        Scalar REINFORCE loss tensor    
    """
    # creating a tensor of shape (T,) while preserving autograd.
    log_probs_tensor = torch.stack(log_probs)

    assert log_probs_tensor.shape == returns.shape   # dimention checking 

    loss = -(log_probs_tensor * returns).sum()
    return loss


def policy_update(optimizer, episode, gamma=GAMMA):

    returns = compute_discounted_returns(episode['rewards'], gamma)
    loss = compute_reinforce_loss(
        episode["log_probs"],
        returns,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.item()



def main():

    env = gym.make("CartPole-v1")                    # creating environment.
    env.action_space.seed(SEED)

    obs_dim = env.observation_space.shape[0]         # Observation space dimentions
    n_actions = env.action_space.n                   # Action counts

    print("Observation dimension:", obs_dim)
    print("Number of actions:", n_actions)

    policy = PolicyNetwork(obs_dim, n_actions)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=1e-2
    )

    episode_results = collect_episode(env, policy, seed=SEED)

    parameters_before = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]

    loss_value = policy_update(optimizer, episode_results, GAMMA)

    parameters_after = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]

    
    parameters_changed =  any(
        not torch.allclose(before, after)
        for before, after in zip(parameters_before, parameters_after)
    )

    print("Episode return:", episode_results["episode_return"])
    print("Episode length:", episode_results["episode_length"])
    print("Loss:", loss_value)
    print("Parameters changed:", parameters_changed)

    env.close()


if __name__ == "__main__":
    main()


